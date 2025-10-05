from __future__ import annotations
import os
import argparse
import json
import time
import glob
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.models.feature_extractor import FeatureExtractor
from training.models.triplet_net import TripletNet
from training.losses.triplet import cosine_distance
from training.dataset_utils import (
    separate_deepfashion_data,
    DeepFashionTripletDataset,
    train_val_split,  # optional, in case you want a real val loader
)

# -------------------- validation (unchanged logic) --------------------
def validate_model(model, dataloader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for anchor, positive, negative in tqdm(dataloader, desc="Validation"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            running_loss += loss.item()

    avg_loss = running_loss / max(1, len(dataloader))
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

# -------------------- model factory (matches your notebook) --------------------
def make_model(
    model_name: str = "vit_b_16",
    pooling: str = "mean_pool",
    pretrained: bool = True,
    embed_dim: int = 768,
    use_projection: bool = True,
) -> TripletNet:
    feat = FeatureExtractor(model_name=model_name, pooling=pooling, pretrained=pretrained)
    if use_projection:
        proj = nn.Linear(feat.output_size, embed_dim)
        feature_stack = nn.Sequential(feat, proj)  # proj(feat(x))
    else:
        feature_stack = feat
    model = TripletNet(feature_extractor=feature_stack)
    return model

# -------------------- dataloaders (uses your dataset utils) --------------------
def build_loaders_from_root(
    data_root: str,
    batch_size: int = 24,
    num_workers: int = 2,
    virtual_epoch_size: int = 2000,
    img_size: int = 224,
    make_val_loader: bool = False,
):
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_dict, query_dict, gallery_dict = separate_deepfashion_data(data_root)

    # optional: split train into train/val by classes for a val loader
    val_loader = None
    if make_val_loader:
        tr_split, va_split = train_val_split(train_dict, val_ratio=0.2, seed=42)
        train_ds = DeepFashionTripletDataset(tr_split, transform=tfm, virtual_epoch_size=virtual_epoch_size)
        val_ds   = DeepFashionTripletDataset(va_split, transform=tfm, virtual_epoch_size=max(1, virtual_epoch_size//5))
        val_loader = DataLoader(val_ds, batch_size=max(1, batch_size//2), shuffle=False,
                                num_workers=max(1, num_workers//2), pin_memory=False, persistent_workers=False)
    else:
        train_ds = DeepFashionTripletDataset(train_dict, transform=tfm, virtual_epoch_size=virtual_epoch_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False, persistent_workers=False)

    return train_loader, val_loader, (train_dict, query_dict, gallery_dict)

# -------------------- your training loop (kept same; only safe guards added) --------------------
def train_model_semi_hard(
    model,
    train_loader,
    val_loader=None,
    criterion=None,
    optimizer=None,
    distance_function=None,
    num_epochs=10,
    device=None,
    online_mining_start_epoch=3,
    checkpoint_dir="checkpoints",
    checkpoint_name="triplet_model",
    resume_from_checkpoint=True,
    early_stop_patience=10,
    top_k_best=3,
    min_epochs_before_saving=2,
    tune_reporter=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_latest.pth")
    best_ckpt_pattern = os.path.join(checkpoint_dir, f"{checkpoint_name}_val*.pth")

    train_losses, val_losses = [], []
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    margin_warmup_epochs = 5
    max_margin = 0.35
    mining_schedule_epochs = 5
    mining_phase_entered = False

    if resume_from_checkpoint and os.path.exists(latest_ckpt_path):
        print(f"Loading model from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_recording=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        mining_phase_entered = checkpoint.get('mining_phase_entered', False)
        print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        is_online_mining_phase = epoch >= online_mining_start_epoch
        if is_online_mining_phase:
            if not mining_phase_entered:
                epochs_no_improve = 0
                mining_phase_entered = True
                print("Entering semi-hard mining phase. Resetting best validation loss tracking.")
                best_val_loss = float('inf')
            mining_progress = min(1.0, (epoch - online_mining_start_epoch + 1) / mining_schedule_epochs)
            semi_hard_ratio = mining_progress
            mode_desc = f"Online Mining ({semi_hard_ratio*100:.0f}%)"
        else:
            semi_hard_ratio = 0.0
            mode_desc = "Standard Training"

        epsilon = 1e-6
        current_margin = max(epsilon, max_margin * min(1.0, (epoch + 1) / margin_warmup_epochs)) if margin_warmup_epochs > 0 else max_margin
        criterion.margin = current_margin

        semi_hard_count = 0
        fallback_total = 0

        for i, (anchor, positive, negative) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} {mode_desc} (margin={current_margin:.3f})")
        ):
            anchor = anchor.to(device); positive = positive.to(device); negative = negative.to(device)
            optimizer.zero_grad()

            a_emb, p_emb, n_emb0 = model(anchor, positive, negative)
            for name, t in {"anchor": a_emb, "positive": p_emb, "negative": n_emb0}.items():
                assert t.ndim == 2, f"{name} embeddings should be (B, D), got {t.shape}"

            if torch.isnan(a_emb).any() or torch.isinf(a_emb).any():     continue
            if torch.isnan(p_emb).any() or torch.isinf(p_emb).any():     continue
            if torch.isnan(n_emb0).any() or torch.isinf(n_emb0).any():   continue

            a_emb = F.normalize(a_emb, p=2, dim=-1)
            p_emb = F.normalize(p_emb, p=2, dim=-1)
            n_emb0 = F.normalize(n_emb0, p=2, dim=-1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if semi_hard_ratio > 0:
                B = a_emb.size(0)
                num_to_mine = int(B * semi_hard_ratio)

                idx = torch.randperm(B, device=device)
                a_perm, p_perm, n_perm = a_emb[idx], p_emb[idx], n_emb0[idx]
                online_a   = a_perm[:num_to_mine]
                online_p   = p_perm[:num_to_mine]
                std_a      = a_perm[num_to_mine:]
                std_p      = p_perm[num_to_mine:]
                std_n      = n_perm[num_to_mine:]

                mined_negatives = []
                if num_to_mine > 0:
                    cos_ap = (online_a * online_p).sum(dim=-1).clamp(-1.0, 1.0)
                    dist_positive = 1.0 - cos_ap
                    cos_an = online_a @ n_emb0.T
                    cos_an = cos_an.clamp(-1.0, 1.0)
                    dist_matrix = 1.0 - cos_an

                    for j in range(num_to_mine):
                        lower = dist_positive[j]
                        upper = lower + current_margin
                        mask = (dist_matrix[j] > lower) & (dist_matrix[j] < upper)
                        cand = torch.nonzero(mask, as_tuple=False).flatten()
                        if cand.numel() > 0:
                            pick = int(torch.randint(0, cand.numel(), (1,), device=device).item())
                            chosen_idx = int(cand[pick].item())
                            mined_negatives.append(n_emb0[chosen_idx])
                            semi_hard_count += 1
                        else:
                            hardest_idx = int(torch.argmin(dist_matrix[j]).item())
                            mined_negatives.append(n_emb0[hardest_idx])
                            fallback_total += 1

                D = n_emb0.size(-1)
                online_mined_neg = torch.stack(mined_negatives) if mined_negatives else torch.empty((0, D), device=device, dtype=n_emb0.dtype)
                negative_embeddings = torch.cat([online_mined_neg, std_n], dim=0)
                anchor_final        = torch.cat([online_a,       std_a],  dim=0)
                positive_final      = torch.cat([online_p,       std_p],  dim=0)

                loss = criterion(anchor_final, positive_final, negative_embeddings)
            else:
                loss = criterion(a_emb, p_emb, n_emb0)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}")
        if is_online_mining_phase:
            print(f"Epoch {epoch+1} Summary: Semi-hard negatives selected: {semi_hard_count}, Fallback negatives selected: {fallback_total}")
        train_losses.append(avg_train_loss)

        if val_loader is not None:
            val_loss = validate_model(model, val_loader, criterion, device=device)
            val_losses.append(val_loss)
        else:
            val_loss = None

        if tune_reporter is not None:
            payload = {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "current_margin": float(criterion.margin),
                "semi_hard_ratio": float(semi_hard_ratio),
            }
            if val_loss is not None and np.isfinite(val_loss):
                payload["val_loss"] = float(val_loss)
            tune_reporter(**payload)

        # save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'mining_phase_entered': mining_phase_entered
        }, latest_ckpt_path)

        if is_online_mining_phase and (val_loss is not None) and (val_loss < best_val_loss) and (epoch + 1) >= min_epochs_before_saving:
            best_val_loss = val_loss
            timestamp = int(time.time())
            best_model_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_val{val_loss:.4f}_{timestamp}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, best_model_path)
            print(f"Validation improved. Saved best model: {best_model_path}")

            # keep only top-k best
            all_checkpoints = glob.glob(best_ckpt_pattern)
            val_losses_and_paths = []
            for path in all_checkpoints:
                try:
                    val_loss_str = path.split('_val')[1].split('_')[0]
                    val_losses_and_paths.append((float(val_loss_str), path))
                except Exception:
                    continue
            val_losses_and_paths.sort(key=lambda x: x[0])
            if len(val_losses_and_paths) > top_k_best:
                for _, to_del in val_losses_and_paths[top_k_best:]:
                    if os.path.exists(to_del):
                        os.remove(to_del)
                        print(f"Deleted old checkpoint: {to_del}")
        else:
            epochs_no_improve += 1
            if val_loader is not None:
                print(f"No improvement. Patience: {epochs_no_improve}/{early_stop_patience}")

        if val_loader is not None and early_stop_patience is not None:
            if is_online_mining_phase and epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("Finished Training!")
    return train_losses, val_losses

# -------------------- Ray trainable (wired to make_model) --------------------
def build_loaders_for_ray(cfg):
    return build_loaders_from_root(
        data_root=cfg["data_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        virtual_epoch_size=cfg["virtual_epoch_size"],
        img_size=cfg["img_size"],
        make_val_loader=True,  # give Ray a val loader so it can compare
    )[:2]  # return (train_loader, val_loader)

def ray_train(config):
    from ray.air import session
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_loaders_for_ray(config)

    model = make_model(
        model_name=config["model_name"],
        pooling=config["pooling"],
        pretrained=config.get("pretrained", False),
        embed_dim=config["embed_dim"],
        use_projection=config.get("use_projection", True),
    ).to(device)

    criterion = nn.TripletMarginWithDistanceLoss(
        margin=config["max_margin"],
        distance_function=cosine_distance,
    )

    if config["opt"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["opt"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"], nesterov=True)

    ckpt_dir = os.path.join("/tmp", "raytune_ckpts", os.path.basename(session.get_trial_dir()))
    os.makedirs(ckpt_dir, exist_ok=True)

    def reporter(**metrics): session.report(metrics)

    try:
        train_model_semi_hard(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            distance_function=cosine_distance,
            num_epochs=config["epochs"],
            device=device,
            online_mining_start_epoch=config["online_mining_start_epoch"],
            checkpoint_dir=ckpt_dir,
            checkpoint_name="triplet_model",
            resume_from_checkpoint=False,
            early_stop_patience=config["early_stop_patience"],
            top_k_best=0,
            min_epochs_before_saving=10**9,
            tune_reporter=reporter,
        )
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

# -------------------- runners --------------------
def run_small_scale_tune(args):
    import ray
    from ray import air, tune
    from ray.tune.schedulers import ASHAScheduler

    scheduler = ASHAScheduler(metric=args.ray_metric, mode=args.ray_mode)
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # NOTE: JSON can’t hold tune.loguniform() etc. Use plain values or build param_space in code.
    param_space = {
        "seed": 42,
        "data_root": args.data_root,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "virtual_epoch_size": args.virtual_epoch_size,
        "model_name": args.model_name,
        "pooling": args.pooling,
        "pretrained": args.pretrained,
        "embed_dim": args.embed_dim,
        "use_projection": not args.no_projection,
        "opt": "adam",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_margin": args.margin,
        "epochs": args.ray_stop_iters,
        "early_stop_patience": 5,
        "online_mining_start_epoch": 2,
    }

    tuner = tune.Tuner(
        ray_train,
        run_config=air.RunConfig(
            name=args.ray_name,
            storage_path=args.ray_dir,
            checkpoint_config=air.CheckpointConfig(num_to_keep=2),
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.ray_num_samples,
            metric=args.ray_metric,
            mode=args.ray_mode,
        ),
        param_space=param_space,
    )
    results = tuner.fit()
    print("Ray Tune finished.")
    try:
        best = results.get_best_result(metric=args.ray_metric, mode=args.ray_mode)
        print("Best:", best.metrics)
    except Exception as e:
        print("Unable to fetch best result:", e)

def run_full_training(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_model(
        model_name=args.model_name,
        pooling=args.pooling,
        pretrained=args.pretrained,
        embed_dim=args.embed_dim,
        use_projection=not args.no_projection,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, val_loader, splits = build_loaders_from_root(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        virtual_epoch_size=args.virtual_epoch_size,
        img_size=args.img_size,
        make_val_loader=True,
    )

    criterion = nn.TripletMarginWithDistanceLoss(
        margin=args.margin,
        distance_function=cosine_distance,
    )

    train_model_semi_hard(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        distance_function=cosine_distance,
        num_epochs=args.epochs,
        device=device,
        online_mining_start_epoch=3,
        checkpoint_dir=args.ckpt_dir,
        checkpoint_name="triplet_model",
        resume_from_checkpoint=True,
        early_stop_patience=10,
        top_k_best=3,
        min_epochs_before_saving=2,
        tune_reporter=None,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    final_path = os.path.join(args.ckpt_dir, "latest.pth")
    torch.save({"state_dict": model.state_dict()}, final_path)
    print(f"Saved: {final_path}")

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "train"], default="tune")
    ap.add_argument("--data_root", default="/home/ec2-user/fashion-recommender/gallery_images")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--virtual_epoch_size", type=int, default=2000)

    ap.add_argument("--model_name", default="vit_b_16", choices=["vit_b_16","resnet18","resnet50"])
    ap.add_argument("--pooling", default="mean_pool", choices=["mean_pool","cls"])
    ap.add_argument("--pretrained", action="store_true", default=False)
    ap.add_argument("--embed_dim", type=int, default=768)
    ap.add_argument("--no_projection", action="store_true")

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=2.6e-4)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)

    ap.add_argument("--ckpt_dir", default="checkpoints")

    ap.add_argument("--ray_name", default="triplet_tune")
    ap.add_argument("--ray_dir", default="ray_results")
    ap.add_argument("--ray_num_samples", type=int, default=1)  # start small on EC2
    ap.add_argument("--ray_metric", default="val_loss")
    ap.add_argument("--ray_mode", default="min", choices=["min","max"])
    ap.add_argument("--ray_stop_iters", type=int, default=2)  # keep tiny initially

    return ap.parse_args()

def main():
    args = parse_args()
    if args.mode == "tune":
        run_small_scale_tune(args)
    else:
        run_full_training(args)

if __name__ == "__main__":
    main()
