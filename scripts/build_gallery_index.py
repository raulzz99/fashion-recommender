#!/usr/bin/env python3
import os, json, argparse, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Try FAISS; fall back gracefully if missing
try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False

class ONNXEmbedder:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
    def __call__(self, batch):
        arr = batch.cpu().numpy().astype(np.float32)   # (B,3,224,224)
        out = self.sess.run(["embeddings"], {self.input_name: arr})[0]
        return out.astype(np.float32)

eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def iter_paths_from_folder(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                paths.append(os.path.join(root, fn))
    return paths

def iter_paths_from_partition(partition_file, gallery_root):
    """
    DeepFashion eval file format: first two header lines, then:
    <relative_path> <item_id> <split>
    We keep only split == 'gallery'
    """
    paths = []
    with open(partition_file, "r") as f:
        _ = f.readline()
        _ = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rel_path, _, split = parts[0], parts[1], parts[2]
            if split == "gallery":
                full = os.path.join(gallery_root, rel_path)
                paths.append(full)
    return paths

def load_images_to_batch(paths):
    imgs, keep = [], []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            imgs.append(eval_tf(im))
            keep.append(p)
        except Exception as e:
            print(f"[skip] {p}: {e}")
    if not imgs:
        return torch.empty(0, 3, 224, 224), []
    return torch.stack(imgs, dim=0), keep
   

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True, help="Path to the main project directory")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Relative path under project-root")
    ap.add_argument("--outdir",        default="index",    help="Relative path under project-root")
    ap.add_argument("--onnx-file",     default="fashion_embedder_v1.onnx",
                    help="File name of the ONNX model inside artifacts-dir")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-partition", action="store_true",
                    help="Use list_eval_partition.txt in project root to index only gallery split")
    ap.add_argument("--sample", type=int, default=0,
                    help="If >0, randomly sample N images (for a small-scale dry run)")

    args = ap.parse_args()

    project_root = os.path.abspath(args.project_root)
    gallery_root = os.path.join(project_root, "gallery_images")
    partition_file = os.path.join(gallery_root, "list_eval_partition.txt")  # <- at project root
    artifacts_dir = os.path.join(project_root, args.artifacts_dir)
    outdir = os.path.join(project_root, args.outdir)
    onnx_path = os.path.join(artifacts_dir, args.onnx_file)

    os.makedirs(outdir, exist_ok=True)

    print(f"Project Root : {project_root}")
    print(f"Gallery Root : {gallery_root}")
    print(f"Eval File    : {partition_file}")
    print(f"Artifacts Dir: {artifacts_dir}")
    print(f"ONNX path    : {onnx_path}")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}\n"
                                f"Hint: scp it there or use --onnx-file to point to the right name.")

    print("Loading ONNX model…")
    embedder = ONNXEmbedder(onnx_path)
    print("Using ONNX embedder ✅")

    
    if args.use_partition and os.path.exists(partition_file):
        print("Using DeepFashion partition file to choose gallery images…")
        paths = iter_paths_from_partition(partition_file, gallery_root)
    else:
        print("Indexing all images under gallery_images/ recursively…")
        paths = iter_paths_from_folder(gallery_root)

    if not paths:
        raise RuntimeError("No gallery images found.")

    if args.sample and args.sample > 0 and args.sample < len(paths):
        import random
        random.seed(0)
        paths = random.sample(paths, args.sample)
        print(f"[sample mode] Using {len(paths)} images")

    print(f"Total images to index: {len(paths)}")

    # Embed in batches
    B = args.batch_size
    vecs, kept = [], []
    for i in tqdm(range(0, len(paths), B), desc="Embedding gallery"):
        batch_paths = paths[i:i+B]
        batch_imgs, ok_paths = load_images_to_batch(batch_paths)
        if batch_imgs.numel() == 0:
            continue
        V = embedder(batch_imgs)  # (b, D) float32, L2-normalized
        vecs.append(V); kept.extend(ok_paths)

    G = np.ascontiguousarray(np.vstack(vecs).astype(np.float32))
    print("Embeddings array:", G.shape)

    # Save metadata (id, path, label inferred from folder just above filename)
    meta = [{"id": i,
             "path": kept[i],
             "label": os.path.normpath(kept[i]).split(os.sep)[-2]} for i in range(len(kept))]
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    print(f"Saved metadata -> {os.path.join(outdir, 'metadata.json')}")

    # Build index
    if FAISS_OK:
        d = G.shape[1]
        index = faiss.IndexFlatIP(d)       # IP == cosine since embeddings are unit-norm
        index.add(G)
        faiss.write_index(index, os.path.join(outdir, "gallery.index"))
        print(f"Saved FAISS index -> {os.path.join(outdir, 'gallery.index')}")
    else:
        np.save(os.path.join(outdir, "gallery_embeddings.npy"), G)
        print("[warn] FAISS not available — saved embeddings to gallery_embeddings.npy instead")

    print("Done.")


if __name__ == "__main__":
    main()
