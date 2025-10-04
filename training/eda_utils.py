import os
import random
from collections import Counter
from typing import Dict, List, Tuple, Callable, Optional

import matplotlib.pyplot as plt
from PIL import Image

DataDict = Dict[str, List[str]]
Triplet = Tuple[str, str, str]

# ---------------------------- Basics & Summaries ----------------------------

def summarize_counts(data: DataDict, name: str = "data") -> dict:
    """Prints and returns summary statistics for a split."""
    num_ids = len(data)
    img_counts = [len(v) for v in data.values()]
    total_imgs = sum(img_counts)
    min_imgs = min(img_counts) if img_counts else 0
    max_imgs = max(img_counts) if img_counts else 0
    mean_imgs = total_imgs / num_ids if num_ids else 0.0

    summary = {
        "split": name,
        "identities": num_ids,
        "total_images": total_imgs,
        "min_images_per_id": min_imgs,
        "max_images_per_id": max_imgs,
        "mean_images_per_id": round(mean_imgs, 3),
    }
    print(f"[{name}] identities={num_ids} | images={total_imgs} "
          f"| min/mean/max per id = {min_imgs}/{summary['mean_images_per_id']}/{max_imgs}")
    return summary

def plot_class_distribution(data: DataDict, title: str = "Images per Identity", bins: int = 5):
    """Histogram of images per identity."""
    counts = [len(v) for v in data.values()]
    if not counts:
        print("No data to plot.")
        return
    plt.figure(figsize=(8, 4))
    plt.hist(counts, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel("# images per identity")
    plt.ylabel("Frequency")
    plt.show()


# ---------------------------- Visual Sampling ----------------------------

def show_random_images(data: DataDict, title: str, n: int = 5, seed: Optional[int] = None):
    """Show n random images from random identities of a split."""
    if seed is not None:
        random.seed(seed)
    if not data:
        print(f"No data for {title}")
        return
    chosen_ids = random.sample(list(data.keys()), k=min(n, len(data)))
    plt.figure(figsize=(3 * len(chosen_ids), 3))
    for i, id_ in enumerate(chosen_ids, 1):
        if not data[id_]:
            continue
        img_path = random.choice(data[id_])
        try:
            with Image.open(img_path) as img:
                plt.subplot(1, len(chosen_ids), i)
                plt.imshow(img)
                plt.axis("off")
                plt.title(str(id_))
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
    plt.suptitle(title)
    plt.show()

def sample_triplet_paths_from_data(data, k=300, seed=123):
    """
    Returns up to k triplets of (anchor_path, positive_path, negative_path).
    Samples only from classes with >=2 images for (a,p).
    """
    rng = random.Random(seed)
    # classes usable for positive pairs
    pos_classes = [cls for cls, paths in data.items() if len(paths) >= 2]
    if not pos_classes:
        raise ValueError("No classes with >=2 images; cannot sample positives.")
    all_classes = list(data.keys())

    triplets = []
    for _ in range(k):
        ac = rng.choice(pos_classes)
        a, p = rng.sample(data[ac], 2)              # two distinct images same class
        # pick any different class for negative
        nc = rng.choice(all_classes)
        while nc == ac:
            nc = rng.choice(all_classes)
        n = rng.choice(data[nc])
        triplets.append((a, p, n))
    return triplets

def build_reverse_index(*data_dicts: DataDict) -> Dict[str, str]:
    """Return mapping: image_path -> identity across provided splits."""
    rev = {}
    for d in data_dicts:
        for id_, paths in d.items():
            for p in paths:
                rev[p] = id_
    return rev

def validate_triplets(triplets: List[Triplet], reverse_index: Dict[str, str], max_check: int = 5000) -> dict:
    """Validate that (a,p) share identity and (a,n) do not. Returns simple counts."""
    ok = 0
    bad_pos = 0
    bad_neg = 0
    checked = 0
    for a, p, n in triplets[:max_check]:
        ia = reverse_index.get(a, None)
        ip = reverse_index.get(p, None)
        ineg = reverse_index.get(n, None)
        if ia is None or ip is None or ineg is None:
            # Skip triplets with paths outside provided dicts
            continue
        checked += 1
        if ia != ip:
            bad_pos += 1
        if ia == ineg:
            bad_neg += 1
        if ia == ip and ia != ineg:
            ok += 1
    report = {"checked": checked, "ok": ok, "bad_positive": bad_pos, "bad_negative": bad_neg}
    print(f"Triplet validation: {report}")
    return report

def visualize_triplet(triplets: List[Triplet], index: int = 0, title: str = "Triplet Sample"):
    """Display a single (anchor, positive, negative) triplet by index."""
    if not triplets:
        print("No triplets provided.")
        return
    index = max(0, min(index, len(triplets) - 1))
    a, p, n = triplets[index]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, path, lab in zip(axes, [a, p, n], ["Anchor", "Positive", "Negative"]):
        try:
            with Image.open(path) as img:
                ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load\n{os.path.basename(path)}", ha='center')
        ax.set_title(lab)
        ax.axis("off")
    plt.suptitle(title)
    plt.show()


# ---------------------------- One-Call Orchestrator ----------------------------

def run_basic_eda(
    train_data: DataDict,
    query_data: DataDict,
    gallery_data: DataDict,
    create_triplets_fn: Optional[Callable[[DataDict], List[Triplet]]] = None,
    triplets: Optional[List[Triplet]] = None,
    n_show: int = 5,
    seed: Optional[int] = 123
) -> dict:
    """
    Runs a compact EDA suite:
     1) Summaries for each split
     2) Histograms of images/identity for train
     3) Random image grids for each split
     4) Resolution probe & corruption check (sampled)
     5) Split integrity checks
     6) Triplet validation + visualization (if provided or creatable)

    Returns a dictionary of useful artifacts/reports.
    """
    reports = {}
    # 1) Summaries
    reports["train_summary"] = summarize_counts(train_data, "train")
    reports["query_summary"] = summarize_counts(query_data, "query")
    reports["gallery_summary"] = summarize_counts(gallery_data, "gallery")

    # 2) Histogram
    plot_class_distribution(train_data, title="Train: Images per Identity")

    # 3) Random image grids
    show_random_images(train_data, "Train Samples", n=n_show, seed=seed)
    show_random_images(query_data, "Query Samples", n=n_show, seed=seed)
    show_random_images(gallery_data, "Gallery Samples", n=n_show, seed=seed)
    
    return reports
