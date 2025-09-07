# 📸 Simple Image Recommendation App (Streamlit)

A small demo that lets users upload an image and see "similar" images from a gallery. The app is designed to evolve from a placeholder recommender to a real embedding + FAISS similarity system running **CPU‑only** by default (works great on EC2 without CUDA).

---

## ✨ Features

* Upload an image (JPG/PNG)
* Display the uploaded image
* Show a set of gallery images as “recommendations” (placeholder logic today)
* Ready for ONNX embedding + FAISS index integration
* Works locally or on a remote server (e.g., AWS EC2)

---

## 🧱 Project Structure

```
fashion-recommender/
├─ app.py                            # Streamlit UI
├─ scripts/
│  └─ build_gallery_index.py         # WIP: loads ONNX model; extend to write FAISS index
├─ gallery_images/                   # place your dataset here
│  ├─ img/WOMEN/...                  # DeepFashion-like layout (example)
│  └─ img/MEN/...
│  └─ list_eval_partition.txt        # optional eval mapping file
├─ artifacts/                        # models & artifacts
│  └─ fashion_embedder_v1.onnx       # ONNX embedder (copied via scp)
├─ index/                            # FAISS index + metadata output (to be generated)
├─ requirements.txt                  # Python deps (see below)
└─ README.md
```

> **Note**: The current `scripts/build_gallery_index.py` prints resolved paths and loads the ONNX model. Extend it to: iterate images → preprocess → run ONNX → build FAISS index → write `index/gallery.faiss` and `index/metadata.json`.

---

## ✅ Prerequisites

* Python **3.9** (tested)
* No GPU required (CPU-only wheels)
* On EC2, open inbound **TCP 8501** in the Security Group to access the Streamlit app remotely.

---

## ⚙️ Setup

### 1) Clone the repo

```bash
git clone <your-repo-url>
cd fashion-recommender
```

### 2) Create & activate a virtualenv (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows PowerShell
```

### 3) Install dependencies (CPU‑only)

> Use `python -m pip` so installs land in the active venv.

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install --no-cache-dir "numpy<2" pillow faiss-cpu==1.8.0.post1 tqdm onnxruntime onnx==1.16.2 streamlit
```

**Quick sanity check**

```bash
python - <<'PY'
import torch, faiss, onnxruntime as ort, numpy as np, PIL
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("faiss:", getattr(faiss, "__version__", "?"))
print("onnxruntime:", ort.__version__, "providers:", ort.get_available_providers())
print("numpy:", np.__version__)
print("pillow:", getattr(PIL, "__version__", "?"))
PY
```

---

## 🖼️ Add Gallery Images

Place your dataset under `gallery_images/`. Example:

```
gallery_images/
└─ img/
   ├─ WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg
   ├─ WOMEN/Blouses_Shirts/id_00000001/02_3_back.jpg
   └─ MEN/Denim/id_00000007/...
```

(Optional) If you use DeepFashion partitions, copy `list_eval_partition.txt` into `gallery_images/`.

**Copying to EC2 example**

```bash
# from local → EC2 (adjust IP/paths)
scp -i ~/.ssh/<key>.pem -r ~/Desktop/gallery_images \
  ec2-user@<EC2_PUBLIC_IP>:/home/ec2-user/fashion-recommender/
```

---

## 🧠 ONNX Model (Embedder)

Put your exported ONNX model into `artifacts/` as `fashion_embedder_v1.onnx`:

```bash
scp -i ~/.ssh/<key>.pem ~/Desktop/fashion_embedder_v1.onnx \
  ec2-user@<EC2_PUBLIC_IP>:/home/ec2-user/fashion-recommender/artifacts/
```

> If you need an exporter:

```bash
python - <<'PY'
import torch, os
m = torch.nn.Identity()               # replace with your model
x = torch.randn(1,3,224,224)
os.makedirs('artifacts', exist_ok=True)
torch.onnx.export(
  m, x, 'artifacts/fashion_embedder_v1.onnx',
  input_names=['images'], output_names=['embeddings'],
  opset_version=17,
  dynamic_axes={'images':{0:'batch'}, 'embeddings':{0:'batch'}})
print('Exported artifacts/fashion_embedder_v1.onnx')
PY
```

---

## 🔎 (WIP) Build Gallery Index with FAISS

Run the helper script to validate paths and load the model (extend it to actually build the index):

```bash
python ./scripts/build_gallery_index.py \
  --project-root "/home/ec2-user/fashion-recommender/" \
  --onnx-file "fashion_embedder_v1.onnx" \
  --batch-size 64
```

Expected output includes the resolved `Project Root`, `Artifacts Dir`, `ONNX path`, and a message like `Using ONNX embedder ✅`.

**Planned outputs (once implemented):**

* `index/gallery.faiss` — FAISS index (e.g., `IndexFlatL2` or `IndexIVFFlat`)
* `index/metadata.json` — mapping of row → image path / id

---

## ▶️ Run the Streamlit App

### Local

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Remote (EC2)

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://<EC2_PUBLIC_IP>:8501` (ensure SG allows port 8501). For long sessions, consider `tmux`:

```bash
tmux new -s recapp
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
# detach: Ctrl-b d  |  reattach: tmux attach -t recapp
```

---

## 📦 requirements.txt (reference)

If you prefer a single file, this works for CPU-only:

```
# core
numpy<2
pillow
streamlit

tqdm
onnx==1.16.2
onnxruntime
faiss-cpu==1.8.0.post1

# vision
torch==2.8.0+cpu        ; platform_system != "Windows"
torchvision==0.23.0+cpu ; platform_system != "Windows"
--extra-index-url https://download.pytorch.org/whl/cpu
```

> On Windows, drop the `+cpu` pins and install from the standard index.

---

## 🧪 Dev Notes (how embedding should work next)

1. Preprocess each image with torchvision transforms:

   * `Resize(256) → CenterCrop(224) → ToTensor() → Normalize(mean,std)`
2. Batch to **float32** tensor `(B,3,224,224)` and run ONNXRuntime with `CPUExecutionProvider`.
3. L2-normalize embeddings → add to FAISS index; store metadata (paths/ids) in parallel.
4. In `app.py`, compute query embedding and run `index.search(query, k)` to get top‑K image ids.

---

## 🛠️ Troubleshooting

* **Pip installs but app says deps missing** → You used a different interpreter. Always use `python -m pip install ...` and verify with:

  ```bash
  which python; python -V
  python -m pip -V
  python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"],"\n",sysconfig.get_paths()["platlib"])'
  ```
* **Pip tries to download CUDA wheels & runs out of space** → Force CPU index and no cache:

  ```bash
  python -m pip uninstall -y torch torchvision torchaudio nvidia-*
  python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision
  ```
* **NumPy ABI error** (compiled against 1.x, running on 2.x) → pin NumPy < 2 and reinstall affected pkgs:

  ```bash
  python -m pip install --no-cache-dir 'numpy<2' --upgrade --force-reinstall
  python -m pip install --no-cache-dir --force-reinstall faiss-cpu==1.8.0.post1 onnxruntime
  ```
* **ONNX model not found** → ensure it exists at `artifacts/fashion_embedder_v1.onnx` or pass `--onnx-file`.
* **Port 8501 not reachable** → open SG inbound rule and check EC2 firewall.

---

## 🔒 Reproducibility

Export a lockfile of exact versions:

```bash
python -m pip freeze > requirements.lock.txt
```

Install from it later:

```bash
python -m pip install -r requirements.lock.txt
```

---
