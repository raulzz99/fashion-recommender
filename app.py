import streamlit as st
from PIL import Image
import os
import io, time, json
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import faiss
except Exception:
    faiss = None
   

CONFIG_PATH = "/home/ec2-user/fashion-recommender/config/config.json"
TOP_K = 10
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

EMBEDDER_ONNX = cfg["onnx_path"]
FAISS_INDEX_PATH = cfg["faiss_index"]
METADATA_JSON_PATH = cfg["metadata_json"]
GALLERY_PATH = cfg["gallery_path"]
IMG_SIZE = cfg.get("img_size", 224)
EXPECTED_DIM = cfg.get("embedding_dim", 768)

@st.cache_resource(show_spinner=False)
def load_onnx_session(onnx_path: str):
    if ort is None:
        raise RuntimeError("onnxruntime not installed. `pip install onnxruntime` (or onnxruntime-gpu)")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Embedder ONNX not found at {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    return sess, inp_name

@st.cache_resource(show_spinner=False)
def get_onnx_session_and_names():
    sess, inp_name = load_onnx_session(EMBEDDER_ONNX)
    out_name = sess.get_outputs()[0].name
    # warm-up to avoid first-run timing skew
    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    _ = sess.run([out_name], {inp_name: dummy})
    return sess, inp_name, out_name


def preprocess(pil_img: Image.Image) -> np.ndarray:
    """ Match EXACTLY when building gallery embeddings."""
    im = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    x = np.array(im).astype(np.float32) / 255.0

    # EDIT mean/std if your training used different stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std  # HWC -> normalized

    x = np.transpose(x, (2, 0, 1))      # CHW
    x = x[None, ...].astype(np.float32) # NCHW with batch=1
    return x

def embed_once(pil_img: Image.Image) -> np.ndarray:
    """Return (1, D) float32 contiguous embedding."""
    sess, inp_name, out_name = get_onnx_session_and_names()
    x = preprocess(pil_img)                             # (1,3,H,W) float32
    out = sess.run([out_name], {inp_name: x})[0]       # e.g. (1,D) or (1,D,1,1)
    if out.ndim == 4 and out.shape[2:] == (1, 1):
        out = out[:, :, 0, 0]
    out = out.astype(np.float32, copy=False)
    return np.ascontiguousarray(out)                   # (1, D)

def run_embed_and_stats(pil_img: Image.Image):
    """Return (embedding_vector, stats_dict)."""
    sess, inp_name = load_onnx_session(EMBEDDER_ONNX)
    x = preprocess(pil_img)

    t0 = time.time()
    out = sess.run(None, {inp_name: x})
    ms = (time.time() - t0) * 1000.0

    vec = out[0].squeeze().astype(np.float32)  # assume output shape (1, D)
    stats = {
        "shape": tuple(vec.shape),
        "dtype": str(vec.dtype),
        "min": float(np.min(vec)),
        "max": float(np.max(vec)),
        "mean": float(np.mean(vec)),
        "std": float(np.std(vec)),
        "l2_norm": float(np.linalg.norm(vec)),
        "num_nan": int(np.isnan(vec).sum()),
        "num_inf": int(np.isinf(vec).sum()),
        "inference_ms": round(ms, 2),
    }
    return vec, stats

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    return v / (np.linalg.norm(v) + eps)

@st.cache_resource(show_spinner=False)
def load_faiss_index(index_path: str):
    if faiss is None:
        raise RuntimeError("faiss is not installed. pip install faiss-cpu or faiss-gpu.")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")
    return faiss.read_index(index_path)

@st.cache_data(show_spinner=False)
def load_metadata(meta_path: str):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    with open(meta_path, "r") as f:
        data = json.load(f)
    # Your builder saved: [{"id": i, "path": "...", "label": "..."}]
    # Keep as list for direct idx access
    if not isinstance(data, list):
        raise ValueError("metadata.json must be a list aligned with FAISS index order")
    return data

# Title
st.title("Simple Image Recommendation App")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# # You can put some test images in this folder manually (scp or wget them)
gallery_images = [os.path.join(GALLERY_PATH, f) for f in os.listdir(GALLERY_PATH) if f.endswith(("jpg","jpeg","png"))]
faiss_index = load_faiss_index(FAISS_INDEX_PATH)
metadata = load_metadata(METADATA_JSON_PATH)

st.caption(f"Index size: {faiss_index.ntotal:,} | Metadata entries: {len(metadata):,}")
if faiss_index.ntotal != len(metadata):
    st.warning("Index count != metadata count — paths may be misaligned.")

############################################
# 🖼️ UI
############################################


st.set_page_config(page_title="Image Recommender", layout="wide")
st.title("🔎 Image Similarity Search (FAISS)")

with st.sidebar:
    st.subheader("🔧 Self-test")
    if st.button("Run basic checks"):
        try:
            # l2_normalize
            v = np.array([3., 4.], dtype=np.float32)
            u = l2_normalize(v)
            st.write("‣ l2_normalize([3,4]) norm:", float(np.linalg.norm(u)))

            z = l2_normalize(np.zeros(5, np.float32))
            st.write("‣ l2_normalize(zeros) ->", z.tolist())

            # load_metadata / load_faiss_index already ran above
            st.success("Loaded FAISS index and metadata successfully ✅")

            # Smoke-test: dummy search path works
            rand_q = np.random.randn(faiss_index.d).astype(np.float32)
            rand_q = l2_normalize(rand_q)
            D, I = faiss_index.search(rand_q[None, :], 3)
            st.write("‣ Dummy search top-3 indices:", I[0].tolist())
            st.write("‣ Dummy search scores:", [float(x) for x in D[0]])
        except Exception as e:
            st.exception(e)


if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Display uploaded image
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔎 Embedding Diagnostics (no search yet)")
    st.subheader("Query Image")
    st.image(pil_img, use_column_width=True)

    # Embed + search
    st.subheader("Results")
    with st.spinner("Embedding & searching..."):
        t0 = time.time()
        vec = embed_once(pil_img)                  # (1, D)
        # optional: cosine search behavior via unit norm
        vec = l2_normalize(vec)                    # keep if you built index with unit-normed vectors
        D, I = faiss_index.search(vec, TOP_K)      # D: (1, K), I: (1, K)
        elapsed = (time.time() - t0) * 1000.0

    st.caption(f"Searched {faiss_index.ntotal:,} vectors in {elapsed:.1f} ms")

    # Render a grid of results
    cols_per_row = 4
    hits = I[0].tolist()
    dists = D[0].tolist()

    for start in range(0, len(hits), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for j, col in enumerate(cols):
            i = start + j
            if i >= len(hits):
                break
            idx = int(hits[i])
            dist = float(dists[i])

            # get path from metadata (robust to slightly different keys)
            path = None
            if 0 <= idx < len(metadata):
                m = metadata[idx]
                if isinstance(m, dict):
                    path = m.get("path") or m.get("img_path") or m.get("file")
                elif isinstance(m, str):
                    path = m

            with col:
                st.markdown(f"**#{i+1} — idx {idx}**")
                st.caption(f"distance: {dist:.4f}")
                if path and os.path.exists(path):
                    try:
                        st.image(Image.open(path).convert("RGB"), use_column_width=True,
                                 caption=os.path.basename(path))
                        st.code(path, language="text")
                    except Exception as e:
                        st.error(f"Failed to open image for idx {idx}: {e}")
                else:
                    st.warning(f"No image path for idx {idx}.")
else:
    st.info("Upload an image to see similar results.")

#     try:
#         vec, stats = run_embed_and_stats(pil_img)

#         # Show key stats
#         st.json(stats)

#         # Sanity checks
#         if stats["num_nan"] > 0 or stats["num_inf"] > 0:
#             st.error("Embedding contains NaN/Inf. Check preprocessing/model output.")
#         if EXPECTED_DIM and stats["shape"] not in [(EXPECTED_DIM,), (1, EXPECTED_DIM)]:
#             st.warning(f"Unexpected embedding dim: {stats['shape']} (expected {EXPECTED_DIM})")

#         # Preview first 32 values
#         preview = np.array2string(vec[:32], precision=4, separator=", ")
#         st.code(f"vec[:32] = {preview}")
#     except Exception as e:
#         st.exception(e)

#     # Display uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

#     st.write("### Recommended Similar Images:")

#     # For now: just display all gallery images (later you’d add similarity search)
#     for img_path in gallery_images:
#         img = Image.open(img_path)
#         st.image(img, caption=os.path.basename(img_path), use_column_width=True)

# # if not gallery_images:
# #     st.info("No gallery images found. Add some images to the 'gallery/' folder.")
