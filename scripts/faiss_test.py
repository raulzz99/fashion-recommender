import os
import faiss
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
embedding_dim = 768        # typical embedding size
num_gallery_images = 200  # dummy gallery size

# -----------------------------
# Step 1: Set up robust path for index
# -----------------------------
# Get absolute path to the parent folder of this script (streamlit-app)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Create index folder under streamlit-app
base_dir = os.path.join(root_dir, "index")
os.makedirs(base_dir, exist_ok=True)

# Path to gallery.index
index_file = os.path.join(base_dir, "gallery.index")
print(f"FAISS index will be saved to: {index_file}")

# -----------------------------
# Step 2: Generate random embeddings
# -----------------------------
np.random.seed(42)
gallery_embeddings = np.random.rand(num_gallery_images, embedding_dim).astype('float32')
print(f"Generated {num_gallery_images} random embeddings of size {embedding_dim}.")

# -----------------------------
# Step 3: Create FAISS index (IndexFlatL2 for exact search)
# -----------------------------
index = faiss.IndexFlatL2(embedding_dim)
print("Index created. Is trained:", index.is_trained)

# Add embeddings
index.add(gallery_embeddings)
print("Number of vectors in index:", index.ntotal)

# -----------------------------
# Step 4: Save index to disk
# -----------------------------
faiss.write_index(index, index_file)
print(f"FAISS index saved successfully to '{index_file}'")

# -----------------------------
# Step 5: Load index and test search
# -----------------------------
loaded_index = faiss.read_index(index_file)
print("Loaded index vectors:", loaded_index.ntotal)

# Test with a random query embedding
query = np.random.rand(1, embedding_dim).astype('float32')
distances, indices = loaded_index.search(query, k=5)
print("Top 5 nearest indices:", indices)
print("Distances:", distances)
