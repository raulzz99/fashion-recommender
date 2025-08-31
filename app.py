import streamlit as st
from PIL import Image
import os

# Title
st.title("Simple Image Recommendation App")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Dummy gallery images (place a few sample images in a folder called 'gallery')
gallery_path = "gallery"
os.makedirs(gallery_path, exist_ok=True)

# You can put some test images in this folder manually (scp or wget them)
gallery_images = [os.path.join(gallery_path, f) for f in os.listdir(gallery_path) if f.endswith(("jpg","jpeg","png"))]

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    st.write("### Recommended Similar Images:")

    # For now: just display all gallery images (later you’d add similarity search)
    for img_path in gallery_images:
        img = Image.open(img_path)
        st.image(img, caption=os.path.basename(img_path), use_column_width=True)

if not gallery_images:
    st.info("No gallery images found. Add some images to the 'gallery/' folder.")
