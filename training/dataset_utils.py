
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def separate_deepfashion_data(dataset_base_path):
    """
    Parses the dataset file to separate data into training, validation, and testing sets
    based on the provided annotations.

    Args:
        dataset_base_path (str): The root directory of the dataset.

    Returns:
        tuple: A tuple containing dictionaries for train_data, query_data, and gallery_data.
               Each dictionary maps a class label to a list of image file paths.
    """
    annotations_file = os.path.join(dataset_base_path, 'Eval', 'list_eval_partition.txt')

    # Dictionaries to hold file paths for each partition
    train_data = defaultdict(list)
    query_data = defaultdict(list)
    gallery_data = defaultdict(list)

    with open(annotations_file, 'r') as f:
        # Skip header lines
        _ = f.readline()
        _ = f.readline()

        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                img_path = parts[0]
                img_class = parts[1]
                partition = parts[2]

                # Construct full image path
                full_img_path = os.path.join(dataset_base_path, img_path)

                # Add to the correct dictionary based on the partition
                if partition == 'train':
                    train_data[img_class].append(full_img_path)
                elif partition == 'query':
                    query_data[img_class].append(full_img_path)
                elif partition == 'gallery':
                   gallery_data[img_class].append(full_img_path)

    # Filter out classes with fewer than 2 images to ensure we can form a pair
    train_data = {cls: paths for cls, paths in train_data.items() if len(paths) > 1}
    query_data = {cls: paths for cls, paths in query_data.items() if len(paths) > 1}
    gallery_data = {cls: paths for cls, paths in gallery_data.items() if len(paths) > 1}

    return train_data, query_data, gallery_data

def train_val_split(train_data, val_ratio=0.2, seed=42):
    """
    Splits the training data into separate training and validation dictionaries.

    Args:
        train_data (dict): Dictionary mapping class labels to image paths.
        val_ratio (float): The ratio of images to allocate to the validation set.
        seed (int): The random seed for reproducibility.

    Returns:
        tuple: A tuple containing the new train_split and val_split dictionaries.
    """
    random.seed(seed)
    train_split = defaultdict(list)
    val_split = defaultdict(list)

    for cls, images in train_data.items():
        n_val = max(1, int(len(images) * val_ratio))  # at least 1 image for val
        shuffled_images = images.copy()
        random.shuffle(shuffled_images)
        val_images = shuffled_images[:n_val]
        train_images = shuffled_images[n_val:]
        
        if len(train_images) > 0:
            train_split[cls] = train_images
        if len(val_images) > 0:
            val_split[cls] = val_images

    return train_split, val_split


def create_triplet_training_data(train_images, num_triplets=5000):
    """
    Creates a list of (anchor, positive, negative) triplets for training.

    This function generates random triplets, which is a good starting point before
    implementing more advanced hard negative mining techniques.

    Args:
        train_images (list): A list of dictionaries, where each dictionary
                             contains the 'path' and 'item_id' of a training image.

    Returns:
        list: A list of dictionaries, where each dictionary represents a triplet
              with 'anchor', 'positive', and 'negative' image information.
    """
    # Group images by item_id to easily select positives and negatives
    item_id_to_images = defaultdict(list)
    for img_info in train_images:
        item_id_to_images[img_info['item_id']].append(img_info)

    # Filter out item_ids with only one image, as we can't form a positive pair
    valid_item_ids = [item_id for item_id, images in item_id_to_images.items() if len(images) > 1]

    print(f"Found {len(valid_item_ids)} item IDs with more than one image.")
    print(f"Print random elements from the valid_item_ids", random.sample(list(item_id_to_images.items()), 2) )
    # Generate a list of triplets
    triplets = []
    # Use a fixed number of triplets for demonstration and small-scale testing

    for _ in range(num_triplets):
        # 1. Select the Anchor
        anchor_item_id = random.choice(valid_item_ids)
        anchor_images = item_id_to_images[anchor_item_id]

        # Select two different images from the same item_id group
        anchor, positive = random.sample(anchor_images, 2)

        # 2. Select the Negative
        # Get a list of all other item IDs to choose a negative from
        negative_item_ids = [item_id for item_id in valid_item_ids if item_id != anchor_item_id]
        if not negative_item_ids:
            continue # Skip if there's no other item to form a negative from

        negative_item_id = random.choice(negative_item_ids)
        negative = random.choice(item_id_to_images[negative_item_id])

        triplets.append({
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        })

    return triplets

class DeepFashionTripletDataset(Dataset):
    """
    A PyTorch Dataset for on-the-fly generation of triplets. This is
    more memory-efficient and scalable than pre-generating a fixed list
    of triplets.
    """
    def __init__(self, data, transform=None, virtual_epoch_size=50000):
        """
        Args:
            data (dict): Dictionary mapping class labels to image paths.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
            virtual_epoch_size (int): The number of triplets to generate
                                      per epoch.
        """
        self.data = data
        self.transform = transform
        self.classes = list(data.keys())
        self.virtual_epoch_size = virtual_epoch_size

        # Create a reverse lookup for faster access
        self.class_to_images = {cls: paths for cls, paths in data.items()}
        self.all_images = [img_path for paths in self.class_to_images.values() for img_path in paths]
        self.image_to_class = {img_path: cls for cls, paths in self.class_to_images.items() for img_path in paths}

        # Check for empty dataset after initialization
        if not self.all_images:
            raise ValueError(
                "The dataset is empty. Please ensure the 'base_path' is correct and "
                "that the 'list_eval_partition.txt' file contains classes with more than one image."
            )

    def __len__(self):
        # We return a large number as a "virtual" epoch size.
        # This ensures the DataLoader keeps requesting triplets.
        return self.virtual_epoch_size

    def __getitem__(self, idx):
        while True:
            # 1. Select an Anchor image
            anchor_path = random.choice(self.all_images)
            anchor_class = self.image_to_class[anchor_path]

            # 2. Find a Positive image (same class as anchor)
            positive_candidates = [path for path in self.class_to_images[anchor_class] if path != anchor_path]
            
            # Ensure there are positive candidates to prevent an error
            if not positive_candidates:
                continue
            positive_path = random.choice(positive_candidates)

            # 3. Find a Negative image (different class from anchor)
            negative_candidates = [cls for cls in self.classes if cls != anchor_class]
            
            # Ensure there are negative candidates
            if not negative_candidates:
                continue
            negative_class = random.choice(negative_candidates)
            negative_path = random.choice(self.class_to_images[negative_class])
            
            # If a valid triplet is found, break the loop
            break

        # Load images
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img
