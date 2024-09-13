import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageOps
import numpy as np

# Step 1: Loading image links from CSV
image_links_df = pd.read_csv('train.csv')
image_links_df = image_links_df.head(32)                     #delete: only required for sample 32 data inputs right now


# Function to resize or pad images
def resize_image_with_padding(img, target_size=(1500, 1500)):  # Use 1600x1600 as the target size
    width, height = img.size
    if width < target_size[0] or height < target_size[1]:
        return ImageOps.pad(img, target_size, color=(0, 0, 0))  # Black padding if smaller
    else:
        return ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)  # Resize if larger


# Step 2: Function to fetch and process images
def load_image_from_url(url, target_size=(1500, 1500)):
    try:
        # Fetch the image from the URL
        img_response = requests.get(url)
        img = Image.open(BytesIO(img_response.content))

        # Convert to RGB if needed and resize
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize or pad the image
        img = resize_image_with_padding(img, target_size)

        img = np.array(img) / 255.0  # Normalize image

        return img
    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return None

# Step 3: Preparing image data
def prepare_image_data(image_links, target_size=(1500, 1500)):
    images = []

    for url in image_links:
        image = load_image_from_url(url, target_size)
        if image is not None:
            images.append(image)

    return np.array(images)


images = prepare_image_data(image_links_df['image_link'],target_size=(1500, 1500))