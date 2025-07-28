import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import random

OUTPUT_DIR = "images_28x28"
LABEL_DIR = "labels_28x28"
IMG_SIZE = 28
N_IMAGES = 10_000

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

def random_color():
    """Returns a random RGB color tuple."""
    return tuple(np.random.randint(0, 256, size=3))

def generate_random_shape_params():
    """
    Generates random parameters for a single shape.
    
    Returns:
        shape_type (str): "rectangle" or "ellipse"
        x0, y0, x1, y1 (int): coordinates of the bounding box (integers)
        color (tuple): RGB color
    """
    shape_type = random.choice(["rectangle", "ellipse"])
    x0, y0 = np.random.randint(0, IMG_SIZE - 1 - 2, size=2)
    x1, y1 = np.random.randint(2, IMG_SIZE - 1, size=2)
    x0, x1 = sorted([x0, x1])
    y0, y1 = sorted([y0, y1])

    if x1 == x0:
        x0 -= 1
        x1 += 1

    if y1 == y0:
        y0 -= 1
        y1 += 1
    
    if x1 - x0 == 1:
        if random.choice([0,1]) == 0:
            x0 -= 1
        else:
            x1 += 1
            
    if y1 - y0 == 1:
        if random.choice([0,1]) == 0:
            y0 -= 1
        else:
            y1 += 1

    color = random_color()
    return shape_type, x0, y0, x1, y1, color

def draw_shape(draw, shape_type, x0, y0, x1, y1, color):
    """Draws a shape (rectangle or ellipse) on the given draw object."""
    if shape_type == "rectangle":
        draw.rectangle([x0, y0, x1, y1], fill=color)
    else:
        draw.ellipse([x0, y0, x1, y1], fill=color)

def generate_image(i):
    """
    Generates one image with a single random shape and its corresponding label file.
    
    The label file format is:
        type x0 y0 x1 y1 R G B
        
    Where:
        - type: 0 = rectangle, 1 = ellipse
        - x0, y0, x1, y1: bounding box coordinates (integers)
        - R G B: color values (0–255)
    """
    shape_type, x0, y0, x1, y1, color = generate_random_shape_params()
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape_type, x0, y0, x1, y1, color)
    img.save(os.path.join(OUTPUT_DIR, f"{i:05}.png"))

    shape_id = 0 if shape_type == "rectangle" else 1
    label_path = os.path.join(LABEL_DIR, f"{i:05}.txt")
    with open(label_path, "w") as f:
        f.write(f"{shape_id} {x0} {y0} {x1} {y1} {color[0]} {color[1]} {color[2]}\n")

if __name__ == "__main__":
    for i in tqdm(range(N_IMAGES), desc="Generating images"):
        generate_image(i)
