import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# Load model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        a1 = self.net[0](x)
        self.a1 = self.net[1](a1)
        a2 = self.net[2](self.a1)
        self.a2 = self.net[3](a2)
        out = self.net[4](self.a2)
        return out

model = SimpleMLP()
model.load_state_dict(torch.load("shape_classifier.pth"))
model.eval()

# Dataset paths
IMG_DIR = "images_28x28"
LABEL_DIR = "labels_28x28"

# Image transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Helper functions
def load_image(index):
    fname = f"{index:05}.png"
    path = os.path.join(IMG_DIR, fname)
    return Image.open(path).convert("RGB")

def update_display():
    idx = int(index_var.get())
    img = load_image(idx)

    # Resize image to 280x280
    img_resized = img.resize((280, 280), Image.NEAREST)
    tk_img = ImageTk.PhotoImage(img_resized)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # Run through model
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(input_tensor)
        a1 = model.a1.squeeze().numpy()
        a2 = model.a2.squeeze().numpy()
        output = pred.item()

    with open(os.path.join(LABEL_DIR, f"{idx:05}.txt"), "r") as f:
        label_text = f.readline().strip()

    # Update labels
    label_var.set(f"Label: {label_text}")
    pred_var.set(f"Model prediction: {output:.4f}")

    # Update activations
    update_activation_canvas(activation_canvas1, a1, rows=64, cols=2)
    update_activation_canvas(activation_canvas2, a2, rows=64, cols=1)

def update_activation_canvas(canvas, activations, rows, cols):
    canvas.delete("all")
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    cell_w = w // cols
    cell_h = h // rows

    a_min = np.min(activations)
    a_max = np.max(activations)
    norm = (activations - a_min) / (a_max - a_min + 1e-8)

    for i in range(len(activations)):
        r = i % rows
        c = i // rows
        val = norm[i]
        gray = int(val * 255)
        color = f"#{gray:02x}{gray:02x}{gray:02x}"
        x0 = c * cell_w
        y0 = r * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

# Tkinter app
root = tk.Tk()
root.title("Neural Net Explorer")

# Left panel (Image + label)
left_frame = ttk.Frame(root)
left_frame.grid(row=0, column=0, padx=10, pady=10)

index_var = tk.StringVar(value="0")
entry = ttk.Entry(left_frame, textvariable=index_var, width=5)
entry.grid(row=0, column=1)

def prev_image():
    idx = max(0, int(index_var.get()) - 1)
    index_var.set(str(idx))
    update_display()

def next_image():
    idx = int(index_var.get()) + 1
    index_var.set(str(idx))
    update_display()

ttk.Button(left_frame, text="<", command=prev_image).grid(row=0, column=0)
ttk.Button(left_frame, text=">", command=next_image).grid(row=0, column=2)

image_label = ttk.Label(left_frame)
image_label.grid(row=1, column=0, columnspan=3, pady=10)

label_var = tk.StringVar()
pred_var = tk.StringVar()
ttk.Label(left_frame, textvariable=label_var).grid(row=2, column=0, columnspan=3)
ttk.Label(left_frame, textvariable=pred_var).grid(row=3, column=0, columnspan=3)

# Middle panel (128 neurons from layer 1)
middle_frame = ttk.Frame(root)
middle_frame.grid(row=0, column=1, padx=10, pady=10)
ttk.Label(middle_frame, text="Layer 1 Activations (128)").pack()
activation_canvas1 = tk.Canvas(middle_frame, width=160, height=320)
activation_canvas1.pack()

# Right panel (64 neurons from layer 2)
right_frame = ttk.Frame(root)
right_frame.grid(row=0, column=2, padx=10, pady=10)
ttk.Label(right_frame, text="Layer 2 Activations (64)").pack()
activation_canvas2 = tk.Canvas(right_frame, width=80, height=320)
activation_canvas2.pack()

# Initial render
update_display()
root.mainloop()
