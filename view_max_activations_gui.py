import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Diretórios
IMG_DIR = "images_28x28"
TOP_DIR = "neuron_top_indices"

# Estado atual
image_list = []
image_labels = []

def load_image(index):
    path = os.path.join(IMG_DIR, f"{index:05}.png")
    img = Image.open(path).convert("RGB")
    img = img.resize((64, 64), Image.NEAREST)
    return ImageTk.PhotoImage(img)

def update_display():
    for label in image_labels:
        label.destroy()
    image_labels.clear()

    layer_name = layer.get()

    if layer_name == "layer3":
        idx = neuron_index.get()
        fname = os.path.join(TOP_DIR, "layer3", "000.txt")
        if not os.path.exists(fname):
            return
        with open(fname, "r") as f:
            indices = [int(line.strip()) for line in f.readlines()]
        indices = indices[idx:idx + 10]
    else:
        idx = neuron_index.get()
        fname = os.path.join(TOP_DIR, layer_name, f"{idx:03}.txt")
        if not os.path.exists(fname):
            return
        with open(fname, "r") as f:
            indices = [int(line.strip()) for line in f.readlines()]

    row = 1
    for i, img_idx in enumerate(indices):
        img = load_image(img_idx)
        lbl = ttk.Label(right_frame, image=img)
        lbl.image = img
        lbl.grid(row=row, column=i, padx=4, pady=4)
        image_labels.append(lbl)

def prev_neuron():
    neuron_index.set(max(0, neuron_index.get() - 1))
    update_display()

def next_neuron():
    neuron_index.set(neuron_index.get() + 1)
    update_display()

# GUI
root = tk.Tk()
root.title("Neuron Activation Explorer")

# Variáveis ligadas ao root
layer = tk.StringVar(value="layer1")
neuron_index = tk.IntVar(value=0)

left_frame = ttk.Frame(root)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

ttk.Label(left_frame, text="Layer").pack()
ttk.Radiobutton(left_frame, text="Layer 1", variable=layer, value="layer1", command=update_display).pack(anchor="w")
ttk.Radiobutton(left_frame, text="Layer 2", variable=layer, value="layer2", command=update_display).pack(anchor="w")
ttk.Radiobutton(left_frame, text="Layer 3", variable=layer, value="layer3", command=update_display).pack(anchor="w")

ttk.Label(left_frame, text="Neuron index").pack(pady=(10, 0))
entry = ttk.Entry(left_frame, textvariable=neuron_index, width=5)
entry.pack()

ttk.Button(left_frame, text="<", command=prev_neuron).pack(pady=5)
ttk.Button(left_frame, text=">", command=next_neuron).pack()

right_frame = ttk.Frame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10)

update_display()
root.mainloop()
