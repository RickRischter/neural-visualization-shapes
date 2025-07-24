# Neural Visualization: Simple Shape Classifier

This is an educational project that demonstrates the complete process of training a neural network (MLP) to classify simple geometric shapes and visually exploring its inner neuron activations.

## 🧠 Overview

The pipeline includes:

- **Synthetic dataset generation**: Colored rectangles and ellipses on black background
- **Model training**: A simple fully connected network with 2 hidden layers
- **Neuron activation recording**: Saving which inputs maximally activate each neuron
- **Interactive visualization**: GUI tools to inspect the learned behavior

This project is intended for students learning about neural networks, activations, and classification from scratch.

---

## 📁 Project Structure

```
.
├── generate_shapes.py              # Generates synthetic images and labels
├── train_model.py                  # Trains a simple MLP classifier
├── record_max_activations.py       # Records which images activate each neuron the most
├── explore_model.py                # GUI: shows model output and activations for a given image
├── view_max_activations_gui.py     # GUI: visualizes top-activating inputs for each neuron
├── images_28x28/                   # Auto-generated images (not included in repo)
├── labels_28x28/                   # Auto-generated label files (not included in repo)
├── neuron_top_indices/             # Generated neuron data (not included in repo)
├── shape_classifier.pth            # Trained model (not included in repo)
├── shape_classifier_accuracy.png   # Accuracy plot saved after training (not included in repo)
├── suggestions.md                  # List of ideas and possible improvements
└── README.md                       # This file
```

---

## 🖼️ Image Format

Each image is `28x28` pixels with **one shape** per image.

Each label file contains:

```
type x0 y0 x1 y1 R G B
```

- `type`: `0 = rectangle`, `1 = ellipse`
- `x0 y0 x1 y1`: bounding box coordinates
- `R G B`: color of the shape

---

## 🔧 Usage Instructions

### 1. Generate Dataset

```bash
python generate_shapes.py
```

This will create:

- `images_28x28/` with PNG images
- `labels_28x28/` with text labels

### 2. Train the Classifier

```bash
python train_model.py
```

The script:

- Trains a small MLP for 50 epochs
- Saves the model as `shape_classifier.pth`

### 3. Record Neuron Activations

```bash
python record_max_activations.py
```

This will:

- Run all images through the model
- Save which ones maximally activate each neuron in `neuron_top_indices/`

### 4. Visualize Neuron Behavior (Two GUIs)

#### a) Explore by Image

```bash
python explore_model.py
```

- Displays one image at a time
- Shows predicted output
- Shows activation maps of both hidden layers

#### b) Explore by Neuron

```bash
python view_max_activations_gui.py
```

- Lets you select a layer and neuron index
- Shows the top 10 images that most activate that neuron

---

## 💡 Educational Goals

- Understand how MLPs process visual patterns
- Observe how neurons specialize in different features
- Connect activations to model decisions
- Encourage exploratory analysis of neural networks

---

## 📚 License

This project is shared for educational purposes and is licensed under the MIT License.

---

## 🙋 Author

Developed by [@RickRischter](https://github.com/RickRischter)  
Feel free to fork, adapt, or use it in your own classes or learning!

---

### 🐍 Optional: Creating a Virtual Environment

If you want to keep dependencies isolated, you can create a virtual environment:

```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Linux/macOS)
source venv/bin/activate

# Then install the dependencies
pip install torch torchvision pillow tqdm matplotlib
```

This ensures all required packages are installed only inside this project folder.
