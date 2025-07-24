import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# Paths for the images and labels
IMG_DIR = "images_28x28"
LABEL_DIR = "labels_28x28"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# Custom dataset to load image and label
class ShapeDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.file_names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Load image
        img_name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms (to tensor, normalization)
        if self.transform:
            image = self.transform(image)

        # Load label (first number in the label file)
        label_path = os.path.join(self.label_dir, img_name.replace(".png", ".txt"))
        with open(label_path, "r") as f:
            label = int(f.readline().split()[0])

        return image, torch.tensor(label, dtype=torch.float32)

# Image transformations: convert to tensor and flatten
transform = transforms.Compose([
    transforms.ToTensor(),                      # Converts to [0,1] and shape (C,H,W)
    transforms.Lambda(lambda x: x.view(-1))     # Flatten to 1D vector
])

# Prepare dataset and split
dataset = ShapeDataset(IMG_DIR, LABEL_DIR, transform=transform)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Neural network: MLP with 3 Linear layers and ReLU activations
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28 * 3, 128),   # First hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),            # Second hidden layer
            nn.ReLU(),
            nn.Linear(64, 1),              # Output layer (1 neuron)
            nn.Sigmoid()                   # Activation for binary classification
        )

    def forward(self, x):
        return self.net(x)

# Instantiate model, loss function and optimizer
model = SimpleMLP()
criterion = nn.BCELoss()                     # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)


# Track metrics
train_acc_list = []
val_acc_list = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        running_loss += loss.item() * inputs.size(0)
        predictions = (outputs >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if epoch == 0:
            print(
                f"  [Batch {batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} - "
                f"Batch Acc: {(predictions == labels).float().mean().item() * 100:.2f}%"
            )

    avg_loss = running_loss / total
    train_acc = 100 * correct / total
    train_acc_list.append(train_acc)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze(1)
            predictions = (outputs >= 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
    val_acc = 100 * val_correct / val_total
    val_acc_list.append(val_acc)

    epoch_time = time.time() - start_time
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Avg Loss: {avg_loss:.4f} - "
        f"Train Acc: {train_acc:.2f}% - "
        f"Val Acc: {val_acc:.2f}% - "
        f"Time: {epoch_time:.2f}s - "
        f"LR: {current_lr:.6f}"
    )

# Save model
torch.save(model.state_dict(), "shape_classifier.pth")
print("Model saved to shape_classifier.pth")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_acc_list, label="Train Acc", marker="o")
plt.plot(val_acc_list, label="Val Acc", marker="s")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("shape_classifier_accuracy.png", dpi=150)
plt.show()