import os
import heapq
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Diretórios
IMG_DIR = "images_28x28"
OUT_DIR = "neuron_top_indices"
os.makedirs(os.path.join(OUT_DIR, "layer1"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "layer2"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "layer3"), exist_ok=True)


# Modelo
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
        return self.net[4](self.a2)

model = SimpleMLP()
model.load_state_dict(torch.load("shape_classifier.pth"))
model.eval()

# Transforma imagem em tensor 1D
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Inicia lista de heaps para cada neurônio
topk = 10
layer1 = [ [] for _ in range(128) ]
layer2 = [ [] for _ in range(64) ]
layer3_list = []

# Itera sobre as imagens
n_imgs = len(os.listdir(IMG_DIR))
for i in tqdm(range(n_imgs), desc="Processing images"):
    path = os.path.join(IMG_DIR, f"{i:05}.png")
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        _ = model(x)
        a1 = model.a1.squeeze().tolist()
        a2 = model.a2.squeeze().tolist()
        out3 = model(x).squeeze().item()
        layer3_list.append((out3, i))

    for j, val in enumerate(a1):
        heapq.heappush(layer1[j], (val, i))
        if len(layer1[j]) > topk:
            heapq.heappop(layer1[j])

    for j, val in enumerate(a2):
        heapq.heappush(layer2[j], (val, i))
        if len(layer2[j]) > topk:
            heapq.heappop(layer2[j])

# Salva resultados
def save_heap(layer, name):
    for idx, heap in enumerate(layer):
        heap.sort(reverse=True)
        indices = [str(i) for _, i in heap]
        with open(os.path.join(OUT_DIR, name, f"{idx:03}.txt"), "w") as f:
            f.write("\n".join(indices))

save_heap(layer1, "layer1")
save_heap(layer2, "layer2")

layer3_list.sort(reverse=True)
with open(os.path.join(OUT_DIR, "layer3", "000.txt"), "w") as f:
    for _, i in layer3_list:
        f.write(f"{i}\n")

print("Done.")
