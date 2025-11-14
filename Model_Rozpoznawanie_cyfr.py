import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import TensorDataset, DataLoader

# Architektura modelu
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=11):  # 0-9 + kropka
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 28->14
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 14->7
        x = x.reshape(-1, 32*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Wczytanie obrazów i etykiet
def load_images(folder):
    X = []
    y = []
    for file in os.listdir(folder):
        if not file.endswith(".jpg"):
            continue
        label = int(os.path.splitext(file)[0])
        img = Image.open(os.path.join(folder, file))
        img = transform(img)
        X.append(img)
        y.append(label)
    return torch.stack(X), torch.tensor(y)

# Augmentacja ręczna: np. przesunięcia, skalowanie
def augment(img):
    # img: tensor 1x28x28
    return img  # na razie brak transformacji, można dodać np. small rotation/shift

if __name__ == "__main__":
    # Transformacje: konwersja do tensor i normalizacja
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # [0,1]
    ])

    model = SimpleCNN()

    X, y = load_images("cyfyry")
    print(X.shape, y.shape)

    X_aug = torch.stack([augment(x) for x in X])
    y_aug = y

    dataset = TensorDataset(X_aug, y_aug)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Proste trenowanie 50 epok
    for epoch in range(50):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "model_cyfr_weights.pth")




