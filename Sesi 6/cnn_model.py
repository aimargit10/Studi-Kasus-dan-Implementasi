import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleCNN(nn.Module):
    """Arsitektur CNN sederhana untuk klasifikasi citra land cover/satelit."""

    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_dummy_dataset(num_samples: int = 200, image_size: int = 32, channels: int = 3, num_classes: int = 4):
    """Membuat tensor citra dummy dan label dummy untuk simulasi dataset klasifikasi citra."""
    images = torch.randn(num_samples, channels, image_size, image_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


def train_model(model, train_loader, criterion, optimizer, device):
    """Training loop dengan optimizer Adam dan loss CrossEntropyLoss."""
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def evaluate_model(model, test_loader, device):
    """Menghitung akurasi model pada data uji."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    """Pipeline utama pembelajaran mesin untuk simulasi klasifikasi citra."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    num_classes = 4
    batch_size = 32
    epochs = 5

    dataset = create_dummy_dataset(num_samples=200, image_size=32, channels=3, num_classes=num_classes)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        epoch_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"\nAkurasi pada data uji: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
