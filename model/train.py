import os
import argparse

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EmotionCNN  # assumes model.py is in the same folder


def train(data_dir, epochs, batch_size, lr, save_path):
    # define transforms: grayscale, resize to 48×48, normalize
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # point ImageFolder at your train/ and test/ dirs
    train_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform
    )
    val_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = EmotionCNN(num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # ---- training phase ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # ---- validation phase ----
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds   = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_acc = correct / len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs}  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Acc: {val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"-- New best model saved to {save_path} (Val Acc: {best_acc:.4f})")

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train EmotionCNN with ImageFolder data")
    parser.add_argument('--data_dir',   type=str, default='../data',
                        help="root directory containing train/ and test/ subfolders")
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--save_path',  type=str, default='../model.pth',
                        help="where to write the best model state_dict")
    args = parser.parse_args()

    train(
        data_dir   = args.data_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        save_path  = args.save_path
    )
