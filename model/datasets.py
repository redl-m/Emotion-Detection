from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# define transforms:
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# root train/test dirs:
train_folder = '../data/train'
test_folder  = '../data/test'

train_ds = datasets.ImageFolder(root=train_folder, transform=transform)
val_ds   = datasets.ImageFolder(root=test_folder,  transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)
