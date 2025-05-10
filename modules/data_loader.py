import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class MIMIIDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.samples = []
        self.labels = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for CNNs like ResNet
            transforms.ToTensor(),
        ])
        label_map = {'normal': 0, 'abnormal': 1}

        for snr_folder in os.listdir(root_dir):
            snr_path = os.path.join(root_dir, snr_folder)
            if not os.path.isdir(snr_path):
                continue
            for machine_type in os.listdir(snr_path):
                machine_path = os.path.join(snr_path, machine_type)
                for label_name in ['normal', 'abnormal']:
                    label_path = os.path.join(machine_path, label_name)
                    if os.path.exists(label_path):
                        for file in os.listdir(label_path):
                            if file.endswith('.png'):
                                self.samples.append(os.path.join(label_path, file))
                                self.labels.append(label_map[label_name])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(dataset_path, batch_size, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    full_dataset = MIMIIDataset(dataset_path, transform=transform)
    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_split, stratify=full_dataset.labels)

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_validation_loader(dataset_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    val_dataset = MIMIIDataset(dataset_path, transform=transform, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return val_loader
