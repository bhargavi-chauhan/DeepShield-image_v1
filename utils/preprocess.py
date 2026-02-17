import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepfakeImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = ["fake", "real"]
        self.class_to_idx = {"fake": 0, "real": 1}

        self.samples = []
        for label in self.classes:
            folder = os.path.join(root_dir, label)
            for file in os.listdir(folder):
                self.samples.append((os.path.join(folder, file), self.class_to_idx[label]))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label
