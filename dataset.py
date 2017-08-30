# Custom dataset
from PIL import Image
import torch.utils.data as data
import os


class FacadesDataset(data.Dataset):
    def __init__(self, image_dir='facades/', subfolder='train', transform=None):
        super(FacadesDataset, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.transform = transform

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn)
        input = img.crop((img.width // 2, 0, img.width, img.height))
        target = img.crop((0, 0, img.width // 2, img.height))
        if self.transform is not None:
            input = self.transform(input)
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class CityscapeDataset(data.Dataset):
    def __init__(self, image_dir='cityscape/', subfolder='train', transform=None):
        super(CityscapeDataset, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.transform = transform

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn)
        input = img.crop((img.width // 2, 0, img.width, img.height))
        target = img.crop((0, 0, img.width // 2, img.height))
        if self.transform is not None:
            input = self.transform(input)
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)