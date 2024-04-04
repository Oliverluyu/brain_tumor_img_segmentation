import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from os.path import join


class tumourClassificationDataset(data.Dataset):
    def __init__(self, data_opts, data_split, transform_opts=None):
        super(tumourClassificationDataset, self).__init__()
        self.img_dir = join(data_opts.root_dir, data_split)

        self.transform = transforms.Compose([
            transforms.Resize(tuple(transform_opts.img_shape)),
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=tuple(transform_opts.random_rotate)),
            transforms.RandomHorizontalFlip(p=transform_opts.h_flip_prob),
            transforms.RandomVerticalFlip(p=transform_opts.v_flip_prob),
            transforms.Normalize(mean=transform_opts.mean,
                                 std=transform_opts.std)
        ]) if transform_opts else None

        self.dataset = ImageFolder(root=self.img_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label


