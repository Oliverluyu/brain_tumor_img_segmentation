import torch.utils.data as data
from torchvision import transforms
from torchvision.io import read_image

from os import listdir
from os.path import join


class tumourSegmentationDataset(data.Dataset):
    def __init__(self, data_opts, data_split, transform_opts=None):
        super(tumourSegmentationDataset, self).__init__()
        image_dir = join(data_opts.root_dir, data_split, 'image')
        target_dir = join(data_opts.root_dir, data_split, 'label')
        self.image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir)])
        self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir)])
        assert len(self.image_filenames) == len(self.target_filenames)

        self.image_transform = transforms.Compose([
            transforms.Resize(tuple(transform_opts.img_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_opts.mean,
                                 std=transform_opts.std)
        ]) if transform_opts else None

        self.target_transform = transforms.Compose([
            transforms.Resize(tuple(transform_opts.img_shape), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ]) if transform_opts else None

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = read_image(self.image_filenames[index])
        target = read_image(self.target_filenames[index])

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target