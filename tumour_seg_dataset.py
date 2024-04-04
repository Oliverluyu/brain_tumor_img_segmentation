import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from os import listdir
from os.path import join


class tumourSegmentationDataset(data.Dataset):
    def __init__(self, data_opts, data_split, transform_opts=None):
        super(tumourSegmentationDataset, self).__init__()

        if not data_opts.with_notumour:
            image_dir = join(data_opts.root_dir, data_split, 'images', 'tumour-present')
            target_dir = join(data_opts.root_dir, data_split, 'masks', 'tumour-present')
            self.image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir)])
            self.target_filenames = sorted([join(target_dir, x) for x in listdir(target_dir)])
        else:
            self.image_filenames = []
            self.target_filenames = []

            # include both cancerous and no tumour images/masks
            # define subdirectories
            sub_dirs = ['tumour-present', 'no-tumour']

            for sub_dir in sub_dirs:
                image_dir = join(data_opts.root_dir, data_split, 'images', sub_dir)
                target_dir = join(data_opts.root_dir, data_split, 'masks', sub_dir)

                images = sorted(listdir(image_dir))
                targets = sorted(listdir(target_dir))

                self.image_filenames += [join(image_dir, img) for img in images]
                self.target_filenames += [join(target_dir, tgt) for tgt in targets]

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
        image = Image.open(self.image_filenames[index]).convert('RGB')
        target = Image.open(self.target_filenames[index]).convert('L')

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        target = target.squeeze().long()  # remove channel dimension and convert masks to long

        return image, target
