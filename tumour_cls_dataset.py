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
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=transform_opts.mean,
                                 std=transform_opts.std)
        ]) if transform_opts else None

        self.dataset = ImageFolder(root=self.img_dir, transform=self.transform)
        # self.train_size = int(data_opts.train_ratio * self.__len__())
        # self.test_size = int(data_opts.test_ratio * self.__len__())
        # self.val_size = self.__len__() - self.train_size - self.test_size
        #
        # self.train, self.val, self.test = data.random_split(dataset=self.dataset,
        #                                 lengths=[self.train_size, self.val_size, self.test_size],
        #                                 generator=torch.Generator().manual_seed(data_opts.seed))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label


