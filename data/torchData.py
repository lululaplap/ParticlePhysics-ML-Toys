import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from skimage import io
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['input'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        return {'input': torch.from_numpy(image),
                'target': target}  # torch.from_numpy(target)}


class duneADCdata(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        img = io.imread(img_name)
        img = img.reshape((1, 1500, 1500))
        img_class = self.data.iloc[idx, 1]
#         print(img_class)
        target = np.zeros(2, dtype=int)
        target[img_class] = 1
        sample = {"input": img, "target": img_class}
#         print(sample)
        if self.transform:
            sample = self.transform(sample)
#         print(sample)
        return sample
    

# d = duneADCdata("mydata.csv","./")
# print(d[1])
