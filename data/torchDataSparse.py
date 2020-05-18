import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from skimage import io
import numpy as np
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import sys

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
    def __init__(self, csv_file, root_dir, transform=None,batch_size=1):
        self.maxSize = 200
        self.data = pd.read_csv(csv_file)
        self.coords = []
        self.features = []
        self.targets = []
        self.root_dir = root_dir
        self.transform = transform
        toolbar_width = len(self.data)
        self.batch_size = batch_size
        sys.stdout.write("Loading dataset:\n")
#         sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
#         sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
#         self.len = 0
#         for i in range(len(self.data)):
#                 img_name = os.path.join(self.root_dir, self.data.iloc[i, 0])
#                 img = io.imread(img_name)
#                 img = coo_matrix(img)
#                 features = torch.FloatTensor(img.data)
#                 if features.size(0) <= self.maxSize and features.size(0) != 0:
#                     self.len += 1

#         nBatches = int(np.ceil(self.len/batch_size))
#         print(nBatches)
#         for batchIdx in range(nBatches):
        for i in range(len(self.data)):
            img_name = os.path.join(self.root_dir, self.data.iloc[i, 0])
            img = io.imread(img_name)
            img = coo_matrix(img)
            features = torch.FloatTensor(img.data)
            if features.size(0) <= self.maxSize and features.size(0) != 0:
                coordsx = img.row
                coordsy = img.col
                coords = np.dstack((coordsx,coordsy,np.ones(features.size(0))*i))#batch_size))
                coords = torch.FloatTensor(coords)


                if not(np.isnan(coords).any()) or not(np.isnan(features).any()):
                    pad1 = (0,self.maxSize-features.size(0))
                    pad2 = (0,0,0,self.maxSize-features.size(0))

                    coords = F.pad(coords,pad2,'constant',0)
                    coords[0,:,2] = batch_size

                    features = F.pad(features,pad1,'constant',0)

                    target = self.data.iloc[i, 1]
                    self.coords.append(coords)
                    self.features.append(features)
                    self.targets.append(target)
                else:
                    print("NaN")
            sys.stdout.write("{}%\r".format(int(i*100/(len(self.data)))))
            sys.stdout.flush()

    #         sys.stdout.write("]\n")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {"input": (self.coords[idx],self.features[idx]), "target": self.targets[idx]}
      

        if self.transform:
            sample = self.transform(sample)
        return sample
    

# d = duneADCdata("mydata.csv","./")
# print(d[1])
