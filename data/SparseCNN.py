import torch
import torch.nn as nn
import sparseconvnet as scn
import torch.nn.functional as F

# from data import get_iterators

# two-dimensional SparseConvNet
class SparseNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential(
            scn.SubmanifoldConvolution(2, 200, 16, 3, True),
            scn.MaxPooling(2, 3, 2),
#             scn.SparseResNet(2, 8, [
#                         ['b', 8, 2, 1],
#                         ['b', 16, 2, 2],
#                         ['b', 24, 2, 2],
#                         ['b', 32, 2, 2]]),
            scn.Convolution(2, 16, 32, 5, 1, False),
            scn.MaxPooling(2, 3, 2),

#             scn.BatchNormReLU(12),
            scn.SparseToDense(2, 32))
        self.spatial_size= self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
#         print(self.spatial_size)
#         self.spatial_size=1500#torch.IntTensor(1500)
#         print(self.spatial_size)
        self.inputLayer = scn.InputLayer(1,self.spatial_size,2)
#         self.linear = nn.Linear(16, 8)

        self.linear = nn.Linear(32, 2)

    def forward(self, x):
        x = self.inputLayer(x)
#         print(x)
#         print(self.sparseModel.nIn)
        x = self.sparseModel(x)
#         print(x)
        x = x.view(-1, 32)
#         print(x)
        x = F.softmax(self.linear(x))
        print(x)
        return x
