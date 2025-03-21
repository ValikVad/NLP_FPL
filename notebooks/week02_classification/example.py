import torch 
import torch.nn as nn
# input = torch.randn(5, 5)
# m = nn.Dropout(p=0.5)
# print(input)
# output = m(input)
# print(output)


# With Learnable Parameters
m = nn.BatchNorm1d(5)
# Without Learnable Parameters
m = nn.BatchNorm1d(5, affine=False)
input = torch.randn(5, 5)
output = m(input)
print(input, "\n\n", output)