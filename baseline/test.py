import numpy as np
import torch

a = np.array([[1, 2], [3, 4]])

print(torch.cat(torch.tensor(a), dim=0))