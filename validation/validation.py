import torch

tensor1 = torch.tensor([[[1., 0], [2., 3.]], [[4., 0], [5., 6.]]])
tensor1.to_sparse()

print(" First Tensor: ", tensor1)