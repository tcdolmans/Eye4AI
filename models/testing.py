import numpy as np
import sys
import os
import torch
import torch.nn as nn
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from pipeline.utils import list_files # noqa
from pipeline.dataset_constructor import return_data_loaders # noqa


# folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pipeline', 'tensors'))
# files = list_files(folder)
# train_dataloader, test_dataloader = return_data_loaders(files, 125, 1, batch_size=64)
# for i, data in enumerate(train_dataloader):
#     inputs, labels = data

# conv_layer = nn.Conv1d(in_channels=4, out_channels=, kernel_size=250, stride=250)
# # output_tensor = conv_layer(inputs.permute(0, 2, 1).float())
# # print(output_tensor.shape)
# a = np.array([[1, 1], [2, 2], [3, 3]])
# a = np.insert(a, 0, 5, axis=0)
# print(a)

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
  
# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device{torch.cuda.current_device()}")
        
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")