"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:08:55
 * @modify date 2023-05-11 12:08:55
 * @desc [description]
"""
import os
import glob
import torch
import scipy
import numpy as np
from PIL import Image
import scipy.io as sio


def list_files(directory):
    "List all files in the dataset directory."
    files = []
    for filename in os.listdir(directory):
        # if filename.endswith(".csv"):
        single = os.path.join(directory, filename)
        files.append(single)
    return files


def list_files_recursively(path):
    "List all files in the dataset directory recursively."
    file_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_list.append(os.path.join(dirpath, filename))
    return file_list


def retrieve_semantic_label(file):
    mat = sio.loadmat(file)
    mat = np.array(mat['data'])
    return mat


def load_img_sem_data(img_folder='OSIE_imgs',
                      sem_folder='OSIE_tags',
                      img_extension='*.jpg',
                      sem_extension='*.mat'):
    """
    Load images and semantic labels from the OSIE dataset.
    Inputs:
    - img_folder: folder containing the images
    - sem_folder: folder containing the semantic labels
    Outputs:
    - img_tensor_all: tensor containing all images
    - sem_tensor_all: tensor containing all semantic labels
    """
    img_paths = sorted(glob.glob(os.path.join(img_folder, img_extension)))
    sem_paths = sorted(glob.glob(os.path.join(sem_folder, sem_extension)))
    img_tensors = []
    sem_tensors = []

    for img_path, sem_path in zip(img_paths, sem_paths):
        # Load image and convert to tensor
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

        # Load semantic labels and convert to tensor
        sem_mat = scipy.io.loadmat(sem_path)
        sem_tensor = torch.tensor(sem_mat['data'], dtype=torch.float32)

        img_tensors.append(img_tensor)
        sem_tensors.append(sem_tensor)

    # Concatenate tensors along the first dimension
    img_tensor_all = torch.stack(img_tensors, dim=0)
    sem_tensor_all = torch.stack(sem_tensors, dim=0)

    return img_tensor_all, sem_tensor_all


if __name__ == "__main__":
    current_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..', '..', '..', 'Data', 'GazeBase', 'Data'))
    file_list = list_files_recursively(current_path)
