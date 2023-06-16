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
import matplotlib.pyplot as plt


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


def split_tensor(tensor, sampling_rate=100, selection_length=3):
    """
    Splits every input tensor into multiple usable sections.
    Outputs a composite tensor that contains trainable samples.
    """
    selection_samples = int(sampling_rate * selection_length)
    labels = [data[0] for data in tensor]
    selections = [data[1:, :] for data in tensor]
    data_tensors = []
    for j, selection in enumerate(selections):
        for i in range(0, len(selection) - selection_samples, selection_samples):
            end = i + selection_samples
            _slice = selection[i:end]
            if _slice is not False:
                _labels = labels[j].unsqueeze(0)
                data_tensors.append(torch.cat((_labels, _slice)).unsqueeze(0))
            else:
                print("Selection {} rejected".format(j))
    return torch.cat(data_tensors)


def topk_accuracy(true_labels, pred_labels, k=5):
    """
    Calculate the top k accuracy.
    Inputs:
    - true_labels: list of true labels
    - pred_labels: list of raw output scores from the model
    - k: number of top classes to consider
    Outputs:
    - accuracy: top k accuracy
    """

    # Convert the list of predicted scores to a tensor
    pred_scores = torch.tensor(pred_labels)

    # Compute the softmax over the predicted scores to get probabilities
    pred_probs = torch.softmax(pred_scores, dim=-1)
    # Get the top k classes for each prediction
    _, topk_classes = pred_probs.topk(k, dim=-1)
    # Compute the accuracy for each prediction
    correct = 0
    for i in range(len(true_labels)):
        correct += sum([true_labels[i] in pred for pred in topk_classes[i]])
    topk_accuracy = correct / len(true_labels)

    return topk_accuracy, topk_classes


if __name__ == "__main__":
    current_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..', '..', '..', 'Data', 'GazeBase', 'Data'))
    file_list = list_files_recursively(current_path)
