"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:08:55
 * @modify date 2023-05-11 12:08:55
 * @desc [description]
"""
import os
import scipy.io as sio
import numpy as np


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


if __name__ == "__main__":
    current_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..', '..', '..', 'Data', 'GazeBase', 'Data'))
    file_list = list_files_recursively(current_path)
