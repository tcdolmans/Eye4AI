import os
import sys
import time
import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from pipeline.utils import list_files # noqa


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {(end - start):.6f} seconds")
        return result
    return wrapper


def alert_deprication(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} depricated and took {(end - start):.6f} seconds")
        return result
    return wrapper


# @time_it
def replace_nans(input_data, need_sanity_check=False):
    """
    Takes some input tensor suitable for PyTorch, then linearly interpolates the Nans away.
    This works decently well, but can be improved by beign tested for speed.
    """
    # Below assumes that the input dimension you care to evaluate by
    # will be returned by simply calling len().
    # TODO: Generalise to differently ranked inputs.
    # @time_it
    def pad_with_interp(data):
        bad_indexes = np.isnan(data)
        good_indexes = np.logical_not(bad_indexes)
        good_data = data[good_indexes]
        if len(good_data) == 0:
            return False
        elif len(good_data) / len(data) < 0.85:
            #  The sample is rejected because it did not met the threshold of .85.
            return False
        # Craft linearly interpolated data over the seleceted data.
        interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
        data[bad_indexes] = interpolated
        return data

    output = torch.tensor(np.apply_along_axis(pad_with_interp, 0, input_data))
    if need_sanity_check:
        print(output.shape)
    return output


# @time_it
def downsample(input_tensor, dsf=10):
    """
    Takes input_tensor and donsamples the data by down_sampling_frequency(dsf)
    by return a shortened array which contains the mode for each of the sample windows.
    Can be altered to return !mode, e.g.,mean, distribution, etc.
    Little bit of unpacking and looping, hence the decorator;
    please remember to switch it off when you are not testing
    """
    ds = []
    for i in range(0, len(input_tensor), dsf):
        selection = stats.mode(input_tensor[i: i+dsf], axis=0, keepdims=True)[0][0]
        ds.append(torch.tensor(np.array(selection)).unsqueeze(0))
    return torch.cat(ds)


# @time_it
def downsample_task_tensor(file, dsf=10):
    tensor = torch.load(file)
    for i, t in enumerate(tensor):
        labels = t[0].unsqueeze(0)
        _data = t[1:]
        _data = replace_nans(downsample(_data))
        tensor[i] = torch.cat((labels, _data))
    return tensor


def split_tensor(tensor, sampling_rate=100, selection_length=3):
    """
    Splits every input tensor into multiple usable sections.
    Outputs a composite tensor that contains trainable samples.
    Gets faster as dsf increases in downsample_task_tensor().
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
                # _slice = downsample(_slice)
                # _slice = replace_nans(_slice)
                data_tensors.append(torch.cat((_labels, _slice)).unsqueeze(0))
            else:
                print("Selection {} rejected".format(j))
    return torch.cat(data_tensors)


@alert_deprication
def split_ds_tensor(tensor, sampling_rate=100, selection_length=3):
    selection_samples = int(sampling_rate * selection_length)
    label = tensor[0].unsqueeze(0)
    data = tensor[1:]
    print(label.shape, data.shape)
    data_t = []
    for i in range(0, len(data) - selection_samples, selection_samples):
        end = i + selection_samples
        _slice = data[i:end]
        if _slice is not False:
            data_t.append(torch.cat((label, _slice)).unsqueeze(0))
        else:
            print("Selection rejected")
    return data_t


@time_it
def return_data_loaders(files, train_idxs, test_idxs, batch_size=64):
    train = files[:train_idxs]  # NOTE: usually is :train_idxs, but this is for testing
    test = files[-test_idxs:]  # NOTE: usually is -test_idxs:, but this is for testing
    train_data_tensors = []
    test_data_tensors = []
    print("Loading training data")
    for file in train:
        # print(file[-15:-10])
        split_tensors = split_tensor(torch.load(file))
        train_data_tensors.append(split_tensors)
    print("Loading testing data")
    for file in test:
        # print(file[-15:-10])
        split_tensors = split_tensor(torch.load(file))
        test_data_tensors.append(split_tensors)
    train_dataloader = DataLoader(torch.cat(train_data_tensors),
                                  batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(torch.cat(test_data_tensors),
                                 batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    """@Ippa"""
    # NOTE: Please teach me how to do this elegantly:
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Pipeline', 'tensors'))
    files = list_files(folder)
    # Thank you.

    for file in files:
        tensor = downsample_task_tensor(file)
        name_string = "DS10_{}_tensor.pt".format(file[-15:-10])
        torch.save(tensor, name_string)

    # train_dataloader, test_dataloader = return_data_loaders(files, 125, 2, 64)
    # # Train test split based on task. Eval acc per task to select interests.
    # train_sample = next(iter(train_dataloader))
    # test_sample = next(iter(test_dataloader))
    # print(train_sample.shape, test_sample.shape)
    # for sample in train_sample:
    #     print(sample)
