import os
import re
import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """
    Dataset class for the multimodal data, in this case:
    - Eye-tracking data
    - Images
    - Semantic labels
    Inputs:
    - et_folder: folder containing the eye-tracking data
    - img_tensor: tensor containing the images
    - sem_tensor: tensor containing the semantic labels
    Outputs (per item):
    - et: eye-tracking data
    - img: image data
    - sem: semantic label data
    - p_num: participant number
    - s_num: stimulus number
    TODO: Change p_num to also include session number
    """
    def __init__(self, et_folder, img_tensor, sem_tensor):
        self.et_folder = et_folder
        self.et_files = sorted(os.listdir(self.et_folder))
        self.img_tensor = img_tensor
        self.sem_tensor = sem_tensor

    def __getitem__(self, index):
        et_path = os.path.join(self.et_folder, self.et_files[index])
        et_data = torch.load(et_path)

        # Extract the participant number and remove the first row
        pattern = re.compile(r'(?P<participant_number>\w+)_(?P<stimulus_number>\d+)\.pt')
        match = pattern.match(self.et_files[index])
        p_num = match.group('participant_number')[0]
        s_num = int(match.group('stimulus_number'))
        p_num = torch.tensor(int(p_num))
        s_num = torch.tensor(int(s_num))  # TODO: Integrate this info, unused now.

        img_data = self.img_tensor[s_num - 1001]
        sem_data = self.sem_tensor[s_num - 1001]

        return {"et": et_data, "img": img_data, "sem": sem_data, "p_num": p_num}

    def __len__(self):
        return len(self.et_files)


class GazeBaseDataset(Dataset):
    """
    Dataset class for the GazeBase data.
    Inputs:
    - files: list of files to load
    - device: device to load the data on
    - sampling_rate: sampling rate of the data
    - selection_length: length of the selection in seconds
    Outputs (per item):
    - et: eye-tracking data
    - p_num: participant number
    """
    def __init__(self, files, device, sampling_rate=100, selection_length=3):
        self.files = files
        self.device = device
        self.sampling_rate = sampling_rate
        self.selection_length = selection_length
        self.tensors = self.load_and_split_tensors()

    def load_and_split_tensors(self):
        data_tensors = []
        for file in self.files:
            tensor = torch.load(file)
            split_tensors = self.split_tensor(tensor)
            data_tensors.extend(split_tensors)
        return data_tensors

    def split_tensor(self, tensors):
        """
        Splits every input tensor into multiple usable sections.
        Outputs a composite tensor that contains trainable samples.
        """
        data_tensors = []
        for tensor in tensors:
            label = tensor[0:1, :].type(torch.LongTensor)[0, 0] - 1
            selections = tensor[1:, :].float()  # Whole tensor, exclude label
            selections.requires_grad = True
            selection_samples = int(self.sampling_rate * self.selection_length)
            for i in range(0, len(selections) - selection_samples, selection_samples):
                end = i + selection_samples
                _slice = selections[i:end]
                if _slice is not False:
                    data_tensors.append({"et": _slice.to(self.device),
                                        "p_num": label.to(self.device)})
        return data_tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)
