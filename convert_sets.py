import pandas as pd
import numpy as np
import os
import torch
from utils import list_files
from scipy import stats
import time


def replace_nans(input_tensor):

    def pad(data):
        bad_indexes = np.isnan(data)
        good_indexes = np.logical_not(bad_indexes)
        good_data = data[good_indexes]
        if len(good_data) == 0:
            return False
        elif len(good_data) / len(data) < 0.85:
            # print("Sample Rejected")
            return False
        interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
        data[bad_indexes] = interpolated
        return data
    output = torch.tensor(np.apply_along_axis(pad, 0, input_tensor))
    # print(output)
    return output


def downsample(input_tensor, dsf=10):
    """
    Takes input_tensor and donsamples the data by dsf by return a shorteend array
    that contains the mode for each of the sample windows.
    """
    ds_t = []
    for i in range(0, len(input_tensor), dsf):
        selection = stats.mode(input_tensor[i: i+dsf], axis=0, keepdims=True)[0][0]
        ds_t.append(torch.tensor(np.array(selection)).unsqueeze(0))
    return torch.cat(ds_t)


def create_round_tensor(file_path):
    """
    Creates a single torch tensor file with 'output_name' from the GaseBase
    data structure. The tensor contains 9 sessions and no distinction between
    the tasks as of yet.
    """
    files = list_files(file_path)
    # session_tensors = []
    for i, file in enumerate(files):
        round_tensor = []
        # 9 Rounds
        round_i = list_files(file)
        round_name = round_i[-7:]
        for session in round_i:
            print("Currently on: " + session[-10:])
            # 2 Sessions per round
            sesh = list_files(session)
            for i, task in enumerate(sesh):
                # 7 Tasks per session
                all_parts = list_files(task)
                for part in all_parts:
                    # Varying number of participants
                    df = pd.read_csv(part)
                    data = df.values
                    data = data[:, [0, 1, 2, 4]]
                    pn = int(part[-14:-11])
                    participant_row = np.array((pn, pn, pn, pn))
                    data = np.insert(data, 0, participant_row, axis=0)
                    part_tensor = torch.from_numpy(data)
                    round_tensor.append(part_tensor.unsqueeze(0))  # Maybe this should not be unsquoze
                    print(len(round_tensor))
        name_string = "{}_tensor.pt".format(round_name)
        round_tensor = torch.cat(round_tensor)
        torch.save(round_tensor, name_string)


def create_task_tensor(file_path):
    """
    Creates a single torch tensor file with 'output_name' from the GaseBase
    data structure. The tensor contains 9 sessions and no distinction between
    the tasks as of yet.
    """
    files = list_files(file_path)
    for i, file in enumerate(files):
        # 9 Rounds
        round_i = list_files(file)
        for session in round_i:
            print("Currently on: " + session[-10:])
            # 2 Sessions per round
            sesh = list_files(session)
            sesh_name = session[-10:-3] + session[-2:]
            for i, task in enumerate(sesh):
                # 7 Tasks per session
                start_time = time.time()
                task_tensor = []
                all_parts = list_files(task)
                for part in all_parts:
                    # Varying number of participants
                    pn = int(part[-14:-11])
                    print("Currently on ", pn)
                    p_row = torch.tensor(np.array((pn, pn, pn, pn))).unsqueeze(0)
                    df = pd.read_csv(part)
                    data = df.values
                    data = data[:, [0, 1, 2, 4]]
                    data = downsample(data, 10)
                    data = torch.cat((p_row, data))
                    data = replace_nans(data).unsqueeze(0)
                    task_tensor.append(data)
                name_string = "DS10_{}_{}_tensor.pt".format(sesh_name, i)
                torch.save(torch.tensor(task_tensor), name_string)
                print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    """"Creating Tensors below"""
    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                                'Data', 'GazeBase', 'Data'))
    create_task_tensor(current_path)
