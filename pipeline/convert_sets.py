import pandas as pd
import numpy as np
import os
import torch
from utils import list_files
from dataset_constructor import replace_nans, downsample


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
                task_tensor = []
                all_parts = list_files(task)
                for part in all_parts:
                    # Varying number of participants
                    pn = int(part[-14:-11])
                    print("Currently on ", pn)
                    p_row = torch.tensor(np.array((pn, pn, pn, pn))).unsqueeze(0)
                    df = pd.read_csv(part)
                    # select only the columns we need, namely the 'n', 'x', 'y', 'dP' columns  
                    data = df[['n', 'x', 'y', 'dP']]
                    data = data.values
                    data = replace_nans(downsample(data))
                    data = torch.cat((p_row, data))
                    task_tensor.append(data)
                name_string = "DS10_{}_{}_tensor.pt".format(sesh_name, i)
                torch.save(task_tensor, name_string)


if __name__ == "__main__":
    """"Creating Tensors below"""
    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                                'Data', 'GazeBase', 'Data'))
    create_task_tensor(current_path)
