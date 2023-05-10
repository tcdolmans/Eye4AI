import os
import re
import sys
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from GPMBT import MultimodalBottleneckTransformer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.utils import load_img_sem_data # noqa


class MultimodalDataset(Dataset):
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
        s_num = torch.tensor(int(s_num))

        img_data = self.img_tensor[s_num - 1001]
        sem_data = self.sem_tensor[s_num - 1001]

        return et_data, img_data, sem_data, p_num, s_num

    def __len__(self):
        return len(self.et_files)


def train_model(model, train_dataloader, device, loss_function, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (et_data, img_data, sem_data, p_num, s_num) in enumerate(train_dataloader):
            et_data, img_data, sem_data, p_num, s_num = (et_data.to(device),
                                                         img_data.to(device),
                                                         sem_data.to(device),
                                                         p_num.to(device),
                                                         s_num.to(device))
            out = model(et_data, img_data, sem_data, p_num)

            if model.mode == "et_reconstruction":
                loss = loss_function(out, et_data)
            elif model.mode == "classification":
                loss = loss_function(out, p_num)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   factor=0.5,
                                                                   patience=2,
                                                                   verbose=True)
            scheduler.step(loss)

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item()}")  # noqa: E501


def test_model(model, test_dataloader, device, loss_function):
    model.eval()
    test_loss = 0
    snippet_length = 30
    with torch.no_grad():
        for et_data, img_data, sem_data, p_num, s_num in test_dataloader:
            et_data, img_data, sem_data, p_num, s_num = (et_data.to(device),
                                                         img_data.to(device),
                                                         sem_data.to(device),
                                                         p_num.to(device),
                                                         s_num.to(device))
            # Pass None for the et_data since we are reconstructing the et_data
            if model.mode == "et_reconstruction":
                et_data_snippet = et_data[:, :snippet_length, :]
                reconstructed_et = model(et_data_snippet, img_data, sem_data, p_num)
                print(et_data_snippet.shape, reconstructed_et.shape)
                loss = loss_function(reconstructed_et[:, snippet_length:, :],
                                     et_data[:, snippet_length:, :])
            elif model.mode == "classification":
                pred_p_num = model(et_data, img_data, sem_data)
                loss = loss_function(pred_p_num, p_num)
            test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    return test_loss


def objective(trial, train_dataset, test_dataset, device, mode):
    # num_layers = trial.suggest_int("num_layers", 2, 8)
    if mode == "et_reconstruction":
        p_num_provided = True
    elif mode == "classification":
        p_num_provided = False
    config = {
        "num_layers": 2,
        "heads": 4,  # trial.suggest_int("heads", 4, 24, step=4),
        "forward_expansion": 2,  # trial.suggest_int("forward_expansion", 2, , step=2),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "lr": trial.suggest_float("lr", 0.5, 1),
        "batch_size": 16,  # trial.suggest_int("batch_size", 64, 256, step=64),
        "n_bottlenecks": trial.suggest_int("n_bottlenecks", 4, 16, step=4),
        "fusion_layer": 2,  # trial.suggest_int("fusion_layer", 2, num_layers),
        "num_epochs": 2,  # trial.suggest_int("num_epochs", 50, 200, step=50),
        "L2": trial.suggest_float("L2", 1e-5, 1e-3),
        "p_num_embed_dim": 1,
        "et_embed_dim": 192,
        "et_patch_size": 15,
        "et_seq_len": 300,
        "et_dim": 4,
        "et_stride": 1,
        "img_embed_dim": 192,
        "img_patch_size": 25,
        "img_stride": 12,
        "sem_embed_dim": 192,
        "sem_patch_size": 25,
        "sem_stride": 12,
        "modalities": ["et", "img", "sem"],
        "num_classes": 2,
        "device": device,
        "mode": mode,
        "p_num_provided": p_num_provided,
        "pad_id": 0,
    }

    model = MultimodalBottleneckTransformer(config).to(config["device"])
    # summary(model, input_size=(config["batch_size"], 4, 300,
    #                            config["batch_size"], 3, 600, 800,
    #                            config["batch_size"], 12, 600, 800))
    if config["mode"] == "et_reconstruction":
        loss_function = nn.MSELoss()
    elif config["mode"] == "classification":
        loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["L2"])

    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    train_model(model, train_dl, config["device"], loss_function, optimizer, config["num_epochs"])
    test_loss = test_model(model, test_dl, config["device"], loss_function)
    return test_loss


def hyperparameter_optimization(train_dataset, test_dataset, device, mode, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial,
                                           train_dataset,
                                           test_dataset,
                                           device,
                                           mode), n_trials=n_trials)
    return study.best_params


if __name__ == "__main__":
    """
    This implementation assumes that the participant numbers are used as filenames for the images
    and semantic data files. Adjust the file naming conventions and data loading as needed based
    on your specific dataset organization.
    """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    mode = 'et_reconstruction'

    # Specify the paths to your data folders
    base_path = os.getcwd()
    train_et_folder = os.path.join(base_path, "pipeline", "osieData", "osie_tensors")
    test_et_folder = os.path.join(base_path, "pipeline", "osieData", "osie_tensors")
    img_tensor, sem_tensor = load_img_sem_data(
        img_folder='pipeline/OSIE_imgs',
        sem_folder='pipeline/OSIE_tags')

    # Prepare your data and Dataset
    train_dataset = MultimodalDataset(train_et_folder, img_tensor, sem_tensor)
    test_dataset = MultimodalDataset(test_et_folder, img_tensor, sem_tensor)

    # Hyperparameter optimization
    best_params = hyperparameter_optimization(train_dataset,
                                              test_dataset,
                                              device,
                                              mode,
                                              n_trials=3)
