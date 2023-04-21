import os
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from utils import load_img_sem_data
from torch.utils.data import Dataset, DataLoader
from GPMBT import MultimodalBottleneckTransformer


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
        participant_number = et_data[0, 0].item()
        stimulus_number = None
        et_data = et_data[1:]

        img_data = self.img_tensor[stimulus_number]
        sem_data = self.sem_tensor[stimulus_number]

        return et_data, img_data, sem_data, participant_number, stimulus_number


def train_model(model, train_dataloader, device, loss_function, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (et_data, img_data, sem_data, p_num, s_num) in enumerate(train_dataloader):
            et_data, img_data, sem_data, p_num, s_num = et_data.to(device),
            img_data.to(device), sem_data.to(device), p_num.to(device), s_num.to(device)
            out = model(et_data, img_data, sem_data)

            if model.mode == "et_reconstruction":
                loss = loss_function(out, et_data)
            elif model.mode == "classification":
                loss = loss_function(out, p_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item()}")  # noqa: E501


def test_model(model, test_dataloader, device, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for et_data, img_data, sem_data, p_num, s_num in test_dataloader:
            et_data, img_data, sem_data, p_num, s_num = et_data.to(device),
            img_data.to(device), sem_data.to(device), p_num.to(device), s_num.to(device)
            # Pass None for the et_data since we are reconstructing the et_data
            if model.mode == "et_reconstruction":
                reconstructed_et = model(None, img_data, sem_data)
                loss = loss_function(reconstructed_et, et_data)
            elif model.mode == "classification":
                pred_p_num = model(et_data, img_data, sem_data)
                loss = loss_function(pred_p_num, p_num)
            test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    return test_loss


def objective(trial, train_dataset, test_dataset, device, mode):
    config = {
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "heads": trial.suggest_int("heads", 4, 16, step=4),
        "forward_expansion": trial.suggest_int("forward_expansion", 2, 8, step=2),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3, step=5e-5),
        "batch_size": trial.suggest_int("batch_size", 64, 256, step=64),
        "n_bottlenecks": trial.suggest_int("n_bottlenecks", 1, 4),
        "fusion_layer": trial.suggest_int("fusion_layer", 2, "num_layers"),
        "num_epochs": trial.suggest_int("num_epochs", 10, 50, step=10),
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
        "num_classes": 335,
        "device": device,
        "mode": mode,
        "pad_id": 0,
    }

    model = MultimodalBottleneckTransformer(..., **config, mode="et_reconstruction").to(device)
    if config["mode"] == "et_reconstruction":
        loss_function = nn.MSELoss()
    elif config["mode"] == "classification":
        loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    train_model(model, train_dataloader, device, loss_function, optimizer, config["num_epochs"])
    test_loss = test_model(model, test_dataloader, device, loss_function)
    return test_loss


def hyperparameter_optimization(train_dataset, test_dataset, device, mode, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial,
                                           train_dataset,
                                           test_dataset,
                                           mode,
                                           device), n_trials=n_trials)
    return study.best_params


if __name__ == "__main__":
    """
    This implementation assumes that the participant numbers are used as filenames for the images
    and semantic data files. Adjust the file naming conventions and data loading as needed based
    on your specific dataset organization.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = "et_reconstruction"

    # Specify the paths to your data folders
    train_et_folder = "path/to/train_et_folder"
    test_et_folder = "path/to/test_et_folder"
    img_tensor, sem_tensor = load_img_sem_data()

    # Prepare your data and Dataset
    train_dataset = MultimodalDataset(train_et_folder, img_tensor, sem_tensor)
    test_dataset = MultimodalDataset(test_et_folder, img_tensor, sem_tensor)

    # Hyperparameter optimization
    best_params = hyperparameter_optimization(train_dataset, test_dataset, device, mode, n_trials=50)
