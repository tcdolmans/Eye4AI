import os
import torch
import optuna
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from GPMBT import MultimodalBottleneckTransformer


class MultimodalDataset(Dataset):
    def __init__(self, et_folder, img_folder, sem_folder, transform=None):
        self.et_folder = et_folder
        self.img_folder = img_folder
        self.sem_folder = sem_folder
        self.transform = transform

        self.et_files = sorted(os.listdir(self.et_folder))
        self.img_files = sorted(os.listdir(self.img_folder))
        self.sem_files = sorted(os.listdir(self.sem_folder))
        self.transform = transforms.Compose([
            transforms.Resize((800, 600)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        et_path = os.path.join(self.et_folder, self.et_files[index])
        et_data = torch.load(et_path)

        # Extract the participant number and remove the first row
        participant_number = et_data[0, 0].item()
        et_data = et_data[1:]

        img_path = os.path.join(self.img_folder, f"{participant_number}.jpg")
        img_data = Image.open(img_path)

        sem_path = os.path.join(self.sem_folder, f"{participant_number}.pt")
        sem_data = torch.load(sem_path)
        sem_data = self.transform(sem_data)

        return et_data, img_data, sem_data, participant_number

    def __len__(self):
        return len(self.et_files)


def train_model(model, train_dataloader, device, loss_function, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (et_data, img_data, sem_data, p_num) in enumerate(train_dataloader):
            et_data, img_data, sem_data, p_num = et_data.to(device),
            img_data.to(device), sem_data.to(device), p_num.to(device)
            reconstructed_et = model(et_data, img_data, sem_data)

            loss = loss_function(reconstructed_et, p_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item()}")  # noqa: E501


def test_model(model, test_dataloader, device, loss_function):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for et_data, img_data, sem_data, ground_truth in test_dataloader:
            et_data, img_data, sem_data, ground_truth = et_data.to(device),
            img_data.to(device), sem_data.to(device), ground_truth.to(device)
            # Pass None for the et_data since we are reconstructing the et_data
            reconstructed_et = model(None, img_data, sem_data)

            loss = loss_function(reconstructed_et, ground_truth)
            test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    return test_loss


def objective(trial, train_dataloader, test_dataloader, device):
    config = {
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "heads": trial.suggest_int("heads", 1, 4),
        "forward_expansion": trial.suggest_int("forward_expansion", 2, 8),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "batch_size": trial.suggest_int("batch_size", 16, 128),
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
        "n_bottlenecks": 4,
        "pad_id": 0,
        "mode": "et_reconstruction",
        "fusion_layer": 2,
    }

    model = MultimodalBottleneckTransformer(..., **config, mode="et_reconstruction").to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train_model(model, train_dataloader, device, loss_function, optimizer, num_epochs=10)
    test_loss = test_model(model, test_dataloader, device, loss_function)

    return test_loss


def hyperparameter_optimization(train_dataloader, test_dataloader, device, n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial,
                                           train_dataloader,
                                           test_dataloader,
                                           device), n_trials=n_trials)
    return study.best_params


if __name__ == "__main__":
    """
    This implementation assumes that the participant numbers are used as filenames for the images
    and semantic data files. Adjust the file naming conventions and data loading as needed based
    on your specific dataset organization.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Specify the paths to your data folders
    train_et_folder = "path/to/train_et_folder"
    train_img_folder = "path/to/train_img_folder"
    train_sem_folder = "path/to/train_sem_folder"
    test_et_folder = "path/to/test_et_folder"
    test_img_folder = "path/to/test_img_folder"
    test_sem_folder = "path/to/test_sem_folder"

    # Prepare your data and DataLoader
    train_dataset = MultimodalDataset(train_et_folder, train_img_folder, train_sem_folder)
    test_dataset = MultimodalDataset(test_et_folder, test_img_folder, test_sem_folder)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Hyperparameter optimization
    best_params = hyperparameter_optimization(train_dataloader, test_dataloader, device)
