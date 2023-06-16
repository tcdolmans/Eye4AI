"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:03:56
 * @modify date 2023-05-11 12:03:56
 * @desc [description]
"""
import os
import sys
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
from MLP import ETMLP
from MBT import MultimodalBottleneckTransformer
from dataloaders import MultimodalDataset, GazeBaseDataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.utils import load_img_sem_data, topk_accuracy # noqa


def train_model(model, train_dataloader, device, loss_function, optimizer, num_epochs, mode):
    """
    Main training loop for the model.
    Inputs:
    - model: model to train
    - train_dataloader: dataloader containing the training data. Assumes the presence of:
        - et: eye-tracking data
        - img: image data
        - sem: semantic label data
        - p_num: participant number
        - s_num: stimulus number
        - TODO: Generalise to any number of inputs so it can be used in different settings
    - device: device to train on, e.g. cuda
    - loss_function: loss function to use, varies per model "mode"
    - optimizer: optimizer to use, e.g. Adam
    - num_epochs: number of epochs to train for
    """
    snippet_length = 200
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            data = {key: value.to(device) for key, value in data.items()}
            p_num = data.pop("p_num")
            data_copy = data.copy()
            if mode == "et_reconstruction":
                data_copy["et"] = data["et"][:, :snippet_length, :]
                out = model(data_copy, p_num)
                loss = loss_function(out,
                                     data["et"][:, snippet_length:, :])
            elif mode == "classification":
                out = model(data)
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

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Step: {i}/{len(train_dataloader)}")


def test_model(model, test_dataloader, device, loss_function, mode):
    """
    Main testing loop for the model.
    Inputs:
    - model: model to test
    - test_dataloader: dataloader containing the testing data. Assumes the presence of:
        - et: eye-tracking data
        - img: image data
        - sem: semantic label data
        - p_num: participant number
        - s_num: stimulus number
    - device: device to test on, e.g. cuda
    - loss_function: loss function to use, varies per model "mode"
    """
    model.eval()
    test_loss = 0
    snippet_length = 200
    true_labels = []
    pred_labels = []
    pred_et = []
    true_et = []
    with torch.no_grad():
        for data in test_dataloader:
            data = {key: value.to(device) for key, value in data.items()}
            p_num = data.pop("p_num")
            data_copy = data.copy()
            if mode == "et_reconstruction":
                data_copy["et"] = data["et"][:, :snippet_length, :]
                out = model(data_copy, p_num)
                pred_et.append(out)
                true_et.append(data["et"][:, snippet_length:, :])
                print("pred_et len, shape:", len(pred_et), out.shape)
                loss = loss_function(out,
                                     data["et"][:, snippet_length:, :])
            elif mode == "classification":
                pred_p = model(data_copy)
                loss = loss_function(pred_p, p_num)

                # Append current batch of true and predicted labels to respective lists
                true_labels.extend(p_num.tolist())
                pred_labels.extend(pred_p.tolist())
            test_loss += loss.item()

    test_loss = test_loss / len(test_dataloader)
    return test_loss, true_labels, pred_labels, pred_et, true_et


def objective(trial, train_dataset, test_dataset, device, mode, modalities, summarise=False):
    """
    Main objective function for the Optuna hyperparameter search.
    Inputs:
    - trial: Optuna trial object
    - train_dataset: training dataset
    - test_dataset: testing dataset
    - device: device to train on, e.g. cuda
    - mode: mode of the model, et_reconstruction or classification
    Outputs:
    - test_loss: test loss of the model, used by Optuna to find the best hyperparameters
    """
    if mode == "et_reconstruction":
        p_num_provided = True
        loss_function = nn.MSELoss()
    elif mode == "classification":
        p_num_provided = False
        loss_function = nn.CrossEntropyLoss()

    num_layers = trial.suggest_int("num_layers", 3, 6)
    heads = trial.suggest_int("heads", 8, 16, step=4)
    config = {
        "num_layers": num_layers,
        "heads": heads,
        "forward_expansion": trial.suggest_int("forward_expansion", 2, 4, step=2),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3),
        "batch_size": trial.suggest_int("batch_size", 64, 128, step=64),
        "n_bottlenecks": trial.suggest_int("n_bottlenecks", 4, 16, step=4),
        "fusion_layer": trial.suggest_int("fusion_layer", 2, num_layers-1),
        "num_epochs": 1,  # trial.suggest_int("num_epochs", 2, 5, step=1),
        "L2": trial.suggest_float("L2", 1e-5, 1e-3),
        "embed_dim": trial.suggest_int("embed_dim", 16*heads, 16*heads, step=heads),
        "p_num_embed_dim": 4,
        "et_patch_size": 15,
        "et_seq_len": 100,
        "et_dim": 4,
        "et_stride": 1,
        "img_patch_size": 25,
        "img_stride": 12,
        "sem_patch_size": 25,
        "sem_stride": 12,
        "modalities": modalities,
        "num_classes": 335,
        "device": device,
        "mode": mode,
        "p_num_provided": p_num_provided,
        "pad_id": 0,
    }

    model = MultimodalBottleneckTransformer(config).to(config["device"])
    # model = ETMLP(config).to(config["device"])

    if summarise:
        if task == "OSIE":
            summary(model, input_size=(config["batch_size"], 4, 300,
                                       config["batch_size"], 3, 600, 800,
                                       config["batch_size"], 12, 600, 800))
            print(config)
        elif task == "GB":
            summary(model, input_size=(config["batch_size"], 4, 300))
            print(config)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["L2"])

    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    train_model(model, train_dl, config["device"], loss_function, optimizer,
                config["num_epochs"], mode=model.mode)
    test_loss, true_labels, pred_labels, pred_et, true_et = test_model(model, test_dl, config["device"],
                                                                       loss_function, mode=model.mode)
    if config["mode"] == "classification":
        topk_acc, topk_classes = topk_accuracy(true_labels, pred_labels, k=5)
        print(f"Top-5 accuracy: {topk_acc:.4f}")
        print(f"Top-5 classes: {topk_classes}")
    elif config["mode"] == "et_reconstruction":
        torch.save(pred_et, "pred_et.pt")
        torch.save(pred_et, "true_et.pt")

    print(f"Test loss: {test_loss:.4f}")

    # Save the model if it's the best one so far
    try:
        if trial.should_prune():
            raise optuna.TrialPruned()
        elif trial.study.best_value is None or test_loss < trial.study.best_value:
            torch.save(model.state_dict(), "MBT-ETR.pth")
    except ValueError:
        pass
    return test_loss


def hyperparameter_optimization(train_dataset, test_dataset, device, mode, modalities, n_trials):
    """
    Hyperparameter optimization using Optuna.
    Inputs:
    - train_dataset: training dataset
    - test_dataset: testing dataset
    - device: device to train on, e.g. cuda
    - mode: mode of the model, et_reconstruction or classification
    - n_trials: number of trials to run the hyperparameter search
    Outputs:
    - study.best_params: best hyperparameters found by Optuna
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial,
                                           train_dataset,
                                           test_dataset,
                                           device,
                                           mode,
                                           modalities), n_trials=n_trials)
    return study.best_params


if __name__ == "__main__":
    """
    This implementation assumes that the participant numbers are used as filenames for the images
    and semantic data files. Adjust the file naming conventions and data loading as needed based
    on your specific dataset organization.
    """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'
    mode = 'et_reconstruction'
    task = 'GB'

    # Prepare the data and Dataset
    if task == "OSIE":
        base_path = os.getcwd()
        train_et_folder = os.path.join(base_path, "pipeline", "osieData", "osie_tensors")
        test_et_folder = os.path.join(base_path, "pipeline", "osieData", "osie_tensors")
        img_tensor, sem_tensor = load_img_sem_data(
            img_folder='pipeline/OSIE_imgs',
            sem_folder='pipeline/OSIE_tags')
        train_dataset = MultimodalDataset(train_et_folder, img_tensor, sem_tensor)
        test_dataset = MultimodalDataset(test_et_folder, img_tensor, sem_tensor)
        modalities = ['et', 'img', 'sem']

    elif task == "GB":
        folder = os.path.abspath(os.path.join('DS10'))
        files = sorted(os.listdir(folder))
        train_files = [os.path.join(folder, file) for file in files[:98]]
        test_files = [os.path.join(folder, file) for file in files[-28:]]
        train_dataset = GazeBaseDataset(train_files, device)
        test_dataset = GazeBaseDataset(test_files, device)
        modalities = ['et']

    # Hyperparameter optimization
    best_params = hyperparameter_optimization(train_dataset,
                                              test_dataset,
                                              device,
                                              mode,
                                              modalities,
                                              n_trials=5)
