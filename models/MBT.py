"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:04:15
 * @modify date 2023-05-11 12:04:15
 * @desc [description]
"""
import os
import torch
import torch.nn as nn
from mbt_encoder import Encoder
from embedders import ETPatchEmbed, ImagePatchEmbed, SemanticEmbedding


class MultimodalBottleneckTransformer(nn.Module):
    def __init__(self, config):
        super(MultimodalBottleneckTransformer, self).__init__()

        self.config = config
        self.modalities = config["modalities"]
        self.device = config["device"]
        self.p_num_provided = config["p_num_provided"]
        self.embedders = {}

        if self.p_num_provided:
            in_channels = config["et_dim"] + config["p_num_embed_dim"]
        else:
            in_channels = config["et_dim"]

        for modality in self.modalities:
            if modality == "et":
                self.embedders["et"] = ETPatchEmbed(
                    in_channels=in_channels,
                    embed_dim=config["embed_dim"],
                    kernel_size=config["et_patch_size"],
                    stride=config["et_stride"]
                    )
            elif modality == "img":
                self.embedders["img"] = ImagePatchEmbed(
                    in_channels=3,
                    embed_dim=config["embed_dim"],
                    patch_size=config["img_patch_size"],
                    stride=config["img_stride"],
                    )
            elif modality == "sem":
                self.embedders["sem"] = SemanticEmbedding(
                    in_channels=12,
                    embed_dim=config["embed_dim"],
                    patch_size=config["sem_patch_size"],
                    stride=config["sem_stride"],
                    )

        self.d_model = config["embed_dim"]
        self.mode = config["mode"]
        self.n_bottlenecks = config["n_bottlenecks"]
        self.pad_id = config["pad_id"]
        self.p_num_embed = nn.Embedding(config["num_classes"],
                                        config["p_num_embed_dim"]).to(self.device)
        self.start_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, 1, self.d_model))
            for modality in self.modalities})
        self.class_tokens = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, 1, self.d_model))
            for modality in self.modalities})

        encoder = Encoder(
            embed_size=config["embed_dim"],
            num_layers=config["num_layers"],
            heads=config["heads"],
            fusion_layer=config["fusion_layer"],
            forward_expansion=config["forward_expansion"],
            dropout=config["dropout"],
            device=self.device,
            modalities=self.modalities,
            )
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=config["heads"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=config["num_layers"],
            dim_feedforward=4 * self.d_model,
            custom_encoder=encoder,
            )

        self.classification_head = nn.Linear(self.d_model, config["num_classes"])
        self.et_reconstruction_head = nn.Linear(
            config["embed_dim"] * (config["embed_dim"] + 1),
            config["et_seq_len"] * config["et_dim"])

    def adjust_time_dimension(self, embedding):
        """Adjust the time dimension of the embedding to match the target time steps."""
        current_steps, feature_dim = embedding.shape[-2:]
        target_time_steps = self.d_model
        if current_steps == target_time_steps:
            return embedding
        else:
            embedding = embedding.unsqueeze(1)
            embedding = nn.functional.interpolate(
                embedding,
                size=(target_time_steps, feature_dim),
                mode="bilinear",
                align_corners=False,
            )
            return embedding.squeeze(1)

    def create_src_tgt_sequences(self, x: dict[str, any], start_token=None, tgt_mod="et"):
        """Create the source and target sequences for the transformer model."""
        src = {}
        tgt = {}
        for idx, modality in enumerate(self.modalities):
            src_batch = x[modality]
            actual_batch_size = src_batch.size(0)

            # Concatenate the class token
            class_token = self.class_tokens[modality].repeat(actual_batch_size, 1, 1)
            src_batch = torch.cat((class_token, src_batch), dim=1)

            if start_token is None:
                start_token = self.start_tokens[modality].repeat(actual_batch_size, 1, 1)

            # Shift the source sequence to create the target sequence
            tgt_batch = src_batch[:, :-1]  # Remove the last element
            tgt_batch = torch.cat((start_token, tgt_batch), dim=1)  # Add the start token

            src[modality] = src_batch
            tgt[modality] = tgt_batch
        tgt = tgt[tgt_mod]
        return src, tgt

    def make_src_mask(self, src: dict[str, any]):
        src_mask = {}
        for modality, data in src.items():
            batch_size, seq_len, _ = data.shape
            src_mask[modality] = (data == self.pad_id).float()
            padding = torch.zeros(batch_size, seq_len, 1, dtype=src_mask[modality].dtype,
                                  device=src_mask[modality].device)
            src_mask[modality] = torch.cat((src_mask[modality], padding), dim=2)
        return src_mask

    def make_tgt_mask(self, tgt: torch.Tensor):
        tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(0))).type_as(tgt).to(tgt.device)
        return tgt_mask

    def init_bottleneck(self, src: dict[str, any], modality):
        n, c, t = src[modality].shape
        bottleneck = nn.init.normal_(torch.empty(n, self.n_bottlenecks, t),
                                     mean=0, std=0.02).to(self.device)
        return bottleneck

    def forward(self, data_dict, p_num=None):
        x = {}
        if self.p_num_provided:
            p_num_embed = self.p_num_embed(p_num)
            p_num_embed = p_num_embed.unsqueeze(1).expand(-1, data_dict["et"].size(1), -1)
            data_dict["et"] = torch.cat((data_dict["et"], p_num_embed), dim=-1)

        for modality in self.modalities:
            self.embedders[modality] = self.embedders[modality].to(self.device)
            if modality in data_dict:
                x[modality] = self.adjust_time_dimension(
                    self.embedders[modality](data_dict[modality]))
            else:
                print(f"Data for modality {modality} not has unknown use.")

        src, tgt = self.create_src_tgt_sequences(x)
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        bottleneck = self.init_bottleneck(src, "et")

        out_src = self.transformer.encoder(src, bottleneck, src_key_padding_mask=src_mask["et"])
        out_src_et = out_src[:, :, :self.config["embed_dim"]]
        out = self.transformer.decoder(tgt, out_src_et, tgt_key_padding_mask=tgt_mask)

        if self.mode == "classification":
            out = self.classification_head(out[:, 0, :])
            return out
        elif self.mode == "et_reconstruction":
            actual_batch_size = out.size(0)
            input_flat = out.view(actual_batch_size, -1)
            out_flat = self.et_reconstruction_head(input_flat)
            et_reconstructed = out_flat.view(actual_batch_size, 300, 4)
            return et_reconstructed
        # TODO: Add the other modes like image prediction or semantic prediction.
        else:
            raise ValueError("Mode must be either 'classification' or 'et_reconstruction'.")


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = "cpu"
    mode = "et_reconstruction"
    if mode == "et_reconstruction":
        p_num_provided = True
    elif mode == "classification":
        p_num_provided = False
    config = {
        "num_layers": 2,
        "heads": 4,  # trial.suggest_int("heads", 4, 24, step=4),
        "forward_expansion": 2,  # trial.suggest_int("forward_expansion", 2, , step=2),
        "dropout": 0.1,  # trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "lr": 1e-3,  # trial.suggest_float("lr", 1e-2, 1e-1),
        "batch_size": 64,  # trial.suggest_int("batch_size", 64, 256, step=64),
        "n_bottlenecks": 8,  # trial.suggest_int("n_bottlenecks", 4, 16, step=4),
        "fusion_layer": 2,  # trial.suggest_int("fusion_layer", 2, num_layers),
        "num_epochs": 50,  # trial.suggest_int("num_epochs", 50, 200, step=50),
        "L2": 1e-4,  # trial.suggest_float("L2", 1e-5, 1e-3),
        "p_num_embed_dim": 4,
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
        "p_num_provided": p_num_provided,
        "pad_id": 0,
    }

    model = MultimodalBottleneckTransformer(config).to(config["device"])

    # Initialize input data
    # NOTE: Deprecated: data is now a dictionary with keys "et", "img", "sem"
    et_data = torch.randn(config["batch_size"], 4, 300).to(config["device"])
    img_data = torch.randn(config["batch_size"], 3, 600, 800).to(config["device"])
    sem_data = torch.randn(config["batch_size"], 12, 600, 800).to(config["device"])
    p_num = torch.randint(0, 10, (config["batch_size"],)).to(config["device"])

    # Forward pass
    output = model(et_data, img_data, sem_data, p_num)
    print(output.shape)
