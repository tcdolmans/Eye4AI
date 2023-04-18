import os
import torch
import torch.nn as nn
from mbt_encoder import Encoder
from embedders import ETPatchEmbed, ImagePatchEmbed, SemanticEmbedding


class MultimodalBottleneckTransformer(nn.Module):
    def __init__(self, config):
        super(MultimodalBottleneckTransformer, self).__init__()

        self.modalities = config["modalities"]
        self.device = config["device"]

        self.et_embed = ETPatchEmbed(
            in_channels=config["et_dim"],
            embed_dim=config["et_embed_dim"],
            kernel_size=config["et_patch_size"],
            stride=config["et_stride"],
        )

        self.img_embed = ImagePatchEmbed(
            in_channels=3,
            embed_dim=config["img_embed_dim"],
            patch_size=config["img_patch_size"],
            stride=config["img_stride"],
        )

        self.sem_embed = SemanticEmbedding(
            in_channels=12,
            embed_dim=config["sem_embed_dim"],
            patch_size=config["sem_patch_size"],
            stride=config["sem_stride"],
        )

        self.d_model = config["et_embed_dim"]
        self.mode = config["mode"]
        self.n_bottlenecks = config["n_bottlenecks"]
        self.pad_id = config["pad_id"]

        self.start_tokens = nn.Parameter(torch.randn(1, len(self.modalities),
                                                     self.d_model*len(self.modalities)))
        self.class_tokens = nn.Parameter(torch.randn(1, len(self.modalities),
                                                     self.d_model*len(self.modalities)))

        encoder = Encoder(
            embed_size=config["et_embed_dim"],
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
        self.prediction_head = nn.Linear(self.d_model, config["et_embed_dim"])
        self.et_reconstruction_head = nn.Linear(
            config["et_embed_dim"] * (config["et_embed_dim"] + 1),
            config["et_seq_len"] * config["et_dim"])

    def adjust_time_dimension(self, embedding, target_time_steps=192):
        """Adjust the time dimension of the embedding to match the target time steps."""
        batch_size, current_steps, feature_dim = embedding.shape
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

    def create_src_tgt_sequences(self, x: dict[str, any], start_token=None):
        """Create the source and target sequences for the transformer model."""
        src = {}
        tgt = {}
        for idx, modality in enumerate(self.modalities):
            src_batch = x[modality]

            # Concatenate the class token
            start = idx * config["et_embed_dim"]
            end = (idx + 1) * config["et_embed_dim"]
            class_token = self.class_tokens[:, idx, start:end].unsqueeze(1).repeat(
                src_batch.size(0), 1, 1)
            src_batch = torch.cat((class_token, src_batch), dim=1)

            if start_token is None:
                start = idx * config["et_embed_dim"]
                end = (idx + 1) * config["et_embed_dim"]
                start_token = self.start_tokens[:, idx, start:end].unsqueeze(1).repeat(
                    src_batch.size(0), 1, 1)

            # Shift the source sequence to create the target sequence
            tgt_batch = src_batch[:, :-1]  # Remove the last element
            tgt_batch = torch.cat((start_token, tgt_batch), dim=1)  # Add the start token

            src[modality] = src_batch
            tgt[modality] = tgt_batch
        tgt = tgt["et"]
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

    def forward(self, et_data, img_data, sem_data):
        # Initialise the input dictionary and add the embeddings
        x = {}
        x["et"] = self.adjust_time_dimension(self.et_embed(et_data))
        x["img"] = self.adjust_time_dimension(self.img_embed(img_data))
        x["sem"] = self.adjust_time_dimension(self.sem_embed(sem_data))

        src, tgt = self.create_src_tgt_sequences(x)
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        bottleneck = self.init_bottleneck(src, "et")

        out_src = self.transformer.encoder(src, bottleneck, src_key_padding_mask=src_mask["et"])
        out_src_et = out_src[:, :, :config["et_embed_dim"]]
        out = self.transformer.decoder(tgt, out_src_et, tgt_key_padding_mask=tgt_mask)

        if self.mode == "classification":
            out = self.classification_head(out[:, 0, :])
            return out
        elif self.mode == "prediction":
            out = self.prediction_head(out)  # shape (batch_size, et_embed_dim),
            # Extract only the ET embeddings from the output
            out = out[:, :, :self.et_embed.embed_dim]
            out = torch.mean(out, dim=1)  # NOTE:May not have to mean it here
            return out
        elif self.mode == "et_reconstruction":
            input_flat = out.view(config["batch_size"], -1)
            out_flat = self.et_reconstruction_head(input_flat)
            et_reconstructed = out_flat.view(config["batch_size"], 300, 4)
            return et_reconstructed
        # TODO: Add the other modes like image prediction or semantic prediction.
        else:
            raise ValueError("Invalid mode. Must be either 'classification' or 'prediction'.")


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        "num_layers": 3,
        "heads": 2,
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
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_bottlenecks": 4,
        "pad_id": 0,
        "mode": "et_reconstruction",
        "fusion_layer": 2,
        "forward_expansion": 4,
        "dropout": 0.1,
    }

    model = MultimodalBottleneckTransformer(config).to(config["device"])

    # Initialize input data
    et_data = torch.randn(config["batch_size"], 4, 300).to(config["device"])
    img_data = torch.randn(config["batch_size"], 3, 600, 800).to(config["device"])
    sem_data = torch.randn(config["batch_size"], 12, 600, 800).to(config["device"])

    # Forward pass
    output = model(et_data, img_data, sem_data)
    print(output.shape)
