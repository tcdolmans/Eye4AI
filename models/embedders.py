import torch.nn as nn


class ETPatchEmbed(nn.Module):
    """Takes a batch of ET data and returns their embeddings."""
    def __init__(self,
                 in_channels=4,
                 embed_dim=768,
                 kernel_size=15,
                 stride=1,
                 padding=7):
        super().__init__()
        self.norm = nn.AdaptiveAvgPool1d(embed_dim)
        self.projection = nn.Conv1d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = self.norm(x)
        return x


class ImagePatchEmbed(nn.Module):
    """Takes batch of images and returns their embeddings."""
    def __init__(self,
                 in_channels=3,
                 embed_dim=768,
                 patch_size=25,
                 stride=1,
                 padding=12):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.AdaptiveAvgPool1d(embed_dim)
        self.projection = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class SemanticEmbedding(nn.Module):
    """Takes batch of semantic labels and returns their embeddings."""
    def __init__(self,
                 in_channels=12,
                 embed_dim=768,
                 patch_size=25,
                 stride=1,
                 padding=12):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.AdaptiveAvgPool1d(embed_dim)
        self.projection = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embed_dim,
                                    kernel_size=patch_size,
                                    stride=stride,
                                    padding=padding)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class EyeTrackingEmbedder(nn.Module):
    """
    Alternative for ETPatchEmbed. Uses a CNN-LSTM architecture.

    Input:
    Shape: (batch_size, input_size, sequence_length)
    The input data should be a 3D tensor, where batch_size is the number of samples in the batch,
    input_size is the number of features (e.g., fixation duration, saccade amplitude, etc.),
    and sequence_length is the length of the time series segments.

    Output:
    Shape: x: (batch_size, output_size),
           hidden_states: (lstm_num_layers, batch_size, lstm_hidden_size)
    The output x is a 2D tensor batches of learned features.
    the output hidden_states is a 3D tensor of layers, by batch, by hidden LSTM size.
    """
    def __init__(self,
                 input_size,
                 num_filters,
                 kernel_size,
                 lstm_hidden_size,
                 lstm_num_layers, output_size):
        super(EyeTrackingEmbedder, self).__init__()

        self.conv1 = nn.Conv1d(input_size,
                               num_filters,
                               kernel_size)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size)
        self.lstm = nn.LSTM(num_filters,
                            lstm_hidden_size,
                            lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size,
                            output_size)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x, (hidden_states, _) = self.lstm(x)

        x = self.fc(hidden_states[-1])
        x = self.relu2(x)

        return x, hidden_states
