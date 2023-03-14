import torch
import torch.nn as nn
from models import EyeNet, PicNet
from models import mbt
from torch import trainer

class MultimodalTransformer(nn.Module):
  def __init__(self, num_classes, num_timesteps, num_features, num_pixels, paradigm):
    super().__init__()
    # Select paradigm, 1 = User Classification, 2 = Gaze Predict,  3 = Image Predict
    self.paradigm = paradigm()
    self.et_transformer = EyeNet(
      d_model=num_features,
      nhead=8,
      num_encoder_layers=6,
      num_decoder_layers=6,
      dim_feedforward=2048,
      dropout=0.1
    )

    self.picture_transformer = nn.PicNet(
      d_model=num_pixels,
      nhead=8,
      num_encoder_layers=6,
      num_decoder_layers=6,
      dim_feedforward=2048,
      dropout=0.1
    )

    self.fc1 = nn.Linear(num_features + num_pixels, 256)
    self.fc2 = nn.Linear(256, num_classes)

  def forward(self, eye_tracking_data, pictures):
    # Transform the eye tracking data
    eye_tracking_data = eye_tracking_data.permute(1, 0, 2)  # (batch_size, num_timesteps, num_features)
    eye_tracking_features = self.eye_tracking_transformer(eye_tracking_data, eye_tracking_data)

    # Transform the pictures
    pictures = pictures.permute(0, 3, 1, 2)  # (batch_size, num_channels, height, width)
    picture_features = self.picture_transformer(pictures, pictures)

    # Perform feature fusion at indicated layer L
    features = mbt.fuse((eye_tracking_features, picture_features), dim=2, fusion_layer=f_l, fusion_tokens=f_t )

    # Pass the concatenated features through a fully connected layer
    x = self.fc1(features)
    x = nn.functional.relu(x)

    # Pass the output through another fully connected layer to get the final predictions
    x = self.fc2(x)
    return x

def eval(self, x, paradigm):
    # Select paradigm, 1 = User Classification, 2 = Gaze Predict,  3 = Image Predict
    if paradigm == 1:
        predict_indiv()
    elif paradigm == 2:
        predict_gaze()
    elif paradigm == 3:
        predict_image()
    else:
        print("Incorrect paradigm selection")