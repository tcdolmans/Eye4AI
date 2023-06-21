# Multimodal Bottleneck Transformer Based Eye Movement Prediction
This repository contains the code that I've written for my master's thesis in the Human Neuroscience (MSc) program from the University of Turku, Finland. Currently, `models` and `pipeline` are populated with all relevant files. 
### Models
The `MBT.py` file is the main MBT, which makes use of a custom encoder and embeddings for the Eye Tracking, Vision, and Semantic Labels embeddings. These are defined in `mbt_encoder.py` and `embedders.py` respectively. 
The `trainer.py` file contains the main training and HPO loop. Optuna is used for HPO. 
`MLP.py` contains the MLP model used as a baseline.
Dataloaders for the various datasets can be found in `dataloaders.py`.

-- Please note that `etd_transformer` is deprecated. --
### Pipeline
Files in `pipeline` were mostly used for the dataprep associated with GazeBase [https://doi.org/10.1038/s41597-021-00959-y]. However, new files are being written that prep OSIE data [https://doi.org/10.1167/14.1.28]. These data will then we used to train the MBT with.
`Osie.m` is used to extract and save a tensor for all semantic labels.
`convert_osie.py` converts UTU's collected data to .pt files.
`convert_sets.py` converts GazeBase data to .pt files.
`dataset_constructor.py` converts the GazeBase.pt files to downsampled/filtered .pt files.
`utils.py` contains various helper functions.

