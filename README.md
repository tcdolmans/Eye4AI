# Multimodal Bottleneck Transformer Based Eye Movement Prediction
This repository contains the code that I've written for my master's thesis in the Human Neuroscience (MSc) program from the University of Turku, Finland. Currently, `models` and `pipeline` are populated, but still quite disorganised. 
### Models
The `GPMBT.py` file is the main MBT, which makes use of a custom encoder and embeddings for the Eye Tracking, Vision, and Semantic Labels embeddings. These are defined in `mbt_encoder.py` and `embedders.py` respectively. The `trainer.py` file contains the main training loop and is currently under development. Optuna is used for HPO. 
### Pipeline
Files in `pipeline` were mostly used for the dataprep associated with GazeBase [ref]. However, new files are being written that prep OSIE data [ref]. These data will then we used to train the MBT with. 
