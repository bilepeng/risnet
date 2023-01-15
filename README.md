# RISnet

This repository is the source code and data for the paper

B. Peng, F. Siegismund-Poschmann, E. Jorswieck, "RISnet: a dedicated scalable neural network architecture for optimization of reconfigurable intelligent surfaces", International ITG 26th Workshop on Smart Antennas and 13th Conference on Systems, Communications, and Coding, Braunschweig, 2023.

To train the neural network, run `train.py` with the following arguments:

- --tsnr: the transmit SNR with default value `1e11`.
- --lr: the learning rate with default value `8e-4`.
- --ris_shape: the RIS shape with default value `32, 32`.
- --weights: the user weights in the weighted sum-rate with default value `0.25, 0.25, 0.25, 0.25`.
- --record: `True` if you want to save the tensorboard log and trained models in a folder named after date and time of the beginning of training, `False` otherwise.
- --device: `cpu` or `cuda`.

To test the saved neural network, run `test.py` with the path to the saved model in line 46.