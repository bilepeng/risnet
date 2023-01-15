import numpy as np
import torch
import copy


# RT channel model
params2users = {'lr': 8e-4,
                'epoch': 50,
                'num_users': 2,
                'iter_wmmse': 2000,
                'epoch_per_iter_wmmse': 1,
                'entropy_history_length': 5,
                'alphas': [0.5, 0.5],
                'saving_frequency': 10,
                'wmmse_saving_frequency': 10,
                'batch_size': 512,
                'permutation_invariant': True,
                'results_path': 'results/',
                'tsnr': 1e11,
                "frequencies": np.linspace(1e6, 1e6 + 100, 10),
                'quantile2keep': 0.6,
                "phase_shift": "continuous",
                "discrete_phases": torch.tensor([0, np.pi])[None, None, :],
                'mean_ris': 6.2378e-5,
                'std_ris': 5.0614e-5,
                'mean_direct': 1.4374e-4,
                'std_direct': 3.714e-4,
                'ris_shape': (32, 32),
                'channel_tx_ris_original_shape': (32, 32, 9),
                # width, height of RIS and Tx antennas. Do not change this!
                'channel_ris_rx_original_shape': (16000, 32, 32),  # samples, width, height of RIS and users
                'n_tx_antennas': 9,
                'los': True,
                'precoding': 'wmmse',
                # Debug
                'channel_direct_path': 'data/channels_direct_training.pt',
                'channel_tx_ris_path': 'data/channel_tx_ris.pt',
                'channel_ris_rx_path': 'data/channels_ris_rx_training.pt',
                # 'channel_direct_path': 'data/channels_direct_training_s.pt',
                # 'channel_tx_ris_path': 'data/channel_tx_ris_s.pt',
                # 'channel_ris_rx_path': 'data/channels_ris_rx_training_s.pt',
                'location_path': 'data/locations_training.pt',
                'group_definition_path': 'data/group_definition_2users_training_s.npy',
                'angle_diff_threshold': 0.5,
                'user_distance_threshold': 20,
                'ris_loc': torch.tensor([278.42, 576.97, 2]),
                'trained_mmse_model': None,
                # 'trained_mmse_model': 'results/RISNetPIDiscrete_MMSE_16-05-2022_13-46-01/ris_100000000000.0_(32, 32)_[0.5, 0.5]_4000',
                'channel_estimate_error': 0,
                'discount_long': 0.95,
                'discount_short': 0.4,
                'delta_support': 0.0001,
                }


# If statistical channel
if True:
    params2users["channel_direct_path"] = 'data/channels_direct_training_s.pt'
    params2users["channel_tx_ris_path"] = 'data/channel_tx_ris_s.pt'
    params2users['channel_ris_rx_path'] = 'data/channels_ris_rx_training_s.pt'
    params2users['group_definition_path'] = 'data/group_definition_2users_training_s.npy'
    params2users['mean_ris'] = 7.979e-4
    params2users['std_ris'] = 6.028e-4
    params2users['mean_direct'] = 1.0074e-4
    params2users['std_direct'] = 6.361e-5
    params2users['channel_ris_rx_original_shape'] = (10240, 32, 32)  # samples, width, height of RIS and users

params4users = copy.deepcopy(params2users)
params4users["num_users"] = 4
params4users['group_definition_path'] = 'data/group_definition_4users_training_s.npy'
params4users["alphas"] = [0.25, 0.25, 0.25, 0.25]

params = params4users