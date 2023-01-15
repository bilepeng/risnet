import numpy as np
from util import test_model, prepare_channel_tx_ris, \
    compute_complete_channel_discrete, compute_complete_channel_continuous, compute_wmmse_v, weighted_sum_rate
from train_oma import mmse_precoding
import torch
from core import RISNetPIDiscrete, RISNetDiscrete, RISNet, RISNetPI, RTChannelsWMMSE
from torch.utils.data import DataLoader
from params import params
import matplotlib.pyplot as plt
tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False


def ecdf(data):
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return x, y


if __name__ == '__main__':
    params["channel_direct_path"] = 'data/channels_direct_testing_s.pt'
    params["channel_tx_ris_path"] = 'data/channel_tx_ris_s.pt'
    params["channel_ris_rx_path"] = 'data/channels_ris_rx_testing_s.pt'
    params["group_definition_path"] = 'data/group_definition_4users_testing_s.npy'
    params['channel_ris_rx_original_shape'] = (1024, 32, 32)  # samples, width, height of RIS and users
    params["tsnr"] = 1e11
    params["permutation_invariant"] = False

    if params["permutation_invariant"]:
        if params["phase_shift"] == "discrete":
            model = RISNetPIDiscrete()
        elif params["phase_shift"] == "continuous":
            model = RISNetPI(params)
    else:
        if params["phase_shift"] == "discrete":
            model = RISNetDiscrete()
        elif params["phase_shift"] == "continuous":
            model = RISNet(params)

    device = 'cpu'

    model.load_state_dict(torch.load('results/ris_100000000000.0_(32, 32)_[0.25, 0.25, 0.25, 0.25]_WMMSE_100'))

    model.eval()
    channel_tx_ris, channel_tx_ris_pinv = prepare_channel_tx_ris(params, device)
    data_set = RTChannelsWMMSE(params, channel_tx_ris_pinv, device, test=True)
    test_loader = DataLoader(dataset=data_set, batch_size=1024, shuffle=True)
    # data_set.cut_data(1)
    average = list()
    data_set.wmmse_precode(model, channel_tx_ris, device, 500)
    for batch in test_loader:
        sample_indices, channels_ris_rx_array, channels_ris_rx, channels_direct, location, precoding = batch

        entropy_current_epoch = list()
        fcn_raw_output = model(channels_ris_rx_array)
        complete_channel = compute_complete_channel_continuous(channel_tx_ris, fcn_raw_output, channels_ris_rx,
                                                             channels_direct, params)

        wsr = weighted_sum_rate(complete_channel, precoding, params)

        print('average = {ave}'.format(ave=wsr.mean()))
