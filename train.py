from util import weighted_sum_rate, prepare_channel_tx_ris, compute_complete_channel_continuous
from core import RISNet, RISNetPI, RTChannelsWMMSE
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import numpy as np
from params import params
import argparse
import datetime
from pathlib import Path
tb = True
try:
    from tensorboardX import SummaryWriter
except:
    tb = False
record = False and tb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsnr")
    parser.add_argument("--ris_shape")
    parser.add_argument("--weights")
    parser.add_argument("--lr")
    parser.add_argument("--record")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.tsnr is not None:
        params["tsnr"] = float(args.tsnr)
    if args.lr is not None:
        params["lr"] = float(args.lr)
    if args.weights is not None:
        weights = args.weights.split(',')
        params["alphas"] = np.array([float(w) for w in weights])
    if args.ris_shape is not None:
        ris_shape = args.ris_shape.split(',')
        params["ris_shape"] = tuple([int(s) for s in ris_shape])
    if args.record is not None:
        record = args.record == "True"
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tb = tb and record

    if record:
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        Path(params["results_path"] + dt_string).mkdir(parents=True, exist_ok=True)
        params["results_path"] = params["results_path"] + dt_string + "/"

    params["discrete_phases"] = params["discrete_phases"].to(device)
    params["iter_wmmse"] = 100
    params["wmmse_saving_frequency"] = 10

    if params["permutation_invariant"]:
        model = RISNetPI(params).to(device)
    else:
        model = RISNet(params).to(device)

    channel_tx_ris, channel_tx_ris_pinv = prepare_channel_tx_ris(params, device)
    data_set = RTChannelsWMMSE(params, channel_tx_ris_pinv, device)
    result_name = "ris_" + str(params['tsnr']) + "_" + str(params['ris_shape']) + '_' + str(params['alphas']) + "_"
    train_loader = DataLoader(dataset=data_set, batch_size=params['batch_size'], shuffle=True)
    losses = list()
    if tb:
        writer = SummaryWriter(logdir=params["results_path"])
        tb_counter = 1
    model.train()

    optimizer_wmmse = optim.Adam(model.parameters(), params['lr'])
    # Training with WMMSE precoder
    for wmmse_iter in range(params['iter_wmmse'] + 1):

        data_set.wmmse_precode(model, channel_tx_ris, device)
        for epoch in range(params['epoch_per_iter_wmmse']):
            for batch in train_loader:
                sample_indices, channels_ris_rx_array, channels_ris_rx, channels_direct, location, precoding = batch

                optimizer_wmmse.zero_grad()

                nn_raw_output = model(channels_ris_rx_array)
                complete_channel = compute_complete_channel_continuous(channel_tx_ris, nn_raw_output,
                                                                       channels_ris_rx, channels_direct, params)
                wsr = weighted_sum_rate(complete_channel, precoding, params)
                loss = -torch.mean(wsr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                for name, param in model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print("nan gradient found")
                optimizer_wmmse.step()

                print('WMMSE round {round}, Epoch {epoch}, '
                      'data rate = {loss}'.format(round=wmmse_iter,
                                               loss=-loss,
                                             epoch=epoch))

                if tb and record:
                    writer.add_scalar("Training/data_rate", -loss.item(), tb_counter)
                    tb_counter += 1

        if record and wmmse_iter % params['wmmse_saving_frequency'] == 0:
            torch.save(model.state_dict(), params['results_path'] + result_name +
                       'WMMSE_{iter}'.format(iter=wmmse_iter))
            if tb:
                writer.flush()
