import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from util import prepare_channel_direct_features, cp2array_risnet, compute_complete_channel_continuous, \
    compute_wmmse_v_v2, mmse_precoding
from joblib import cpu_count
from torch.utils.data import Dataset


class RISNet(nn.Module):
    def __init__(self, params):
        super(RISNet, self).__init__()
        self.feature_dim = 4 * params["num_users"]
        self.output_dim = 1
        self.local_info_dim = 16
        self.global_info_dim = 16
        self.skip_connection = False
        self.conv1 = nn.Conv1d(self.feature_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv2 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv3 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv4 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv5 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv6 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv7 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.local_info_dim + self.global_info_dim, 1)
        self.conv8 = nn.Conv1d(self.feature_dim + self.local_info_dim + self.global_info_dim,
                               self.output_dim, 1)

    def forward(self, channel):
        def postprocess_layer(channel_, conv_output, n_entries):
            local_info = conv_output[:, :self.local_info_dim, :]
            global_info = torch.mean(conv_output[:, -self.global_info_dim:, :], dim=2, keepdim=True)
            global_info = global_info.repeat([1, 1, n_entries])
            layer_output = torch.cat((channel_, local_info, global_info), 1)
            return layer_output

        _, _, n_antennas = channel.shape

        r = F.relu(self.conv1(channel))
        r = postprocess_layer(channel, r, n_antennas)

        if self.skip_connection:
            r1 = r

        r = F.relu(self.conv2(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv3(r))
        r = postprocess_layer(channel, r, n_antennas)

        if self.skip_connection:
            r3 = r

        r = F.relu(self.conv4(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv5(r))
        r = postprocess_layer(channel, r, n_antennas)

        if self.skip_connection:
            r = (r + r1) / 2

        r = F.relu(self.conv6(r))
        r = postprocess_layer(channel, r, n_antennas)

        r = F.relu(self.conv7(r))
        r = postprocess_layer(channel, r, n_antennas)

        if self.skip_connection:
            r = (r + r3) / 2

        r = self.conv8(r) * np.pi

        return r


class RISNetPI(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.feature_dim = 4
        self.output_dim = 1
        self.local_info_dim = 8
        self.global_info_dim = 8
        self.num_users = params["num_users"]
        self.skip_connection = False
        self.conv_ego1 = nn.Conv1d(self.feature_dim,
                                   self.local_info_dim + self.global_info_dim, 1)
        self.conv_ego2 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                   self.local_info_dim + self.global_info_dim, 1)
        self.conv_ego3 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                   self.local_info_dim + self.global_info_dim, 1)
        self.conv_ego4 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                   self.local_info_dim + self.global_info_dim, 1)
        self.conv_ego5 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                   self.local_info_dim + self.global_info_dim, 1)
        self.conv_ego6 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                   self.local_info_dim + self.global_info_dim, 1)
        self.conv_ego7 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                   self.local_info_dim + self.global_info_dim, 1)

        self.conv_opposite1 = nn.Conv1d(self.feature_dim,
                                        self.local_info_dim + self.global_info_dim, 1)
        self.conv_opposite2 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                        self.local_info_dim + self.global_info_dim, 1)
        self.conv_opposite3 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                        self.local_info_dim + self.global_info_dim, 1)
        self.conv_opposite4 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                        self.local_info_dim + self.global_info_dim, 1)
        self.conv_opposite5 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                        self.local_info_dim + self.global_info_dim, 1)
        self.conv_opposite6 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                        self.local_info_dim + self.global_info_dim, 1)
        self.conv_opposite7 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                        self.local_info_dim + self.global_info_dim, 1)

        self.conv_8 = nn.Conv1d(self.feature_dim + 2 * (self.local_info_dim + self.global_info_dim),
                                self.output_dim, 1)

    def forward(self, channel):
        def process_layer(features, layer_ego, layer_opposite):
            outputs_ego = [F.relu(layer_ego(f)) for f in features]

            outputs_opposite = list()
            for user_idx in range(self.num_users):
                output_opposite = torch.stack([F.relu(layer_opposite(f))
                                               for idx, f in enumerate(features) if idx != user_idx], dim=0).mean(dim=0)
                outputs_opposite.append(output_opposite)

            return outputs_ego, outputs_opposite

        def postprocess_layer(channels, outputs_ego, outputs_opposite, n_entries):
            global_infos_ego = [torch.mean(o[:, -self.global_info_dim:, :], dim=2,
                                           keepdim=True).repeat([1, 1, n_entries])
                                for o in outputs_ego]
            global_infos_opposite = [torch.mean(o[:, -self.global_info_dim:, :], dim=2,
                                                keepdim=True).repeat([1, 1, n_entries])
                                     for o in outputs_opposite]

            output_features = [torch.cat([channel,
                                          local_ego[:, :self.local_info_dim, :],
                                          global_ego,
                                          local_opposite[:, :self.local_info_dim, :],
                                          global_opposite], dim=1) for
                               channel, local_ego, global_ego, local_opposite, global_opposite in
                               zip(channels, outputs_ego, global_infos_ego, outputs_opposite, global_infos_opposite)]
            return output_features

        _, n_features, n_antennas = channel.shape
        n_f_per_user = int(n_features / self.num_users)

        channels = [channel[:, (n_f_per_user * user_idx): (n_f_per_user * user_idx + n_f_per_user), :]
                    for user_idx in range(self.num_users)]

        o_ego, o_opposite = process_layer(channels, self.conv_ego1, self.conv_opposite1)
        o = postprocess_layer(channels, o_ego, o_opposite, n_antennas)

        o_ego, o_opposite = process_layer(o, self.conv_ego2, self.conv_opposite2)
        o = postprocess_layer(channels, o_ego, o_opposite, n_antennas)

        o_ego, o_opposite = process_layer(o, self.conv_ego3, self.conv_opposite3)
        o = postprocess_layer(channels, o_ego, o_opposite, n_antennas)

        o_ego, o_opposite = process_layer(o, self.conv_ego4, self.conv_opposite4)
        o = postprocess_layer(channels, o_ego, o_opposite, n_antennas)

        o_ego, o_opposite = process_layer(o, self.conv_ego5, self.conv_opposite5)
        o = postprocess_layer(channels, o_ego, o_opposite, n_antennas)

        o_ego, o_opposite = process_layer(o, self.conv_ego6, self.conv_opposite6)
        o = postprocess_layer(channels, o_ego, o_opposite, n_antennas)

        o_ego, o_opposite = process_layer(o, self.conv_ego7, self.conv_opposite7)
        o = postprocess_layer(channels, o_ego, o_opposite, n_antennas)

        o = self.conv_8(torch.stack(o, dim=0).mean(dim=0)) * np.pi
        return o


class RTChannels(Dataset):
    def __init__(self, params, channel_tx_ris_pinv, device='cpu', test=False):
        self.params = params
        self.device = device
        self.locations = torch.load(params["location_path"]).cfloat()
        self.group_definition = np.load(params["group_definition_path"])
        # self.group_definition = np.random.choice(10240, (int(10240 / params["num_users"]), params["num_users"]), False)
        self.channels_ris_rx = torch.load(params['channel_ris_rx_path'], map_location=torch.device(device)).cfloat()

        self.channels_ris_rx = torch.reshape(self.channels_ris_rx, params['channel_ris_rx_original_shape'])[:,
                               : params['ris_shape'][0],
                               : params['ris_shape'][1]]
        self.channels_ris_rx = torch.reshape(self.channels_ris_rx, (
            -1, 1, params['ris_shape'][0] * params['ris_shape'][1]))

        self.channel_array = cp2array_risnet(self.channels_ris_rx,
                                             1 / params['std_ris'],
                                             params['mean_ris'],
                                             device=device)

        self.channels_direct = torch.load(params['channel_direct_path'], map_location=torch.device(device)).cfloat()
        channels_direct_array = prepare_channel_direct_features(self.channels_direct, channel_tx_ris_pinv,
                                                                self.params, self.device)
        self.channel_array = torch.cat([self.channel_array, channels_direct_array], 1)

        self.test = test

    def __getitem__(self, item):
        user_indices = self.group_definition[item, :]
        channel_features = torch.cat([self.channel_array[i, :, :] for i in user_indices])
        channels_ris_rx = torch.squeeze(self.channels_ris_rx[user_indices, :, :])
        locations = self.locations[user_indices, :]
        channels_direct = torch.squeeze(self.channels_direct[user_indices, :])
        return [item, channel_features, channels_ris_rx, channels_direct, locations]

    def __len__(self):
        return self.group_definition.shape[0]


class RTChannelsWMMSE(RTChannels):
    def __init__(self, params, channel_tx_ris_pinv, device='cpu', test=False):
        super(RTChannelsWMMSE, self).__init__(params, channel_tx_ris_pinv, device, test=test)
        self.num_cpus = cpu_count()
        self.v = None

    def wmmse_precode(self, model, channels_tx_ris, device='cpu', num_iters=5):
        total_samples = len(self)
        num_tx_antennas = self.channels_direct.shape[2]

        v = np.empty((total_samples, num_tx_antennas, self.params["num_users"]), dtype=np.complex)
        for idx in range(0, len(self)):
            batch = self.__getitem__(idx)
            channels_ris_rx_array = batch[1][None, :, :]
            channel_ris_rx = batch[2][None, :, :]
            channel_direct = batch[3][None, :, :]
            if self.params["phase_shift"] == "discrete":
                fo = model(channels_ris_rx_array)[0].detach()
            else:
                fo = model(channels_ris_rx_array)[0].detach()
            h = compute_complete_channel_continuous(channels_tx_ris, fo,
                                                    channel_ris_rx, channel_direct,
                                                    self.params)
            if self.v is None:
                init_v = mmse_precoding(h, self.params, device)[0, :, :]
            else:
                init_v = self.v[idx, :, :]
            p = compute_wmmse_v_v2(h[0, :, :].cpu().detach().numpy(),
                                   init_v.cpu().detach().numpy(), 1, 1 / self.params['tsnr'],
                                   self.params, num_iters=num_iters)
            v[idx, :, :] = p
        self.v = torch.from_numpy(v).cfloat().to(self.device)

    def cut_data(self, num):
        self.channels_ris_rx_array = self.channel_array[:num, :, :, :]
        self.channels_ris_rx = self.channels_ris_rx[:num, :, :]
        self.channels_direct = self.channels_direct[:num, :, :]

    def __getitem__(self, item):
        data = super(RTChannelsWMMSE, self).__getitem__(item)
        if self.v is not None:
            data.append(self.v[item, :, :])

        return data

    def reset_v(self):
        self.v = None


