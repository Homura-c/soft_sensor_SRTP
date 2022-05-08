import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from math import sqrt

import torch.nn.functional as F

def reverse_sequence(x, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py
    Parameters
    ----------
    x: tensor (b, T_max, input_dim)
    seq_lengths: tensor (b, )
    Returns
    -------
    x_reverse: tensor (b, T_max, input_dim)
        The input x in reversed order w.r.t. time-axis
    """
    x_reverse = torch.zeros_like(x)
    for b in range(x.size(0)):
        t = seq_lengths[b]
        time_slice = torch.arange(t - 1, -1, -1, device=x.device)
        reverse_seq = torch.index_select(x[b, :, :], 0, time_slice)
        x_reverse[b, 0:t, :] = reverse_seq

    return x_reverse

def pad_and_reverse(rnn_output, seq_lengths):
    """
    Brought from
    https://github.com/pyro-ppl/pyro/blob/dev/examples/dmm/polyphonic_data_loader.py
    Parameters
    ----------
    rnn_output: tensor  # shape to be confirmed, should be packed rnn output
    seq_lengths: tensor (b, )
    Returns
    -------
    reversed_output: tensor (b, T_max, input_dim)
        The input sequence, unpacked and padded,
        in reversed order w.r.t. time-axis
    """
    # 没东西返回去的时候补0, 然后反转序列
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequence(rnn_output, seq_lengths)
    return reversed_output


class Emitter(nn.Module):
    def __init__(self, input_dim, z_dim, emission_dim, act_func='sigmoid'):
        super(Emitter, self).__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        self.act_func = nn.Sigmoid() if act_func == 'sigmoid' else nn.ReLU()

    def forward(self, z_t):
        """
        :param z_t: 给定latent variable的值
        :return: 给出一个分布，参量化p(x_t|z_t)
        """

        # 从z到中间态h再到output态
        h1 = self.act_func(self.lin_z_to_hidden(z_t))
        h2 = self.act_func(self.lin_hidden_to_hidden(h1))
        output = self.lin_hidden_to_input(h2)
        return output


class GatedTransition(nn.Module):
    def __init__(self, z_dim, transiton_dim):
        super(GatedTransition, self).__init__()

        # 6个转移方程
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transiton_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transiton_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transiton_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transiton_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)

        # 初始化参数
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, z_t_1):
        """
        :param z_t_1: 潜变量z_{t-1}在t-1时间步骤,
        :return: 返回参量化p(z_t | z_{t-1})的向量
        """
        # 计算门控函数
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        # _gate = self.lin_gate_z_to_hidden(z_t_1)
        gate = self.sigmoid(self.lin_gate_hidden_to_z(_gate))

        # 计算均值函数
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        # _proposed_mean = self.lin_proposed_mean_z_to_hidden(z_t_1)
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)

        # 门控机制融合(门控残差结构)
        mean = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean

        # 计算方差
        logvar = self.lin_sig(self.relu(proposed_mean))

        return mean, logvar

class Combiner(nn.Module):
    """
    构造一个反向卡尔曼平滑的滤波器
    """
    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()

        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        """
        变分推断的网络
        :param z_t_1: 过去时间的z_{t-1}
        :param h_rnn: 对未来信息的估计x_{t:T}
        :return: q(z_t|z_{t-1}, x_{t:T})
        """

        # print('[Info] The z_t_1 shape {}, h_rnn shape {}, lin_z_to_hidden {}'.format(z_t_1.shape, h_rnn.shape, self.lin_z_to_hidden))

        # 卡尔曼平滑的结果
        # h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # 计算均值和方差
        mean = self.lin_hidden_to_loc(h_combined)
        logvar = self.lin_hidden_to_scale(h_combined)

        return mean, logvar


class RNNEncoder(nn.Module):
    def __init__(self, input_dim, rnn_dim, n_layer, dropout, reverse_input=True):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.dropout = dropout
        self.reverse_input = reverse_input

        self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_dim, batch_first=True, num_layers=n_layer, dropout=dropout)

    def forward(self, x, seq_length):
        _h_rnn = self.rnn(x)
        h_rnn = pad_and_reverse(_h_rnn, seq_length)
        return h_rnn


class RNNInferenceNetwork(nn.Module):
    def __init__(self, input_dim, rnn_dim, n_layer, dropout):
        super(RNNInferenceNetwork, self).__init__()

        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layer = n_layer
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_dim, batch_first=True, num_layers=n_layer, dropout=dropout)

    def forward(self, x):
        # 反转一次, 反向计算序列
        _h_rnn = self.rnn(torch.flip(input=x, dims=[1]))

        # 再反转一次, 取切片
        h_rnn = torch.flip(_h_rnn[0], dims=[1])
        return h_rnn

class CovariateEncoderRNN(nn.Module):
    def __init__(self, covariate_dim, rnn_dim, rnn_layer, dropout):
        super(CovariateEncoderRNN, self).__init__()
        self.covariate_dim = covariate_dim
        self.rnn_dim = rnn_dim

        # GRU(input_size=input_dim, hidden_size=rnn_dim, batch_first=True, num_layers=n_layer, dropout=dropout)
        self.covariate_network = nn.GRU(input_size=covariate_dim, hidden_size=rnn_dim, num_layers=rnn_layer, dropout=dropout)

    def forward(self, x):
        return self.covariate_network(x)[0]


class CovariateEncoderMLP(nn.Module):
    def __init__(self, covariate_dim, hidden_dim):
        super(CovariateEncoderMLP, self).__init__()
        self.covariate_dim = covariate_dim
        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(nn.Linear(in_features=covariate_dim, out_features=hidden_dim),
                                     nn.Sigmoid(),
                                     nn.Linear(in_features=hidden_dim, out_features=hidden_dim))

    def forward(self, x):
        return self.network(x)


class DeepMarkovModelCovariate(nn.Module):
    def __init__(self, covariate_dim, input_dim, z_dim, emission_dim, transiton_dim, rnn_dim, rnn_layers, rnn_dropout, trainable_init):
        super(DeepMarkovModelCovariate, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transiton_dim = transiton_dim
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.rnn_dropout = rnn_dropout

        # self.reverse_rnn_input = reverse_rnn_input

        # input_dim, z_dim, emission_dim, act_func='sigmoid'
        self.emitter = Emitter(input_dim=input_dim, z_dim=z_dim, emission_dim=emission_dim)


        #  z_dim, transifiton_dim
        self.transition = GatedTransition(z_dim=z_dim, transiton_dim=transiton_dim)

        # 推断模型
        #  z_dim, rnn_dim
        self.combiner = Combiner(z_dim=z_dim, rnn_dim=rnn_dim)

        # input_dim, rnn_dim, n_layer, dropout, reverse_input=True
        self.encoder = RNNInferenceNetwork(input_dim=input_dim, rnn_dim=rnn_dim, n_layer=rnn_layers,
                                           dropout=rnn_dropout)

        self.encoder_covariate = CovariateEncoderMLP(covariate_dim=covariate_dim, hidden_dim=rnn_dim)
        self.decoder_covariate = CovariateEncoderMLP(covariate_dim=covariate_dim, hidden_dim=z_dim)

        self.z_q_0 = nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable_init)
        self.mu_p_0 = nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable_init)
        self.logvar_p_0 = nn.Parameter(torch.zeros(self.z_dim), requires_grad=trainable_init)

        self.z_q_T = torch.zeros(self.z_dim)
        self.mu_p_T = torch.zeros(self.z_dim)
        self.logvar_p_T = torch.zeros(self.z_dim)

    def _reparameterization(self, mu, logvar):
        """
        if not self.sample:
            return mu
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



    def generate_with_model(self, start_token, batch_x_past, batch_x_now, prediction_length):
        for params in self.parameters():
            params.requires_grad = False
        # 获得大小
        batch_size, seq_length = start_token.shape[0], start_token.shape[1]
        now_length = batch_x_now.shape[1]

        # 过去序列信息的取得(已经反转过序列了)
        h_rnn = self.encoder(start_token)

        z_q_0 = self.z_q_0.expand(batch_size, self.z_dim)
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.z_dim)
        logvar_p_0 = self.logvar_p_0.expand(batch_size, 1, self.z_dim)
        z_prev = z_q_0

        x_recon = torch.zeros([batch_size, seq_length, self.input_dim], device=start_token.device)
        mu_q_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)
        logvar_q_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)

        mu_p_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)
        logvar_p_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)
        z_q_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)
        z_p_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)

        x_past_encoder = self.encoder_covariate(batch_x_past.reshape((batch_size * seq_length, -1))).reshape((batch_size, seq_length, -1))
        x_future_encoder = self.decoder_covariate(batch_x_now.reshape((batch_size * prediction_length, -1))).reshape((batch_size, prediction_length, -1))


        for t in range(seq_length):
            # q(z_t|z_{t-1}, x_{t:T})
            mu_q, logvar_q = self.combiner(h_rnn=(h_rnn[:, t, :] + x_past_encoder[:, t, :]), z_t_1=z_prev)
            zt_q = self._reparameterization(mu=mu_q, logvar=logvar_q)
            z_prev = zt_q


            mu_p, logvar_p = self.transition(z_prev)
            zt_p = self._reparameterization(mu=mu_p, logvar=logvar_p)

            xt_recon = self.emitter(zt_q).contiguous()

            mu_q_seq[:, t, :] = mu_q
            logvar_q_seq[:, t, :] = logvar_q
            z_q_seq[:, t, :] = zt_q
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p_seq[:, t, :] = zt_p
            x_recon[:, t, :] = xt_recon

        # 最后时刻的东西
        # _mu_p_T, _logvar_p_T = torch.mean(mu_p_seq[:, -1, :], dim=0), torch.mean(logvar_p_seq[:, -1, :], dim=0)
        _mu_p_T, _logvar_p_T = mu_p_seq[:, -1, :], logvar_p_seq[:, -1, :]

        x_recon_pred = torch.zeros([batch_size, prediction_length, self.input_dim], device=start_token.device)
        mu_p_pred = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)
        logvar_p_pred = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)
        z_p_pred = torch.zeros([batch_size, seq_length, self.z_dim], device=start_token.device)

        pred_mu, pred_logvar = _mu_p_T, _logvar_p_T



        for t in range(prediction_length):
            # 获得了这个时刻的转移
            mu_p_pred[:, t, :], logvar_p_pred[:, t, :] = pred_mu, pred_logvar
            zp_t = self._reparameterization(mu=pred_mu, logvar=pred_logvar)

            x_t = self.emitter((zp_t + x_future_encoder[:, t, :])).contiguous()
            x_recon_pred[:, t, :], z_p_pred[:, t, :] = x_t, zp_t

            pred_mu, pred_logvar = self.transition(zp_t)
            print('[Info] The time {}, pred_mu {}, pred_logvar {}'.format(t+1, pred_mu.shape, pred_logvar.shape))



        return x_recon_pred, z_p_pred, pred_mu, pred_logvar

    def forward(self, batch_x, batch_y):

        # 获得大小
        batch_size, seq_length = batch_y.shape[0],  batch_y.shape[1]

        covariate_encoder_embedding = self.encoder_covariate(batch_x.reshape((batch_size * seq_length, -1))).reshape((batch_size, seq_length, -1))
        covariate_decoder_embedding = self.decoder_covariate(batch_x.reshape((batch_size * seq_length, -1))).reshape((batch_size, seq_length, -1))

        # 过去序列信息的取得(已经反转过序列了)
        h_rnn = self.encoder(batch_y)

        z_q_0 = self.z_q_0.expand(batch_size, self.z_dim)
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.z_dim)
        logvar_p_0 = self.logvar_p_0.expand(batch_size, 1, self.z_dim)
        z_prev = z_q_0

        x_recon = torch.zeros([batch_size, seq_length, self.input_dim], device=batch_y.device)
        mu_q_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=batch_y.device)
        logvar_q_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=batch_y.device)

        mu_p_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=batch_y.device)
        logvar_p_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=batch_y.device)
        z_q_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=batch_y.device)
        z_p_seq = torch.zeros([batch_size, seq_length, self.z_dim], device=batch_y.device)

        # print('[Info] The h_rnn shape is {}'.format(h_rnn.shape))

        for t in range(seq_length):

            #  q(z_t|z_{t-1}, x_{t:T})
            mu_q, logvar_q = self.combiner(h_rnn=(h_rnn[:, t, :] + covariate_encoder_embedding[:, t, :]), z_t_1=z_prev)

            # print('[Info] The t is {}, the h_rnn shape is {}'.format(t, h_rnn.shape))

            zt_q = self._reparameterization(mu=mu_q, logvar=logvar_q)
            z_prev = zt_q

            # p(z_t | z_{t-1})
            mu_p, logvar_p = self.transition(z_prev)
            zt_p = self._reparameterization(mu=mu_p, logvar=logvar_p)

            xt_recon = self.emitter((zt_q + covariate_decoder_embedding[:, t, :])).contiguous()

            mu_q_seq[:, t, :] = mu_q
            logvar_q_seq[:, t, :] = logvar_q
            z_q_seq[:, t, :] = zt_q
            mu_p_seq[:, t, :] = mu_p
            logvar_p_seq[:, t, :] = logvar_p
            z_p_seq[:, t, :] = zt_p
            x_recon[:, t, :] = xt_recon

        mu_p_seq = torch.cat([mu_p_0, mu_p_seq[:, :-1, :]], dim=1)
        logvar_p_seq = torch.cat([logvar_p_0, logvar_p_seq[:, :-1, :]], dim=1)

        z_p_0 = self._reparameterization(mu_p_0, logvar_p_0)
        z_p_seq = torch.cat([z_p_0, z_p_seq[:, :-1, :]], dim=1)

        # 采样序列
        # self.mu_p_T, self.logvar_p_T = torch.mean(mu_p_seq[:, -1, :], dim=0), torch.mean(logvar_p_seq[:, -1, :], dim=0)

        return x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq



def calculate_kl_term(mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq):

    # torch.exp(0.5 * logvar)
    term1 = torch.log(torch.prod(input=torch.exp(logvar_p_seq), dim=-1)) \
            - torch.log(torch.prod(input=torch.exp(logvar_q_seq), dim=-1))

    term2 = torch.sum(torch.ones_like(torch.exp(logvar_p_seq)) / torch.exp(logvar_p_seq) * torch.exp(logvar_q_seq), dim=-1)

    term3 = torch.sum((mu_q_seq - mu_p_seq) * (torch.ones_like(torch.exp(logvar_p_seq)) / torch.exp(logvar_p_seq)) * (mu_q_seq - mu_p_seq), dim=-1)

    total_term = term1 + term2 + term3

    return total_term
