import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
import random
from nonlinearSSCovariate import *
# 数据集定义方式
class MyDataset(Dataset):

    # Initialization
    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[index, :, :], self.label[index, :, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[0]

# 卡种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# SAE训练的代码模型
class TrainModels(BaseEstimator, RegressorMixin):

    def __init__(self, params_dict):
        super(TrainModels, self).__init__()

        # hidden_list=[13, 10, 7, 5], epoch=100, batch_size=64, learning_rate=0.01
        hidden_list = params_dict['hidden_list']


        learning_rate = params_dict['learning_rate']
        seed = params_dict['seed']
        self.input_dimension = params_dict['input_dimension']
        self.output_dimension = params_dict['output_dimension']

        z_dimension = params_dict['z_dimension']
        emission_dimension = params_dict['emission_dimension']
        transition_dimension = params_dict['transition_dimension']
        rnn_dimension = params_dict['rnn_dimension']
        rnn_layers = params_dict['rnn_layers']
        rnn_dropout = params_dict['dropout']
        trainable_init = params_dict['trainable_init']

        self.batch_size = params_dict['batch_size']
        self.device = params_dict['device']
        self.epoch = params_dict['epoch']

        self.seq_length = params_dict['sequence_length']

        # scale化参数大小
        self.scaler_x = MinMaxScaler()
        self.scaler_y = StandardScaler()

        # 模型实例化
        # input_dim, z_dim, emission_dim, transiton_dim, rnn_dim, rnn_layers, rnn_dropout, trainable_init
        # 三种不同的模型开始造作
        """
        self.model = DeepMarkovModel(input_dim=self.output_dimension, z_dim=z_dimension, emission_dim=emission_dimension,
                                     transiton_dim=transition_dimension, rnn_dim=rnn_dimension, rnn_layers=rnn_layers,
                                     rnn_dropout=rnn_dropout, trainable_init=trainable_init).to(self.device)
        """

        self.model = DeepMarkovModelCovariate(covariate_dim=self.input_dimension, input_dim=self.output_dimension, z_dim=z_dimension,
                                              emission_dim=emission_dimension, transiton_dim=transition_dimension,
                                              rnn_dim=rnn_dimension, rnn_layers=rnn_layers,
                                              rnn_dropout=rnn_dropout, trainable_init=trainable_init).to(self.device)
        """
        self.model = DeepMarkovModelCovariateSAM(covariate_dim=self.input_dimension, input_dim=self.output_dimension,
                                              z_dim=z_dimension,
                                              emission_dim=emission_dimension, transiton_dim=transition_dimension,
                                              rnn_dim=rnn_dimension, rnn_layers=rnn_layers,
                                              rnn_dropout=rnn_dropout, trainable_init=trainable_init).to(self.device)

        """


        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.reconstruction_loss = nn.MSELoss(reduction='sum')


    def _loss_function(self, predict_y, label_y, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq):
        def _calculate_kl_term(mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq):
            # torch.exp(0.5 * logvar)
            term1 = torch.log(torch.prod(input=torch.exp(logvar_p_seq), dim=-1)) \
                    - torch.log(torch.prod(input=torch.exp(logvar_q_seq), dim=-1))

            term2 = torch.sum(
                torch.ones_like(torch.exp(logvar_p_seq)) / torch.exp(logvar_p_seq) * torch.exp(logvar_q_seq), dim=-1)

            term3 = torch.sum(
                (mu_q_seq - mu_p_seq) * (torch.ones_like(torch.exp(logvar_p_seq)) / torch.exp(logvar_p_seq)) * (
                            mu_q_seq - mu_p_seq), dim=-1)

            total_term = term1 + term2 + term3

            return total_term

        batch_size = predict_y.shape[0]

        # 计算KL散度
        kl_loss = torch.sum(_calculate_kl_term(mu_q_seq=mu_q_seq, logvar_q_seq=logvar_q_seq,
                                               mu_p_seq=mu_p_seq, logvar_p_seq=logvar_p_seq)) / batch_size

        # 预测误差
        prediction_loss = torch.sum(self.reconstruction_loss(predict_y, label_y)) / batch_size

        # 总误差
        total_loss = prediction_loss + 0.01 * kl_loss

        return total_loss, prediction_loss, kl_loss


    # 数据拟合
    def fit(self, input_value, label):

        X = self.scaler_x.fit_transform(input_value)
        # y = label.reshape(-1, self.output_dimension)
        y = self.scaler_y.fit_transform(label.reshape(-1, self.output_dimension))

        X_3d = list(map(lambda u: X[u : u + self.seq_length], range(X.shape[0] - self.seq_length + 1)))
        y_3d = list(map(lambda u: y[u: u + self.seq_length], range(X.shape[0] - self.seq_length + 1)))

        X_3d = np.stack(X_3d, 0)
        y_3d = np.stack(y_3d, 0)

        dataset = MyDataset(torch.tensor(X_3d, dtype=torch.float32, device=self.device),
                            torch.tensor(y_3d, dtype=torch.float32, device=self.device), '3D')



        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()


        for e in range(self.epoch):
            sum_total_loss = 0.0
            sum_reconstruction_loss = 0.0
            sum_kl_loss = 0.0
            for batch_x, batch_y in train_loader:

                # print('[Info] The batch x shape {}, batch y shape {}'.format(batch_x.shape, batch_y.shape))

                prediction_term, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq  = self.model(batch_x=batch_x, batch_y=batch_y)

                # real_label = torch.cat([batch_x, batch_y], dim=-1)


                # output = torch.cat([reconstruction, prediction], dim=-1)

                # print('[Info] The reconstruction shape {}, the prediction shape {}'.format(reconstruction.shape, prediction.shape))
                #
                # print('[Info] The output shape {} and real shape {}'.format(real_label.shape, output.shape))

                # predict_y, label_y, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq
                total_loss, reconstruction_loss, kl_loss = self._loss_function(predict_y=prediction_term, label_y=batch_y,
                                                                               mu_q_seq=mu_q_seq, logvar_q_seq=logvar_q_seq,
                                                                               mu_p_seq=mu_p_seq, logvar_p_seq=logvar_p_seq)

                # print('[Info] The total loss shape {}'.format(total_loss.shape))


                self.optimizer.zero_grad()
                total_loss.backward()



                self.optimizer.step()
                sum_total_loss = sum_total_loss + total_loss.detach().item()
                sum_reconstruction_loss = sum_reconstruction_loss + reconstruction_loss.detach().item()
                sum_kl_loss = sum_kl_loss + kl_loss.detach().item()
            print('Epoch {}, total loss {}, reconstruction loss {}, kl divergence {}'.format(e+1, sum_total_loss, sum_reconstruction_loss, sum_kl_loss))
        return self

    def predict(self, start_token_x, start_token_y, future_x, prediction_length):

        start_token_batch, start_token_length = start_token_y.shape[0], start_token_y.shape[1]

        x_past_batch, x_past_length = start_token_x.shape[0], start_token_x.shape[1]
        x_future_batch, x_future_length = future_x.shape[0], future_x.shape[1]

        start_token_scaled = self.scaler_y.transform(start_token_y.reshape((-1, self.output_dimension)))

        x_past_scaled = self.scaler_x.transform(start_token_x.reshape((-1, self.input_dimension))).reshape((x_past_batch, x_past_length, self.input_dimension))
        x_future_scaled = self.scaler_x.transform(future_x.reshape((-1, self.input_dimension))).reshape((x_future_batch, x_future_length, self.input_dimension))

        x_future_scaled = torch.as_tensor(x_future_scaled, dtype=torch.float32, device=self.device)
        x_past_scaled = torch.as_tensor(x_past_scaled, dtype=torch.float32, device=self.device)

        start_token_scaled = start_token_scaled.reshape((start_token_batch, start_token_length, self.output_dimension))

        start_token_scaled = torch.as_tensor(start_token_scaled, dtype=torch.float32, device=self.device)

        # for name, parms in self.model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)


        self.model.eval()
        with torch.no_grad():
            prediction_value, z_p_pred, pred_mu, pred_logvar = self.model.generate_with_model(start_token=start_token_scaled,
                                                                                              batch_x_past=x_past_scaled,
                                                                                              batch_x_now=x_future_scaled,
                                                                                              prediction_length=prediction_length)
        prediction_value_reshape = prediction_value.cpu().numpy().reshape((-1, self.output_dimension))
        # prediction_value_scaled = prediction_value_reshape
        prediction_value_scaled = self.scaler_y.inverse_transform(prediction_value_reshape)
        prediction_value_reshape = prediction_value_scaled.reshape((start_token_batch, prediction_length, self.output_dimension))
        return prediction_value_reshape
    """   
    def predict(self, X):
        X = self.scaler_x.transform(X)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            y = self.model(X)[0][1].cpu().numpy()[:, -1].reshape((-1, 1))

        return y
    """


def construct_prediction(test_x, test_y, start_token_length, prediction_length, output_dimension=1):


    data_number = test_x.shape[0]

    start_x = list(map(lambda u: test_x[u: u + start_token_length], range(data_number - (start_token_length + prediction_length) + 1)))
    start_y = list(map(lambda u: test_y[u: u + start_token_length].reshape((-1, output_dimension)), range(data_number - (start_token_length + prediction_length) + 1)))

    future_x = list(map(lambda u: test_x[u + start_token_length: u + start_token_length + prediction_length], range(data_number - (start_token_length + prediction_length) + 1)))
    future_y = list(map(lambda u: test_y[u + start_token_length: u + start_token_length + prediction_length].reshape((-1, output_dimension)), range(data_number - (start_token_length + prediction_length) + 1)))

    start_x, start_y = np.stack(start_x, 0), np.stack(start_y, 0)
    future_x, future_y = np.stack(future_x, 0), np.stack(future_y, 0)
    return start_x, start_y, future_x, future_y



if __name__ == '__main__':
    setup_seed(seed=42)

    # 数据读取:
    data = pd.read_csv('Debutanizer_Data.txt', sep='\s+')
    data = data.values

    train_x = data[:1200, :7]
    train_y = data[:1200, 7]

    x_validation = data[1200:1600, :7]
    y_validation = data[1200:1600, 7]

    test_x = data[1600:, :7]
    test_y = data[1600:, 7]



    """
            learning_rate = params_dict['learning_rate']
        seed = params_dict['seed']
        self.input_dimension = params_dict['input_dimension']
        self.output_dimension = params_dict['output_dimension']

        z_dimension = params_dict['z_dimension']
        emission_dimension = params_dict['emission_dimension']
        transition_dimension = params_dict['transition_dimension']
        rnn_dimension = params_dict['rnn_dimension']
        rnn_layers = params_dict['rnn_layers']
        rnn_dropout = params_dict['dropout']
        trainable_init = params_dict['trainable_init']
    """




    HIDDEN_LIST = [10, 7, 5]
    BATHC_SIZE = 32
    LEARNING_RATE = 0.003
    EPOCH = 100
    SEED = 1024
    INPUPT_DIMENSION = 7
    PREDICTION_DIMENSION = 1
    OUTPUT_DIMENSION = PREDICTION_DIMENSION # + INPUPT_DIMENSION
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Z_DIMENSION = 4
    EMISSION_DIMENSION = 4
    TRANSITION_DIMENSION = 4
    RNN_DIMENSION = 4
    RNN_LAYERS = 1
    RNN_DROPOUT = 0.5
    IS_INIT = True
    SEQUENCE_LENGTH = 16
    PREDICTION_LENGTH = 4
    START_TOKEN_LENGTH = SEQUENCE_LENGTH - PREDICTION_LENGTH

    parameters = {'hidden_list': HIDDEN_LIST, 'batch_size': BATHC_SIZE, 'learning_rate': LEARNING_RATE,
                  'seed': SEED, 'input_dimension': INPUPT_DIMENSION, 'output_dimension': OUTPUT_DIMENSION,
                  'device': DEVICE, 'epoch': EPOCH, 'z_dimension': Z_DIMENSION, 'emission_dimension': EMISSION_DIMENSION,
                  'transition_dimension': TRANSITION_DIMENSION, 'rnn_dimension': RNN_DIMENSION, 'rnn_layers': RNN_LAYERS,
                  'dropout': RNN_DROPOUT, 'trainable_init': IS_INIT, 'sequence_length': SEQUENCE_LENGTH}

    start_token_x, start_token_y, future_x, future_y = construct_prediction(test_x=test_x, test_y=test_y,
                                                                            start_token_length=START_TOKEN_LENGTH,
                                                                            prediction_length=PREDICTION_LENGTH)

    print('[Info] The shape are : start_token_x {}, start_token_y {}, future_x {}, future_y {}'.format(start_token_x.shape, start_token_y.shape, future_x.shape, future_y.shape))

    # sys.exit(0)



    # 开始搞活
    mdl = TrainModels(params_dict=parameters).fit(train_x, train_y)

    # start_token_x, start_token_y, future_x, future_y
    prediction_value = mdl.predict(start_token_x=start_token_x, start_token_y=start_token_y,
                                   future_x=future_x, prediction_length=PREDICTION_LENGTH)

    print('[Info] Print the future known shape {}'.format(prediction_value.shape))

    output_test = prediction_value[:, -1, :]
    real_test = future_y[:, -1, :]

    plt.figure()
    plt.plot(range(len(output_test)), output_test, color='b', label='y_testpre')
    plt.plot(range(len(output_test)), real_test, color='r', label='y_true')
    plt.legend()
    plt.show()
    test_rmse = np.sqrt(mean_squared_error(output_test, real_test))
    test_r2 = r2_score(output_test, real_test)
    print('test_rmse = ' + str(round(test_rmse, 5)), 'r2 = ' + str(round(test_r2, 5)))
    # print()

    """

    output_train = mdl.predict(X=train_x)


    train_rmse = np.sqrt(mean_squared_error(output_train, train_y))
    train_r2 = r2_score(output_train, train_y)

    plt.figure()
    plt.plot(range(len(output_train)), output_train, color='b', label='y_testpre')
    plt.plot(range(len(output_train)), train_y, color='r', label='y_true')
    plt.legend()
    plt.show()

    print('train rmse = ' + str(round(train_rmse, 5)) + ' train r2 = ' + str(train_r2))
    """
    # print('r2 = ', str(test_r2))


    """
    output_train = mdl.predict(train_x)
    output_test = mdl.predict(test_x)

    # 训练集画图
    plt.figure()
    plt.plot(range(len(output_train)), output_train, color='b', label='y_trainpre')
    plt.plot(range(len(output_train)), train_y, color='r', label='y_true')
    plt.legend()
    plt.show()
    train_rmse = np.sqrt(mean_squared_error(output_train, train_y))
    train_r2 = r2_score(output_train, train_y)
    print('train_rmse = ' + str(round(train_rmse, 5)))
    print('r2 = ', str(train_r2))

    # 测试集画图
    plt.figure()
    plt.plot(range(len(output_test)), output_test, color='b', label='y_testpre')
    plt.plot(range(len(output_test)), test_y, color='r', label='y_true')
    plt.legend()
    plt.show()
    test_rmse = np.sqrt(mean_squared_error(output_test, test_y))
    test_r2 = r2_score(output_test, test_y)
    print('test_rmse = ' + str(round(test_rmse, 5)))
    print('r2 = ', str(test_r2))
    """