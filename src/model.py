import torch
from torch.nn.modules import Module, LSTM
from .modules import GaussianWindow, MDN
import numpy as np
from fast_transformers.recurrent.transformers import RecurrentTransformerEncoderLayer
from fast_transformers.recurrent.attention import RecurrentAttentionLayer, RecurrentLinearAttention


class HandwritingGenerator(Module):
    def __init__(
        self, alphabet_size, hidden_size, num_window_components, num_mixture_components
    ):
        super(HandwritingGenerator, self).__init__()
        self.alphabet_size = alphabet_size
        self.hidden_size = hidden_size
        self.num_window_components = num_window_components
        self.num_mixture_components = num_mixture_components
        # print(num_window_components)
        # print(num_mixture_components)
        # print(hidden_size)
        self.input_size = input_size = 3
        n_heads_1 = 2
        n_heads_2 = 4
        # First LSTM layer, takes as input a tuple (x, y, eol)
        # self.lstm1_layer = LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)

        self.lstm1_layer = RecurrentTransformerEncoderLayer(
            RecurrentAttentionLayer(RecurrentLinearAttention(1), input_size, n_heads_1),
            input_size,
            hidden_size,
            activation="gelu"
        )
        self.lstm1_layer2 = torch.nn.Linear(input_size, hidden_size)

        # Gaussian Window layer
        self.window_layer = GaussianWindow(
            input_size=hidden_size, num_components=num_window_components
        )
        # Second LSTM layer, takes as input the concatenation of the input,
        # the output of the first LSTM layer
        # and the output of the Window layer
        # self.lstm2_layer = LSTM(
        #     input_size=3 + hidden_size + alphabet_size + 1,
        #     hidden_size=hidden_size,
        #     batch_first=True,
        # )
        self.lstm2_layer = RecurrentTransformerEncoderLayer(
            RecurrentAttentionLayer(RecurrentLinearAttention(1), 3 + hidden_size + alphabet_size + 1, n_heads_2),
            3 + hidden_size + alphabet_size + 1,
            hidden_size,
            activation="gelu"
        )

        # Third LSTM layer, takes as input the concatenation of the output of the first LSTM layer,
        # the output of the second LSTM layer
        # and the output of the Window layer
        # self.lstm3_layer = LSTM(
        #     input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        # )
        # print( 3 + hidden_size + alphabet_size + 1)
        # print(hidden_size)
        self.lstm3_layer = RecurrentTransformerEncoderLayer(
            RecurrentAttentionLayer(RecurrentLinearAttention(1), 3 + hidden_size + alphabet_size + 1, n_heads_2),
            3 + hidden_size + alphabet_size + 1,
            hidden_size,
            activation="gelu"
        )
        self.lstm3_layer2 = torch.nn.LayerNorm(3 + hidden_size + alphabet_size + 1)

        # Mixture Density Network Layer
        self.output_layer = MDN(
            input_size=3 + hidden_size + alphabet_size + 1, num_mixtures=num_mixture_components
        )

        # Hidden State Variables
        self.prev_kappa = None
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

        # Initiliaze parameters
        self.reset_parameters()

    def forward(self, strokes, onehot, bias=None):
        # First LSTM Layer
        input_ = strokes
        # self.lstm1_layer.flatten_parameters()
        # print(input_.shape)
        output1, self.hidden1 = self.lstm1_layer(input_.reshape(-1,self.input_size), self.hidden1)
        # print(output1.shape)
        output1 = self.lstm1_layer2(output1)
        output1 = output1.reshape(-1,1,self.hidden_size)
        # print(output1.shape)
        # print(onehot.shape)
        # print(self.prev_kappa)
        # print(output1.shape, self.hidden1.shape)
        # output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)
        # output1 = []
        # self.hidden1 = []
        # for i in input_:
        #     o = self.lstm1_layer(i, self.hidden1)
        #     print(o)
        #     output1.append(o)
        #     self.hidden1.append(h1)
        # print(output1.shape)
        # Gaussian Window Layer
        window, self.prev_kappa, phi = self.window_layer(
            output1, onehot, self.prev_kappa
        )
        # print(output1.shape)
        # print(strokes.shape)
        # print(window.shape)
        # print(self.hidden2)
        # Second LSTM Layer
        # torch.squeeze(output1)
        # print(torch.cat((strokes, output1, window), dim=2).shape)
        output2, self.hidden2 = self.lstm2_layer(
            torch.cat((strokes, output1, window), dim=2).reshape(-1,strokes.shape[-1] + output1.shape[-1] + window.shape[-1]), self.hidden2
        )
        # print(output2.shape)
        # print([h.shape for h in self.hidden2])
        # print(self.hidden3.shape)
        # Third LSTM Layer
        output3, self.hidden3 = self.lstm3_layer(output2, self.hidden3)
        output3 = self.lstm3_layer2(output3)
        # MDN Layer
        eos, pi, mu1, mu2, sigma1, sigma2, rho = self.output_layer(output3.reshape(-1,1,output3.shape[-1]), bias)
        return (eos, pi, mu1, mu2, sigma1, sigma2, rho), (window, phi)

    @staticmethod
    def sample_bivariate_gaussian(pi, mu1, mu2, sigma1, sigma2, rho):
        # Pick distribution from the MDN
        p = pi.data[0, 0, :].numpy()
        idx = np.random.choice(p.shape[0], p=p)
        m1 = mu1.data[0, 0, idx]
        m2 = mu2.data[0, 0, idx]
        s1 = sigma1.data[0, 0, idx]
        s2 = sigma2.data[0, 0, idx]
        r = rho.data[0, 0, idx]
        mean = [m1, m2]
        covariance = [[s1 ** 2, r * s1 * s2], [r * s1 * s2, s2 ** 2]]
        Z = torch.autograd.Variable(
            sigma1.data.new(np.random.multivariate_normal(mean, covariance, 1))
        ).unsqueeze(0)
        X = Z[:, :, 0:1]
        Y = Z[:, :, 1:2]
        return X, Y

    def reset_state(self):
        self.prev_kappa = None
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

    def reset_parameters(self):
        for parameter in self.parameters():
            if len(parameter.size()) == 2:
                torch.nn.init.xavier_uniform_(parameter, gain=1.0)
            else:
                stdv = 1.0 / parameter.size(0)
                torch.nn.init.uniform_(parameter, -stdv, stdv)

    def num_parameters(self):
        num = 0
        for weight in self.parameters():
            num = num + weight.numel()
        return num

    @classmethod
    def load_model(cls, parameters: dict, state_dict: dict):
        model = cls(**parameters)
        model.load_state_dict(state_dict)
        return model

    def __deepcopy__(self, *args, **kwargs):
        model = HandwritingGenerator(
            self.alphabet_size,
            self.hidden_size,
            self.num_window_components,
            self.num_mixture_components,
        )
        return model
