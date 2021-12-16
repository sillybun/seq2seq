import torch
import torch.nn as nn
from collections import OrderedDict
from zytlib.classfunc import save_args
from zytlib import vector
from zytlib.wrapper import registered_property
import random
from torchfunction.lossfunc import softmax_cross_entropy_with_logits_sparse
from torchfunction.select import select_index_by_batch
from zytlib.table import table
import math

def low_rank(self, n, r, name="J", order_one_init=False):
    self.__setattr__("_" + name + "_r", r)
    if r == -1:
        if order_one_init:
            self.__setattr__("_" + name, nn.Parameter(torch.randn(n, n)))
        else:
            self.__setattr__("_" + name, nn.Parameter(torch.randn(n, n) * math.sqrt(n)))
    else:
        self.__setattr__("_" + name + "_V", nn.Parameter(torch.randn(n, r)))
        self.__setattr__("_" + name + "_U", nn.Parameter(torch.randn(n, r)))
    def func(self):
        if getattr(self, f"_{name}_r") == -1:
            return getattr(self, f"_{name}")
        else:
            return getattr(self, f"_{name}_U") @ getattr(self, f"_{name}_V").T
    def rmul(self, x):
        if getattr(self, f"_{name}_r") == -1:
            return torch.matmul(x, getattr(self, f"_{name}"))
        else:
            return torch.matmul(torch.matmul(x, getattr(self, f"_{name}_U")), getattr(self, f"_{name}_V").T)
    def load_state_dict(state: table):
        max_rank = r
        if isinstance(state, nn.Module):
            state = state.state_dict()
        state = state.copy()
        if max_rank == -1 and f"_{name}" in state:
            getattr(self, f"_{name}").data.copy_(state[f"_{name}"])
        elif max_rank != -1 and f"_{name}_U" in state and f"_{name}_V" in state:
            U = state[f"_{name}_U"]
            V = state[f"_{name}_V"]
            getattr(self, f"_{name}_V").data.copy_(V)
            getattr(self, f"_{name}_U").data.copy_(U)
        elif max_rank != -1 and f"_{name}" in state:
            U, S, V = torch.pca_lowrank(state[f"_{name}"], q=max_rank, center=False)
            getattr(self, f"_{name}_V").data.copy_(V @ torch.diag(torch.sqrt(S)))
            getattr(self, f"_{name}_U").data.copy_(U @ torch.diag(torch.sqrt(S)))
        elif max_rank == -1 and f"_{name}_U" in state and f"_{name}_V" in state:
            U = state[f"_{name}_U"]
            V = state[f"_{name}_V"]
            getattr(self, f"_{name}").data.copy_(U @ V.T)
    if not hasattr(self, name):
        setattr(self.__class__, name, property(func))
        setattr(self.__class__, name + "_rmul", rmul)
    return load_state_dict

class encoder_rnn(nn.Module):

    def __init__(self, tau, delta_t, embedding_dim, noise_sigma, gain=1.0, p=1.0, encoder_bias=False, encoder_max_rank=-1, order_one_init=False):
        super().__init__()
        save_args(vars())
        self.init()

    def init(self):
        self.load_states = vector()
        lsd = low_rank(self, self.hyper.embedding_dim, self.hyper.encoder_max_rank, name="J", order_one_init=self.hyper.order_one_init)
        self.load_states.append(lsd)
        if self.hyper.encoder_max_rank == -1:
            self._J.gain = self.gain
        else:
            self._J_U.gain = self._J_V.gain = math.sqrt(self.gain)

        self.alpha = self.hyper["delta_t"] / self.hyper["tau"]
        if self.hyper['encoder_bias']:
            self.bias = nn.Parameter(torch.zeros(1, self.hyper['embedding_dim']))
            self.load_states.append(lambda state: self.bias.data.copy_(state["bias"]))
        assert 0 < self.alpha < 1

    def forward(self, x: torch.Tensor, I: torch.Tensor, noise_sigma=None) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(I) * self.hyper["noise_sigma"]
        elif noise_sigma is not None:
            noise = torch.randn_like(I) * noise_sigma
        else:
            noise = torch.zeros_like(I)

        if self.hyper['encoder_bias']:
            I = I + self.bias

        # x = (1 - self.alpha) * x + self.alpha * (torch.matmul(torch.tanh(x), self.J * self.gain) + I + noise)
        x = (1 - self.alpha) * x + self.alpha * (self.J_rmul(torch.tanh(x)) * self.gain + I + noise)
        return x

    @property
    def regulization_parameters(self) -> table:
        if self.hyper.encoder_max_rank == -1:
            return table(J=self._J * self.gain)
        else:
            return table(J_U=self._J_U, J_V=self._J_V).map(value=lambda x: x * x.gain)

    def inspect(self) -> table:
        if self.hyper["encoder_max_rank"] == -1:
            rank = min(10, self.hyper.embedding_dim)
        else:
            rank = self.hyper["encoder_max_rank"]
        U, S, V = torch.pca_lowrank(self.J * self.gain, q=rank)
        return table({"lambda[{}]".format(index + 1): S[index].item() for index in range(rank)})

    def load_state_dict(self, encoder_state: OrderedDict):
        self.load_states.apply(lambda x: x(encoder_state))

    @registered_property
    def gain(self) -> float:
        return self.hyper.gain / self.hyper.embedding_dim ** self.hyper.p

    def __repr__(self) -> str:
        return f"encoder_rnn(n={self.hyper.embedding_dim})"

class linear_decoder_rnn(nn.Module):

    def __init__(self, decoder_dim, decoder_bias=False, decoder_max_rank=-1, order_one_init=False):

        super().__init__()
        save_args(vars())
        self.init()

    def init(self):
        self.load_states = vector()
        lsd = low_rank(self, self.hyper.decoder_dim, self.hyper.decoder_max_rank, name="W", order_one_init=self.hyper.order_one_init)
        self.load_states.append(lsd)
        if self.hyper["decoder_max_rank"] == -1:
            self._W.gain = self.gain
        else:
            self._W_U.gain = self._W_V.gain = math.sqrt(self.gain)
        if self.hyper["decoder_bias"]:
            self.bias = nn.Parameter(torch.zeros(1, self.hyper["decoder_dim"]))
            self.load_states.append(lambda state: self.b.data.copy_(state["bias"]))

    def forward(self, h, I):
        h = self.W_rmul(h) * self.gain
        if self.hyper["decoder_bias"]:
            h = h + self.bias
        return h

    @registered_property
    def gain(self) -> float:
        return 1 / self.hyper.decoder_dim

    @property
    def regulization_parameters(self) -> table:
        if self.hyper["decoder_max_rank"] == -1:
            return table(W=self._W * self.gain)
        else:
            return table(W_U=self._W_U, W_V=self._W_V).map(value=lambda x: x * x.gain)

    def inspect(self) -> table:
        if self.hyper["decoder_max_rank"] == -1:
            rank = min(10, self.hyper.decoder_dim)
        else:
            rank = self.hyper["decoder_max_rank"]
        U, S, V = torch.pca_lowrank(self.W * self.gain, q=rank)
        return table({"lambda[{}]".format(index + 1): S[index].item() for index in range(rank)})

    def load_state_dict(self, decoder_state: OrderedDict) -> None:
        self.load_states.apply(lambda x: x(decoder_state))

# class decoder_rnn(nn.Module):

#     def __init__(self, embedding_dim, input_dim, decoder_bias=True, decoder_max_rank=-1):

#         super().__init__()
#         save_args(vars())
#         self.init()

#     def init(self):
#         if self.hyper["decoder_max_rank"] == -1:
#             self._W1 = nn.Parameter(torch.zeros(self.hyper["embedding_dim"], self.hyper["embedding_dim"]))
#             torch.nn.init.normal_(self._W, std=1 / math.sqrt(self.hyper["embedding_dim"]))
#         else:
#             self._V = nn.Parameter(torch.zeros(self.hyper['embedding_dim'], self.hyper["decoder_max_rank"]))
#             self._U = nn.Parameter(torch.zeros(self.hyper['embedding_dim'], self.hyper["decoder_max_rank"]))
#             nn.init.xavier_normal_(self._V)
#             nn.init.xavier_normal_(self._U)
#         self.W2 = nn.Parameter(torch.zeros(self.hyper["input_dim"], self.hyper["embedding_dim"]))
#         torch.nn.init.normal_(self.W1, std=1/math.sqrt(self.hyper["embedding_dim"]))
#         torch.nn.init.normal_(self.W1, std=1/math.sqrt(self.hyper["embedding_dim"]))
#         if self.hyper["decoder_bias"]:
#             self.bias = nn.Parameter(torch.zeros(1, self.hyper["embedding_dim"]))

#     def forward(self, h, I):
#         if self.hyper["decoder_bias"]:
#             h = torch.tanh(h @ self.W1 + I @ self.W2 + self.bias)
#         else:
#             h = torch.tanh(h @ self.W1 + I @ self.W2)
#         return h

#     @property
#     def W1(self):
#         if self.hyper["decoder_max_rank"] == -1:
#             return self._W
#         else:
#             return self._V @ self._U.T

#     @property
#     def regulization_parameters(self):
#         if self.hyper["decoder_max_rank"] == -1:
#             return vector([self.W1, self.W2])
#         else:
#             return vector([self._U, self._V, self.W2])

#     def inspect(self) -> table:
#         if self.hyper["decoder_max_rank"] == -1:
#             rank = 20
#         else:
#             rank = self.hyper["decoder_max_rank"]
#         U, S, V = torch.pca_lowrank(self.W1, q=rank)
#         return table({"lambda[{}]".format(index + 1): S[index].item() for index in range(rank)})

#     def load_state_dict(self, decoder_state):
#         if self.hyper["decoder_max_rank"] != -1 and "rnn_cell._W1" in decoder_state:
#             U, S, V = torch.pca_lowrank(decoder_state["rnn_cell._W1"].T, q=self.hyper["decoder_max_rank"])
#             self._V.data.copy_(V @ torch.diag(S))
#             self._U.data.copy_(U)

class encoder(nn.Module):

    def __init__(self, tau, delta_t, embedding_dim, noise_sigma, input_dim, input_gain=1.0, embedding=None, is_embedding_fixed=True, encoder_bias=False, encoder_max_rank=-1, timer=None, order_one_init=False):

        super().__init__()
        save_args(vars(), ignore=("timer",))

        self.rnn = encoder_rnn(tau, delta_t, embedding_dim, noise_sigma, encoder_bias=encoder_bias, encoder_max_rank=encoder_max_rank, order_one_init=order_one_init)
        if embedding is None:
            embedding = torch.randn(input_dim, embedding_dim)
        else:
            embedding = embedding
        if is_embedding_fixed:
            self.embedding = embedding
        else:
            self.embedding = nn.Parameter(embedding)
        if timer is not None:
            self.forward = timer(self.forward)

    def forward(self, u, length, noise_sigma=None, only_final_state=False):
        """
        u.shape: [batch, time, input_dim]
        length.shape: [batch]

        output:
        ret.shape: [batch, time(+1), embedding_dim]
        """

        batch_size = u.shape[0]

        sorted_length, length_index = torch.sort(length, dim=-1, descending=True)
        u = u[length_index, :, :]

        x = torch.zeros(sorted_length[0].item() + 1, batch_size, self.hyper['embedding_dim'], device=u.device)

        for index in range(sorted_length[0].item()):

            batch = sum(sorted_length > index).item()
            if batch == 0:
                break
            x[index + 1, :batch, :] = self.rnn(x[index, :batch, :], torch.matmul(u[:batch, index, :], self.embedding) * self.hyper.input_gain, noise_sigma=noise_sigma)

        x = x.transpose(0, 1)

        ret = torch.zeros_like(x)
        ret[length_index, :, :] = x

        final_state = torch.gather(ret, 1, length.view(batch_size, 1, 1).expand(batch_size, 1, self.hyper['embedding_dim'])).squeeze(1)

        if only_final_state:
            return final_state
        else:
            return ret, final_state

    @property
    def regulization_parameters(self) -> table:
        return table(rnn=self.rnn.regulization_parameters)

    def inspect(self) -> table:
        ret = self.rnn.inspect()
        ret += self.regulization_parameters.flatten().map(value=lambda x: torch.sum(torch.square(x)).item())
        return ret.map(key=lambda x: "encoder." + x)

    def load_state_dict(self, encoder_state):
        if isinstance(encoder_state, nn.Module):
            encoder_state = encoder_state.state_dict()
        encoder_state = table.hieratical(encoder_state)
        if "embedding" in encoder_state and isinstance(self.embedding, nn.Parameter):
            self.embedding.data.copy_(encoder_state["embedding"])
        self.rnn.load_state_dict(encoder_state.rnn)

class decoder(nn.Module):

    def __init__(self, decoder_dim, encoder_dim, input_dim, output_dim, input_embedding, linear_decoder=True, decoder_bias=None, encoder_to_decoder_equal_space=False, decoder_max_rank=-1, timer=None, order_one_init=False):

        super().__init__()
        save_args(vars(), ignore=("timer",))
        if linear_decoder:
            if decoder_bias is None:
                decoder_bias = False
            self.rnn_cell = linear_decoder_rnn(decoder_dim, decoder_bias=decoder_bias, decoder_max_rank=decoder_max_rank, order_one_init=order_one_init)
        else:
            raise NotImplementedError()

        self.linear_layer = nn.Parameter(torch.zeros(decoder_dim, output_dim))

        if not encoder_to_decoder_equal_space:
            self.input_to_decoder = nn.Parameter(torch.zeros(encoder_dim, decoder_dim))

        self.init()
        if timer is not None:
            self.forward = timer(self.forward)

    def init(self) -> None:
        nn.init.normal_(self.linear_layer)
        if not self.hyper.order_one_init:
            self.linear_layer.data.mul_(math.sqrt(self.hyper.decoder_dim))
        self.linear_layer.gain = 1 / self.hyper.decoder_dim
        if not self.hyper["encoder_to_decoder_equal_space"]:
            nn.init.normal_(self.input_to_decoder)
            if not self.hyper.order_one_init:
                self.input_to_decoder.data.mul_(math.sqrt(self.hyper.encoder_dim))
            self.input_to_decoder.gain = 1 / self.hyper.encoder_dim

    def readout(self, hidden_state):
        return hidden_state @ self.linear_layer * self.linear_layer.gain

    def forward(self, hidden_state, ground_truth_tensor, ground_truth_length, teaching_forcing_ratio=0.5):
        """
        input:
            hidden_state: [batch, embedding_dim]
            ground_truth_tensor: [batch, max_length]
            ground_truth_length: [batch]

        output:
            ret: [batch, max_length, embedding_dim]
        """


        ground_truth_length, sorted_index = ground_truth_length.sort(dim=-1, descending=True)
        ground_truth_tensor = ground_truth_tensor[sorted_index, :]
        if self.hyper["encoder_to_decoder_equal_space"]:
            hidden_state = torch.tanh(hidden_state[sorted_index, :])
        else:
            hidden_state = torch.tanh(hidden_state[sorted_index, :]) @ self.input_to_decoder * self.input_to_decoder.gain
        input = torch.zeros(hidden_state.shape[0], self.hyper["input_dim"]).to(hidden_state.device)

        batch_size = hidden_state.shape[0]
        max_length = ground_truth_length[0].item()

        outputs = torch.zeros([batch_size, max_length, self.hyper["output_dim"]], device=hidden_state.device)

        decoder_hidden_state = torch.zeros([batch_size, max_length + 1, self.hyper["decoder_dim"]], device=hidden_state.device)
        decoder_hidden_state[:, 0, :] = hidden_state

        for index in range(max_length):
            batch = sum(index < ground_truth_length)
            if batch == 0:
                break

            hidden_state = self.rnn_cell(hidden_state[:batch, :], input[:batch, :])

            # prediction = hidden_state @ self.linear_layer * self.linear_layer.gain
            prediction = self.readout(hidden_state)

            decoder_hidden_state[:batch, index + 1, :] = hidden_state
            outputs[:batch, index, :] = prediction

        ret = torch.zeros_like(outputs)
        hidden_state_ret = torch.zeros_like(decoder_hidden_state)
        ret[sorted_index, :, :] = outputs
        hidden_state_ret[sorted_index, :, :] = decoder_hidden_state

        return ret, hidden_state_ret

    def loss(self, ground_truth, ground_truth_length, prediction):
        """
        input:
            ground_truth: [batch, max_length]
            ground_truth_length: [batch]
            prediction: [batch, max_length, output_dim]
        """

        mask = torch.zeros_like(ground_truth, dtype=torch.bool)
        for index in range(len(ground_truth_length)):
            mask[index, :ground_truth_length[index]] = 1
        ret = softmax_cross_entropy_with_logits_sparse(prediction, ground_truth, reduction="none", mask=mask)
        return ret.sum(-1).mean()

    def residual_loss(self, hidden_state_decoder: torch.Tensor, ground_truth_length: torch.Tensor):
        """
        input:
            hidden_state_decoder: [batch, max_length+1, decoder_dim]
            ground_truth_length: [batch]
        """
        last_decoded = select_index_by_batch(hidden_state_decoder, ground_truth_length)
        projection = self.readout(self.rnn_cell(last_decoded, None))
        return torch.sum(torch.square(projection), dim=1).mean(), projection

    def accuracy(self, ground_truth: torch.Tensor, ground_truth_length: torch.Tensor, prediction: torch.Tensor) -> float:
        """
        input:
            ground_truth: [batch, max_length]
            ground_truth_length: [batch]
            prediction: [batch, max_length, output_dim]
        """

        with torch.no_grad():
            mask = torch.zeros_like(ground_truth, dtype=torch.bool)
            for index in range(len(ground_truth_length)):
                mask[index, :ground_truth_length[index]] = 1
            prediction_label = prediction.argmax(-1)
            correct = (prediction_label == ground_truth) * mask
            correct_num = correct.sum(-1)
            acc = correct_num / ground_truth_length
            return acc.mean().item()

    @property
    def regulization_parameters(self) -> table:
        ret = table()
        ret.rnn_cell = self.rnn_cell.regulization_parameters
        ret.linear_layer = self.linear_layer * self.linear_layer.gain
        if not self.hyper["encoder_to_decoder_equal_space"]:
            ret.input_to_decoder = self.input_to_decoder * self.input_to_decoder.gain
        return ret

    def inspect(self) -> table:
        ret = self.rnn_cell.inspect()
        rank = min(10, self.hyper.encoder_dim, self.hyper.decoder_dim)
        U, S, V = torch.pca_lowrank(self.input_to_decoder * self.input_to_decoder.gain, q=rank)
        ret.update(table({"W'.lambda[{}]".format(index + 1): S[index].item() for index in range(rank)}))
        ret += self.regulization_parameters.flatten().map(value=lambda x: torch.sum(torch.square(x)).item())
        return ret.map(key=lambda x: "decoder." + x)

    def load_state_dict(self, decoder_state):
        if isinstance(decoder_state, nn.Module):
            decoder_state = decoder_state.state_dict()
        decoder_state = table.hieratical(decoder_state)
        if "input_to_decoder" in decoder_state:
            self.input_to_decoder.data.copy_(decoder_state.input_to_decoder)
        self.linear_layer.data.copy_(decoder_state.linear_layer)
        self.rnn_cell.load_state_dict(decoder_state.rnn_cell)
