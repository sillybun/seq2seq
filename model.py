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

def low_rank(self, n, r, name="J", init_method="randn", implicit_gain=False, init_gain=1.0, device=None):
    assert init_method in ["randn", "order_one", "orthogonal"]

    self.__setattr__("_" + name + "_r", r)
    if r == -1:
        if init_method == "randn":
            if implicit_gain:
                self.__setattr__("_" + name, nn.Parameter(torch.randn(n, n, device=device) / math.sqrt(n) * init_gain))
            else:
                self.__setattr__("_" + name, nn.Parameter(torch.randn(n, n, device=device) * math.sqrt(n) * init_gain))
        elif init_method == "order_one":
            if implicit_gain:
                self.__setattr__("_" + name, nn.Parameter(torch.randn(n, n, device=device) / n * init_gain))
            else:
                self.__setattr__("_" + name, nn.Parameter(torch.randn(n, n, device=device) * init_gain))
        else:
            if implicit_gain:
                self.__setattr__("_" + name, nn.Parameter(nn.init.orthogonal_(torch.empty(n, n, device=device), gain=init_gain)))
            else:
                self.__setattr__("_" + name, nn.Parameter(nn.init.orthogonal_(torch.empty(n, n, device=device), gain=init_gain * n)))
    else:
        if implicit_gain:
            self.__setattr__("_" + name + "_V", nn.Parameter(torch.randn(n, r, device=device) / math.sqrt(n) * init_gain))
            self.__setattr__("_" + name + "_U", nn.Parameter(torch.randn(n, r, device=device) / math.sqrt(n) * init_gain))
        else:
            self.__setattr__("_" + name + "_V", nn.Parameter(torch.randn(n, r, device=device) * init_gain))
            self.__setattr__("_" + name + "_U", nn.Parameter(torch.randn(n, r, device=device) * init_gain))
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
        if len(state) == 0:
            return
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

class low_rank_subpopulation_loading_vector(nn.Module):

    def __init__(self, n, vec_num, sub_num, p=None, init_sigma2=1.0, init_covar=0.0, init_mean=0.0, zero_mean=False, determined_partition=False, device=None):

        super().__init__()
        save_args(vars(), ignore=("device", ))

        assert init_sigma2 > 0
        assert abs(init_covar) < init_sigma2

        init_cov_matrix = torch.zeros(vec_num, vec_num, device=device) + init_covar +  torch.eye(vec_num, vec_num, device=device) * (init_sigma2 - init_covar)
        U, S = torch.linalg.eigh(init_cov_matrix)
        L = S @ torch.diag(U ** 0.5)

        self.COV_MATRIX_L = nn.ParameterList([nn.Parameter(L.clone() @ nn.init.orthogonal_(torch.empty_like(L, device=device))) for _ in range(sub_num)])

        if not zero_mean:
            self.MEAN_MATRIX = nn.ParameterList([nn.Parameter(torch.zeros(vec_num, 1, device=device) + init_mean) for _ in range(sub_num)])
        else:
            assert init_mean == 0

        if p is None:
            self.P = vector.constant_vector(1 / sub_num, sub_num)
        else:
            self.P = vector(p)

        self.__freeze = False
        self.sample()

    def sample(self):

        if self.__freeze:
            if hasattr(self, "random_gaussian"):
                return
        if self.hyper.determined_partition:
            self.subp_neuron_num: vector = vector.P.map(lambda p: int(self.hyper.n * p))
            _, min_index = self.subp_neuron_num.min(with_index=True)
            self.subp_neuron_num[min_index] += self.hyper.n - self.subp_neuron_num.sum()
        else:
            self.subp_neuron_num = vector.multinomial(self.hyper.n, self.P)
        self.random_gaussian = [torch.randn(self.hyper.vec_num, self.subp_neuron_num[index], device=(device:= self.COV_MATRIX_L[index].device)) for index in range(self.hyper.sub_num)]
        self.perm_matrix = torch.randperm(self.hyper.n, device=device)
        self.lv_index = vector.range(self.hyper.sub_num).map(lambda index: vector.constant_vector(index, self.subp_neuron_num[index])).flatten()
        self.lv_index = self.lv_index[self.perm_matrix.detach().cpu().numpy().tolist()]
        self.__SubpIndex = table(index=self.lv_index)
        for p_i in range(self.hyper.sub_num):
            self.__SubpIndex[p_i] = self.lv_index.findall(p_i)

        return self

    def clear_sample(self):
        delattr(self, "random_gaussian")
        delattr(self, "perm_matrix")
        delattr(self, "subp_neuron_num")
        delattr(self, "__SubpIndex")

    def LoadingVectors(self):
        self.random_gaussian = [rg.to(self.COV_MATRIX_L[0].device) for rg in self.random_gaussian]
        self.perm_matrix = self.perm_matrix.to(self.COV_MATRIX_L[0].device)

        if self.hyper.zero_mean:
            lv = [(self.COV_MATRIX_L[index] @ self.random_gaussian[index]).T for index in range(self.hyper.sub_num)]
        else:
            lv = [(self.COV_MATRIX_L[index] @ self.random_gaussian[index] + self.MEAN_MATRIX[index]).T for index in range(self.hyper.sub_num)]

        lv = torch.cat(lv, 0)
        lv = lv[self.perm_matrix, :]
        self.loadingvectors = lv
        return self.loadingvectors

    def SubpIndex(self, *args):
        if len(args) == 0:
            return self.__SubpIndex.index
        if len(args) == 1:
            return self.__SubpIndex[int(args[0])]

    def KL_Divergence(self):
        ret = 0
        for index in range(self.hyper.sub_num):
            cov_m_l = self.COV_MATRIX_L[index]
            cov_m = torch.matmul(cov_m_l, cov_m_l.T)
            ret += 1 / 2 * (-torch.log(torch.det(cov_m)) - self.hyper.vec_num + torch.trace(cov_m))
            if not self.hyper.zero_mean:
                mean_m = self.MEAN_MATRIX[index]
                ret += 1 / 2 * torch.sum(torch.square(mean_m))
        return ret

    def l1_Sparsity(self):
        ret = 0
        for index in range(self.hyper.sub_num):
            cov_m_l = self.COV_MATRIX_L[index]
            cov_m = torch.matmul(cov_m_l, cov_m_l.T)
            ret += (cov_m * (1 - torch.eye(self.hyper.vec_num, self.hyper.vec_num, device=cov_m.device))).abs().sum()
        return ret

    def inspect(self):
        ret = table()
        for index in range(self.hyper.sub_num):
            cov_m_l = self.COV_MATRIX_L[index]
            cov_m = torch.matmul(cov_m_l, cov_m_l.T)
            for i in range(self.hyper.vec_num):
                ret[f"ssigma2[{index},{i}]"] = cov_m[i, i].item()
            for i in range(self.hyper.vec_num):
                for j in range(i + 1, self.hyper.vec_num):
                    ret[f"scov[{index},{i},{j}]"] = cov_m[i, j].item()
        return ret

    def freeze(self, _f = True):
        self.__freeze = _f

    def unfreeze(self):
        self.__freeze = False

class empty_encoder_rnn(nn.Module):

    def __init__(self, tau, delta_t, embedding_dim, noise_sigma):
        super().__init__()
        save_args(vars())
        self.alpha = self.hyper["delta_t"] / self.hyper["tau"]
        assert 0 < self.alpha < 1

    def forward(self, U, V, x: torch.Tensor, I: torch.Tensor=None, noise_sigma=None) -> torch.Tensor:
        if I is None:
            I = torch.zeros_like(x)

        if noise_sigma is not None:
            noise = torch.randn_like(I) * noise_sigma
        elif self.training:
            noise = torch.randn_like(I) * self.hyper["noise_sigma"]
        else:
            noise = torch.zeros_like(I)

        x = (1 - self.alpha) * x + self.alpha * (torch.matmul(torch.matmul(torch.tanh(x), V), U.T) / self.hyper.embedding_dim + I + noise)
        # x = (1 - self.alpha) * x + self.alpha * (self.J_rmul(torch.tanh(x)) * self.gain + I + noise)
        return x

    @property
    def regulization_parameters(self) -> table:
        return table()

    def inspect(self) -> table:
        return table()

    def load_state_dict(self, encoder_state: OrderedDict):
        return

    def __repr__(self) -> str:
        return f"empty_encoder_rnn(n={self.hyper.embedding_dim}, alpha={self.alpha})"

class encoder_rnn(nn.Module):

    def __init__(self, tau, delta_t, embedding_dim, noise_sigma, gain=1.0, p=1.0, encoder_bias=False, encoder_max_rank=-1, init_method="randn", init_gain=1.0, device=None):
        super().__init__()
        save_args(vars(), ignore=("device",))
        self.init(device=device)

    def init(self, device=None):
        self.load_states = vector()
        lsd = low_rank(self, self.hyper.embedding_dim, self.hyper.encoder_max_rank, name="J", init_method=self.hyper.init_method, init_gain=self.hyper.init_gain, device=device)
        self.load_states.append(lsd)
        if self.hyper.encoder_max_rank == -1:
            self._J.gain = self.gain
        else:
            self._J_U.gain = self._J_V.gain = math.sqrt(self.gain)

        self.alpha = self.hyper["delta_t"] / self.hyper["tau"]
        if self.hyper['encoder_bias']:
            self.bias = nn.Parameter(torch.zeros(1, self.hyper['embedding_dim'], device=device))
            self.load_states.append(lambda state: self.bias.data.copy_(state["bias"]))
        assert 0 < self.alpha < 1

    def forward(self, x: torch.Tensor, I: torch.Tensor=None, noise_sigma=None) -> torch.Tensor:
        if I is None:
            I = torch.zeros_like(x)

        if self.training:
            noise = torch.randn_like(I) * self.hyper["noise_sigma"]
        elif noise_sigma is not None:
            noise = torch.randn_like(I) * noise_sigma
        else:
            noise = torch.zeros_like(I)

        if self.hyper['encoder_bias']:
            I = I + self.bias

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
        U, S, V = torch.pca_lowrank(self.J * self.gain, q=rank, center=False)
        return table({"lambda[{}]".format(index + 1): S[index].item() for index in range(rank)})

    def load_state_dict(self, encoder_state: OrderedDict):
        if len(encoder_state) == 0:
            return
        self.load_states.apply(lambda x: x(encoder_state))

    @registered_property
    def gain(self) -> float:
        return self.hyper.gain / self.hyper.embedding_dim ** self.hyper.p

    def __repr__(self) -> str:
        return f"encoder_rnn(n={self.hyper.embedding_dim}, alpha={self.alpha})"

class linear_decoder_rnn(nn.Module):

    def __init__(self, decoder_dim, decoder_bias=False, decoder_max_rank=-1, init_method="randn", init_gain=1.0, device=None):

        super().__init__()
        save_args(vars())
        self.init(device=device)

    def init(self, device=None):
        self.load_states = vector()
        lsd = low_rank(self, self.hyper.decoder_dim, self.hyper.decoder_max_rank, name="W", init_method=self.hyper.init_method, init_gain=self.hyper.init_gain, device=device)
        self.load_states.append(lsd)
        if self.hyper["decoder_max_rank"] == -1:
            self._W.gain = self.gain
        else:
            self._W_U.gain = self._W_V.gain = math.sqrt(self.gain)
        if self.hyper["decoder_bias"]:
            self.bias = nn.Parameter(torch.zeros(1, self.hyper["decoder_dim"], device=device))
            self.load_states.append(lambda state: self.b.data.copy_(state["bias"]))

    def forward(self, h, I=None):
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
        U, S, V = torch.pca_lowrank(self.W * self.gain, q=rank, center=False)
        return table({"lambda[{}]".format(index + 1): S[index].item() for index in range(rank)})

    def load_state_dict(self, decoder_state: OrderedDict) -> None:
        if len(decoder_state) == 0:
            return
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

class low_rank_subpopulation_encoder(nn.Module):

    def __init__(self, tau, delta_t, embedding_dim, noise_sigma, input_dim, encoder_rank, readout_rank, encoder_subp_num, init_sigma2=1.0, init_covar=0.0, init_mean=0.0, input_gain=1.0, embedding=None, naive_loadingvectors=False, zero_mean=False, perfect_readout=False, device=None):
        super().__init__()
        save_args(vars(), ignore=("device", ))

        if perfect_readout:
            self.hyper.vec_num = input_dim + 2 * encoder_rank
        else:
            self.hyper.vec_num = input_dim + 2 * encoder_rank + readout_rank

        if naive_loadingvectors:
            self.LoadingVectors = nn.Parameter(torch.randn(embedding_dim, self.hyper.vec_num, device=device))
        else:
            self.LoadingVectors = low_rank_subpopulation_loading_vector(embedding_dim, self.hyper.vec_num, encoder_subp_num, p=None, init_mean=init_mean, init_covar=init_covar, init_sigma2=init_sigma2, zero_mean=zero_mean, device=device)
        self.rnn = empty_encoder_rnn(tau, delta_t, embedding_dim, noise_sigma)
        if embedding is not None:
            self.embedding = embedding

    def forward(self, u, length, noise_sigma=None, only_final_state=False, resample_loadingvectors=True):
        """
        u.shape: [batch, time, input_dim]
        length.shape: [batch]

        output:
        ret.shape: [batch, time(+1), embedding_dim]
        """
        if self.hyper.naive_loadingvectors:
            loading_vec = self.LoadingVectors
        else:
            if resample_loadingvectors:
                self.LoadingVectors.sample()
            loading_vec = self.LoadingVectors.LoadingVectors()

        p = 0
        embedding = loading_vec[:, :(p:=p + self.hyper.input_dim)]
        U = loading_vec[:, p:(p:=p + self.hyper.encoder_rank)]
        V = loading_vec[:, p:(p:=p + self.hyper.encoder_rank)]
        if not self.hyper.perfect_readout:
            W = loading_vec[:, p:(p:=p + self.hyper.readout_rank)]
        else:
            W = U[:, :self.hyper.readout_rank]
        assert p == loading_vec.shape[1]

        batch_size = u.shape[0]

        sorted_length, length_index = torch.sort(length, dim=-1, descending=True)
        u = u[length_index, :, :]

        x = torch.zeros(sorted_length[0].item() + 1, batch_size, self.hyper['embedding_dim'], device=u.device)

        for index in range(sorted_length[0].item()):

            batch = sum(sorted_length > index).item()
            if batch == 0:
                break
            x[index + 1, :batch, :] = self.rnn(U, V, x[index, :batch, :], torch.matmul(u[:batch, index, :] @ self.embedding, embedding.T) * self.hyper.input_gain, noise_sigma=noise_sigma)

        x = x.transpose(0, 1)

        ret = torch.zeros_like(x)
        ret[length_index, :, :] = x

        last_time_state = select_index_by_batch(ret, length)
        final_state = torch.matmul(torch.tanh(last_time_state), W) / self.hyper.embedding_dim

        if only_final_state:
            return final_state
        else:
            return ret, final_state

    @property
    def regulization_parameters(self) -> table:
        if self.hyper.naive_loadingvectors:
            ret = table(loading_vectors=self.LoadingVectors)
            if self.hyper.zero_mean:
                ret.loading_vectors_mean = self.LoadingVectors.mean(0)
        else:
            ret = table()
            for index in range(self.hyper.encoder_subp_num):
                ret["COV{}".format(index)] = self.LoadingVectors.COV_MATRIX_L[index]
        return ret

    def KL_Divergence(self):
        return self.LoadingVectors().KL_Divergence()

    def inspect(self) -> table:
        if self.hyper.naive_loadingvectors:
            return table()
        ret = self.LoadingVectors.inspect()
        return ret.map(key=lambda x: "encoder." + x)

    # def load_state_dict(self, encoder_state):
    #     if encoder_state is None:
    #         return
    #     if isinstance(encoder_state, dict) and len(encoder_state) == 0:
    #         return
    #     if isinstance(encoder_state, nn.Module):
    #         encoder_state = encoder_state.state_dict()
    #     encoder_state = table.hieratical(encoder_state)
    #     if "embedding" in encoder_state and isinstance(self.embedding, nn.Parameter):
    #         self.embedding.data.copy_(encoder_state["embedding"])
    #     if "rnn" in encoder_state:
    #         self.rnn.load_state_dict(encoder_state.rnn)

class encoder(nn.Module):

    def __init__(self, tau, delta_t, encoder_dim, noise_sigma, input_dim, input_gain=1.0, embedding=None, is_embedding_fixed=True, encoder_bias=False, encoder_max_rank=-1, timer=None, init_method="randn", init_gain=1.0, convert_to_hidden_space=False, readout_rank=None, device=None):

        super().__init__()
        save_args(vars(), ignore=("timer", "device",))

        self.rnn = encoder_rnn(tau, delta_t, encoder_dim, noise_sigma, encoder_bias=encoder_bias, encoder_max_rank=encoder_max_rank, init_method=init_method, init_gain=init_gain, device=device)
        if embedding is None:
            embedding = torch.randn(input_dim, encoder_dim, device=device)
        else:
            embedding = embedding.to(device)
        if is_embedding_fixed:
            self.embedding = embedding
        else:
            self.embedding = nn.Parameter(embedding)

        if convert_to_hidden_space:
            self.readout_matrix = nn.Parameter(torch.randn(encoder_dim, readout_rank, device=device))
            if init_method == "randn":
                self.readout_matrix.data.mul_(self.hyper.encoder_dim ** 0.5)
            elif init_method == "orthogonal":
                self.readout_matrix.data.mul_(self.hyper.encoder_dim ** 0.5)
            self.readout_matrix.gain = 1 / self.hyper.encoder_dim ** 0.5

        if timer is not None:
            self.forward = timer(self.forward)

    def forward(self, u, length=None, noise_sigma=None, only_final_state=False):
        """
        u.shape: [batch, time, input_dim]
        length.shape: [batch]

        output:
        ret.shape: [batch, time(+1), encoder_dim]
        """

        batch_size = u.shape[0]

        if length is not None:
            sorted_length, length_index = torch.sort(length, dim=-1, descending=True)
            u = u[length_index, :, :]
            x = torch.zeros(sorted_length[0].item() + 1, batch_size, self.hyper['encoder_dim'], device=u.device)
            max_length = sorted_length[0].item()
        else:
            max_length = u.shape[1]
            x = torch.zeros(max_length+1, batch_size, self.hyper[""])

        batch = u.shape[1]

        for index in range(max_length):

            if length is not None:
                batch = sum(sorted_length > index).item()
            if batch == 0:
                break
            x[index + 1, :batch, :] = self.rnn(x[index, :batch, :], torch.matmul(u[:batch, index, :], self.embedding) * self.hyper.input_gain, noise_sigma=noise_sigma)

        x = x.transpose(0, 1)

        ret = torch.zeros_like(x)
        if length is not None:
            ret[length_index, :, :] = x

        final_state = torch.gather(ret, 1, length.view(batch_size, 1, 1).expand(batch_size, 1, self.hyper['encoder_dim'])).squeeze(1)

        if self.hyper.convert_to_hidden_space:
            final_state = torch.matmul(torch.tanh(final_state), self.readout_matrix) * self.readout_matrix.gain

        if only_final_state:
            return final_state
        else:
            return ret, final_state

    @property
    def regulization_parameters(self) -> table:
        ret = table(rnn=self.rnn.regulization_parameters)
        if self.hyper.convert_to_hidden_space:
            ret.readout_matrix = self.readout_matrix * self.readout_matrix.gain
        return ret

    def inspect(self) -> table:
        ret = self.rnn.inspect()
        ret += self.regulization_parameters.flatten().map(value=lambda x: torch.sum(torch.square(x)).item())
        return ret.map(key=lambda x: "encoder." + x)

    def load_state_dict(self, encoder_state):
        if encoder_state is None:
            return
        if isinstance(encoder_state, dict) and len(encoder_state) == 0:
            return
        if isinstance(encoder_state, nn.Module):
            encoder_state = encoder_state.state_dict()
        encoder_state = table.hieratical(encoder_state)
        if "embedding" in encoder_state and isinstance(self.embedding, nn.Parameter):
            self.embedding.data.copy_(encoder_state["embedding"])
        if "readout_matrix" in encoder_state:
            self.readout_matrix.data.copy_(encoder_state["readout_matrix"])
        if "rnn" in encoder_state:
            self.rnn.load_state_dict(encoder_state.rnn)

class decoder(nn.Module):

    def __init__(self, decoder_dim, input_dim, output_dim, encoder_dim=-1, item_embedding_dim=1, linear_decoder=True, decoder_bias=None, encoder_to_decoder_equal_space=False, decoder_max_rank=-1, timer=None, init_method="init_method", init_gain=1.0, encoder_convert_to_hidden_space=False, perfect_decoder=False, device=None):

        super().__init__()
        save_args(vars(), ignore=("timer", "device", ))
        if linear_decoder:
            if decoder_bias is None:
                decoder_bias = False
            self.rnn_cell = linear_decoder_rnn(decoder_dim, decoder_bias=decoder_bias, decoder_max_rank=decoder_max_rank, init_method=init_method, init_gain=init_gain, device=device)
        else:
            raise NotImplementedError()

        self.linear_layer = nn.Parameter(torch.zeros(decoder_dim, output_dim, device=device))

        if not encoder_to_decoder_equal_space:
            self.input_to_decoder = nn.Parameter(torch.zeros(input_dim, decoder_dim, device=device))

        self.init(device=device)
        if timer is not None:
            self.forward = timer(self.forward)

    def init(self, device=None) -> None:
        if self.hyper.perfect_decoder:
            assert self.hyper.encoder_convert_to_hidden_space
            assert self.hyper.decoder_max_rank > 0
            assert self.hyper.decoder_max_rank == self.hyper.input_dim
            assert self.hyper.decoder_max_rank % self.hyper.item_embedding_dim == 0
            L = self.hyper.decoder_max_rank // self.hyper.item_embedding_dim
            r = self.hyper.item_embedding_dim
            with torch.no_grad():
                M = torch.randn(self.hyper.decoder_dim, (L + 1) * r, device=device)
                M = torch.pca_lowrank(M, q=(L + 1) * r, center=False)[0] * (self.hyper.decoder_dim) ** 0.5
                U = M[:, :(L * r)]
                V = M[:, r:]
                R = M[:, :r]
                C = M[:, r:]
                if self.hyper.item_embedding_dim == 1:
                    assert self.hyper.output_dim == 2
                    R_CONVERT = torch.tensor([[1.0, -1.0]], device=device)
                else:
                    assert self.hyper.item_embedding_dim == 2
                    theta = torch.linspace(0, 2 * math.pi, steps=self.hyper.output_dim + 1, device=device)[:-1]
                    x = torch.cos(theta)
                    y = torch.sin(theta)
                    R_CONVERT = torch.stack([x, y])
                R = R @ R_CONVERT
                self.linear_layer.data.copy_(R)
                self.rnn_cell._W_U.data.copy_(V)
                self.rnn_cell._W_V.data.copy_(U)
                self.input_to_decoder.data.copy_(C.T)
                self.input_to_decoder.gain = 1 / self.hyper.encoder_dim ** 0.5
                self.linear_layer.gain = 1 / self.hyper.decoder_dim
        else:
            with torch.no_grad():
                nn.init.normal_(self.linear_layer)
                if self.hyper.init_method == "randn":
                    self.linear_layer.data.mul_(math.sqrt(self.hyper.decoder_dim))
                self.linear_layer.gain = 1 / self.hyper.decoder_dim

                if self.hyper.encoder_convert_to_hidden_space:
                    nn.init.normal_(self.input_to_decoder)
                    self.input_to_decoder.gain = 1 / self.hyper.encoder_dim ** 0.5
                elif not self.hyper["encoder_to_decoder_equal_space"]:
                    nn.init.normal_(self.input_to_decoder)
                    if self.hyper.init_method == "randn":
                        self.input_to_decoder.data.mul_(math.sqrt(self.hyper.input_dim))
                    self.input_to_decoder.gain = 1 / self.hyper.input_dim

    def readout(self, hidden_state):
        return hidden_state @ self.linear_layer * self.linear_layer.gain

    def forward(self, hidden_state, ground_truth_length):
        """
        input:
            hidden_state: [batch, embedding_dim]
            # ground_truth_tensor: [batch, max_length]
            ground_truth_length: [batch]

        output:
            ret: [batch, max_length, embedding_dim]
        """
        if isinstance(ground_truth_length, int):
            ground_truth_length = torch.LongTensor([ground_truth_length] * hidden_state.shape[0], device=hidden_state.device)

        ground_truth_length, sorted_index = ground_truth_length.sort(dim=-1, descending=True)
        if self.hyper.encoder_convert_to_hidden_space:
            hidden_state = hidden_state[sorted_index, :] @ self.input_to_decoder
        elif self.hyper["encoder_to_decoder_equal_space"]:
            hidden_state = torch.tanh(hidden_state[sorted_index, :])
        else:
            hidden_state = torch.tanh(hidden_state[sorted_index, :]) @ self.input_to_decoder * self.input_to_decoder.gain

        batch_size = hidden_state.shape[0]
        max_length = ground_truth_length[0].item()

        outputs = torch.zeros([batch_size, max_length, self.hyper["output_dim"]], device=hidden_state.device)

        decoder_hidden_state = torch.zeros([batch_size, max_length + 1, self.hyper["decoder_dim"]], device=hidden_state.device)
        decoder_hidden_state[:, 0, :] = hidden_state

        for index in range(max_length):
            batch = sum(index < ground_truth_length)
            if batch == 0:
                break

            hidden_state = self.rnn_cell(hidden_state[:batch, :])

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

    def accuracy(self, ground_truth: torch.Tensor, ground_truth_length: torch.Tensor, prediction: torch.Tensor, by_rank=False) -> float:
        """
        input:
            ground_truth: [batch, max_length]
            ground_truth_length: [batch]
            prediction: [batch, max_length, output_dim]
        """

        if not by_rank:
            with torch.no_grad():
                mask = torch.zeros_like(ground_truth, dtype=torch.bool)
                for index in range(len(ground_truth_length)):
                    mask[index, :ground_truth_length[index]] = 1
                prediction_label = prediction.argmax(-1)
                correct = (prediction_label == ground_truth) * mask
                correct_num = correct.sum(-1)
                acc = correct_num / ground_truth_length
                return acc.mean().item()
        else:
            with torch.no_grad():
                gtl = ground_truth_length[0].item()
                prediction_label = prediction.argmax(-1)
                correct = (prediction_label == ground_truth)
                return correct.float().mean(0)[:gtl].detach().cpu().numpy()

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
        rank = min(10, self.hyper.input_dim, self.hyper.decoder_dim)
        if not self.hyper.encoder_convert_to_hidden_space:
            U, S, V = torch.pca_lowrank(self.input_to_decoder * self.input_to_decoder.gain, q=rank, center=False)
            ret.update(table({"W'.lambda[{}]".format(index + 1): S[index].item() for index in range(rank)}))
        ret += self.regulization_parameters.flatten().map(value=lambda x: torch.sum(torch.square(x)).item())
        return ret.map(key=lambda x: "decoder." + x)

    def load_state_dict(self, decoder_state):
        if decoder_state is None:
            return
        if isinstance(decoder_state, dict) and len(decoder_state) == 0:
            return
        if isinstance(decoder_state, nn.Module):
            decoder_state = decoder_state.state_dict()
        decoder_state = table.hieratical(decoder_state)
        if "input_to_decoder" in decoder_state:
            self.input_to_decoder.data.copy_(decoder_state.input_to_decoder)
        if "linear_layer" in decoder_state:
            self.linear_layer.data.copy_(decoder_state.linear_layer)
        if "rnn_cell" in decoder_state:
            self.rnn_cell.load_state_dict(decoder_state.rnn_cell)
