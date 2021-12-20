from pyctlib import vector
import torch
import torch.nn as nn
from model import encoder, decoder
from dataset import SimulatedDataset, SimulateEmbedding
from torchfunction.device import todevice
from zytlib import vector
from zytlib.table import table
from zytlib import path
from zytlib.wrapper import registered_property
from torchfunction.device import get_device
from typing import List, Tuple, Optional
from zytlib.wrapper import FunctionTimer

class Trainer:

    def __init__(self, **kwargs):
        self.hyper = table({"datapath": "dataset/dataset_rank_3_num_items_6_overlap_1_repeat_100.db",
                "delta_t": 20,
                "tau": 100,
                "embedding_dim": 1024,
                "decoder_dim": 512,
                "noise_sigma": 0.05,
                "input_dim": 6,
                "batch_size": 32,
                "linear_decoder": True,
                "device": "cuda",
                "embedding": "dataset/embedding_inputdim_6_embeddingdim_1024_round.db",
                "learning_rate": 2e-5,
                "is_embedding_fixed": True,
                "l2_reg": 1e-5,
                "encoder_bias": False,
                "decoder_bias": False,
                "encoder_to_decoder_equal_space": False,
                "encoder_max_rank": -1,
                "decoder_max_rank": -1,
                "freeze_parameter": [],
                "timer_disable": True,
                "clip_grad": 0.1,
                "order_one_init": False,
                "residual_loss": 0,
                "load_model_path": None,
                "train_items_crop": -1,
                "lr_final_decay": 1.0,
                "max_epochs": 500,
                })
        self.hyper.update_exist(kwargs)
        if self.hyper.key_not_here(kwargs):
            print("unknown key:", self.hyper.key_not_here(kwargs))
        self.hyper.lock_key()

        self.device = get_device(self.hyper["device"])
        self.timer = FunctionTimer(disable=self.hyper.timer_disable)
        if isinstance(self.hyper["embedding"], str):
            se = SimulateEmbedding.load(self.hyper["embedding"])
            self.hyper.embedding_dim = se.hyper.embedding_dim
            self.hyper.input_dim = se.hyper.input_dim
            self.hyper["embedding"] = SimulateEmbedding.load(self.hyper["embedding"]).embedding
        self.encoder = encoder(self.hyper["tau"],
                 self.hyper["delta_t"],
                 self.hyper["embedding_dim"],
                 self.hyper["noise_sigma"],
                 self.hyper["input_dim"],
                 embedding = self.hyper["embedding"],
                 is_embedding_fixed = self.hyper["is_embedding_fixed"],
                 encoder_bias = self.hyper["encoder_bias"],
                 encoder_max_rank = self.hyper["encoder_max_rank"],
                 timer = self.timer.timer,
                 order_one_init = self.hyper.order_one_init,
                 ).to(self.device)
        self.encoder.embedding = self.encoder.embedding.to(self.device)
        self.decoder = decoder(self.hyper["decoder_dim"],
                 self.hyper["embedding_dim"],
                 self.hyper["input_dim"],
                 self.hyper["input_dim"],
                 self.encoder.embedding,
                 linear_decoder = self.hyper["linear_decoder"],
                 decoder_bias = self.hyper["decoder_bias"],
                 encoder_to_decoder_equal_space = self.hyper["encoder_to_decoder_equal_space"],
                 decoder_max_rank = self.hyper["decoder_max_rank"],
                 timer = self.timer.timer,
                 order_one_init = self.hyper.order_one_init,
                 ).to(self.device)

        for name, param in self.named_parameters():
            if vector(self.hyper["freeze_parameter"]).any(lambda x: name.startswith(x)):
                param.requires_grad = False

        self.dataset = SimulatedDataset(self.hyper["datapath"], self.hyper["input_dim"], self.hyper["batch_size"], train_items_crop=self.hyper.train_items_crop)

        if self.hyper.load_model_path:
            self.load_state_dict(self.hyper.load_model_path)

    def train_step(self, batch, epoch=-1, index=-1):
        return_info = table()
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        input_encoder, length, ground_truth_tensor, ground_truth_length = todevice(batch, device=self.hyper["device"])
        final_state = self.encoder(input_encoder, length, only_final_state=True)
        decoded_seq, hidden_state_decoder = self.decoder(final_state, ground_truth_tensor, ground_truth_length, teaching_forcing_ratio=0.5)

        loss = self.decoder.loss(ground_truth_tensor, ground_truth_length, decoded_seq)
        return_info.ce_loss = loss.item()
        if self.hyper.residual_loss > 0:
            residual_loss, _ = self.decoder.residual_loss(hidden_state_decoder, ground_truth_length)
            loss += self.hyper.residual_loss * residual_loss
            residual_loss = self.hyper.residual_loss * residual_loss.item()
            return_info.residual_loss = residual_loss
        else:
            residual_loss = 0

        l2_reg = 0
        for reg_param in self.encoder.regulization_parameters.flatten().values():
            l2_reg += torch.sum(torch.square(reg_param))
        for reg_param in self.decoder.regulization_parameters.flatten().values():
            l2_reg += torch.sum(torch.square(reg_param))
        return_info.l2_reg = self.hyper["l2_reg"] * l2_reg.item()

        loss = loss + self.hyper["l2_reg"] * l2_reg
        accuracy = self.decoder.accuracy(ground_truth_tensor, ground_truth_length, decoded_seq)
        loss.backward()

        self.justify_grad()
        self.optimizer.step()

        assert return_info.values().all(lambda x: not isinstance(x, torch.Tensor))
        return loss.item(), accuracy, return_info

    def test_step(self, batch, epoch, index):
        self.encoder.eval()
        self.decoder.eval()
        input_encoder, length, ground_truth_tensor, ground_truth_length, in_train = todevice(batch, device=self.hyper["device"])
        with torch.no_grad():
            # hidden_state, final_state = self.encoder(input_encoder, length)
            final_state = self.encoder(input_encoder, length, only_final_state=True)
            decoded_seq, hidden_state_seq = self.decoder(final_state, torch.zeros_like(ground_truth_tensor).fill_(-1), ground_truth_length, teaching_forcing_ratio=0.0)

            loss = self.decoder.loss(ground_truth_tensor, ground_truth_length, decoded_seq)
            accuracy = self.decoder.accuracy(ground_truth_tensor, ground_truth_length, decoded_seq)
        return loss.item(), accuracy

    @property
    def train_dataloader(self):
        return self.dataset.train_dataloader

    @property
    def test_dataloader(self):
        return self.dataset.test_dataloader

    @registered_property
    def optimizer(self):
        return torch.optim.Adam([{"params": self.encoder.parameters()}, {"params": self.decoder.parameters()}], lr=self.hyper["learning_rate"])

    @registered_property
    def lr_schedular(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if self.hyper.max_epochs > 0 and self.hyper.lr_final_decay != 1.0:
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.hyper.lr_final_decay ** (1 / self.hyper.max_epochs))
        else:
            return None

    def save(self, filepath: path) -> None:
        filepath = path(filepath)
        filepath.parent.mkdir()
        saved = dict()
        saved["hyper"] = self.hyper
        saved["encoder state"] = self.encoder.state_dict()
        saved["decoder state"] = self.decoder.state_dict()
        saved["embedding"] = self.encoder.embedding
        torch.save(saved, filepath)

    @staticmethod
    def load(filepath: str, **kwargs) -> "Trainer":
        saved = torch.load(filepath, map_location=get_device("cuda"))
        hyper = table(saved["hyper"])
        hyper["embedding"] = saved["embedding"]
        hyper.update(kwargs)
        ret = Trainer(**hyper)
        ret.encoder.load_state_dict(saved["encoder state"])
        ret.decoder.load_state_dict(saved["decoder state"])
        return ret

    def load_state_dict(self, filepath: str, encoder_state=None, decoder_state=None):
        if isinstance(filepath, Trainer):
            encoder_loaded = filepath.encoder.state_dict()
            decoder_loaded = filepath.decoder.state_dict()
        else:
            saved = torch.load(filepath, map_location=self.device)
            encoder_loaded = saved["encoder state"]
            decoder_loaded = saved["decoder state"]
        if encoder_state is not None:
            encoder_loaded = {name: param for name, param in encoder_loaded.items() if name in encoder_state}
            decoder_loaded = {name: param for name, param in decoder_loaded.items() if name in decoder_loaded}

        self.encoder.load_state_dict(encoder_loaded)
        self.decoder.load_state_dict(decoder_loaded)

    def train(self, mode=True):
        self.encoder.train(mode)
        self.decoder.train(mode)

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def inspect(self):
        with torch.no_grad():
            ret = self.encoder.inspect() + self.decoder.inspect()
        return ret

    def parameters(self):
        return vector(self.encoder.parameters()) + vector(self.decoder.parameters())

    def named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        return vector(self.encoder.named_parameters()).map(lambda name, x: ("encoder." + name, x)) + vector(self.decoder.named_parameters()).map(lambda name, x: ("decoder." + name, x))

    def check(self):
        batch = next(iter(self.train_dataloader))

        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        input_encoder, length, ground_truth_tensor, ground_truth_length = todevice(batch, device=self.hyper["device"])
        final_state = self.encoder(input_encoder, length, only_final_state=True)
        decoded_seq, hidden_state_decoder = self.decoder(final_state, ground_truth_tensor, ground_truth_length, teaching_forcing_ratio=0.5)

        loss = self.decoder.loss(ground_truth_tensor, ground_truth_length, decoded_seq)
        loss.backward()
        self.justify_grad()

        print("loss", loss.item())
        print("loss_grad", table(self.encoder.named_parameters()).filter(value=lambda x: x.requires_grad).map(value=lambda x: (x.abs().mean().item(), self.hyper.learning_rate * x.grad.abs().mean().item())))
        print("loss_grad", table(self.decoder.named_parameters()).filter(value=lambda x: x.requires_grad).map(value=lambda x: (x.abs().mean().item(), self.hyper.learning_rate * x.grad.abs().mean().item())))

        self.optimizer.zero_grad()
        l2_reg = 0
        for reg_param in self.encoder.regulization_parameters.flatten().values():
            l2_reg += torch.sum(torch.square(reg_param))
        for reg_param in self.decoder.regulization_parameters.flatten().values():
            l2_reg += torch.sum(torch.square(reg_param))

        loss = self.hyper["l2_reg"] * l2_reg
        loss.backward()
        self.justify_grad()

        print("l2 loss", loss.item())
        print("l2 loss_grad", table(self.encoder.named_parameters()).filter(value=lambda x: x.requires_grad).map(value=lambda x: (x.abs().mean().item(), self.hyper.learning_rate * x.grad.abs().mean().item())))
        print("l2 loss_grad", table(self.decoder.named_parameters()).filter(value=lambda x: x.requires_grad).map(value=lambda x: (x.abs().mean().item(), self.hyper.learning_rate * x.grad.abs().mean().item())))

        self.optimizer.zero_grad()

    def justify_grad(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if hasattr(param, "gain"):
                    param.grad.data.mul_(1 / param.gain ** 2)
                if self.hyper.clip_grad > 0:
                    param.grad.data.clip_(-self.hyper.clip_grad, self.hyper.clip_grad)

    def adjust_lr(self, lr_mul=1.0):
        for param in self.optimizer.param_groups:
            param["lr"] = param["lr"] * lr_mul

