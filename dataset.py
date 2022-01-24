import torch
import torch.nn as nn
from zytlib import vector
import math
from zytlib.classfunc import save_args
from torch.nn.utils.rnn import pad_sequence
from zytlib.wrapper import second_argument, registered_property
from zytlib.table import table
import math
from torchfunction.utils import seed_torch
from typing import overload

def random_exp(u, num=1):
    assert isinstance(num, int) and num >= 1
    seed = vector.rand(num)
    ret = seed.map(lambda x: - u * math.log(x))
    return ret

def simulate(delta_t, mean_delay, mean_last_delay, mean_rank, rank_candidate, rank_prob=None):

    rank_candidate = vector(rank_candidate)
    rank = rank_candidate.sample(p=rank_prob)

    delay_time = vector.rand(rank, low=mean_delay * 0.5, high=mean_delay * 1.5)
    rank_time = vector.rand(rank, low=mean_rank * 0.5, high=mean_rank * 1.5)
    last_delay_time = vector.rand(1, low=mean_last_delay * 0.5, high=mean_last_delay * 1.5)

    delay_time = delay_time + last_delay_time

    discrete_delay = delay_time.map(lambda x: math.ceil(x / delta_t))
    discrete_rank = rank_time.map(lambda x: math.ceil(x / delta_t))

    return discrete_delay, discrete_rank

class Emulater:

    def __init__(self, delta_t, mean_delay, mean_last_delay, mean_rank):
        save_args(vars())

    def emulate(self, items_list, repeat):
        ret = vector()
        for items in items_list:
            for _ in range(repeat):
                discrete_delay, discrete_rank = simulate(self.hyper['delta_t'], self.hyper['mean_delay'], self.hyper['mean_last_delay'], self.hyper['mean_rank'], vector(len(items)))
                ret.append((items, discrete_delay, discrete_rank))
        return ret

def emulate(delta_t, mean_delay, mean_last_delay, mean_rank, num_items, rank_candidate, repeat=10):
    items_list = vector()
    emulater = Emulater(delta_t, mean_delay, mean_last_delay, mean_rank)
    for rank in rank_candidate:
        items_list = items_list + vector.range(num_items) ** rank
    return emulater.emulate(items_list, repeat)

def train_test_emulate(delta_t, mean_delay, mean_last_delay, mean_rank, num_items, rank_candidate, overlap=1.0, repeat=10, train_prop=0.7):
    items_list = vector()
    emulater = Emulater(delta_t, mean_delay, mean_last_delay, mean_rank)
    for rank in rank_candidate:
        items_list = items_list + vector.range(num_items) ** rank
    only_train, train_test, only_test = items_list.split_random((1 - overlap) * train_prop, overlap, (1 - overlap) * (1 - train_prop))
    only_train_items = emulater.emulate(only_train, repeat)
    only_test_items = emulater.emulate(only_test, repeat).map(lambda x: (*x, 0))
    train_test_items_for_train = emulater.emulate(train_test, int(repeat * train_prop))
    train_test_items_for_test = emulater.emulate(train_test, int(repeat * (1 - train_prop))).map(lambda x: (*x, 1))

    train_items = only_train_items + train_test_items_for_train
    test_items = only_test_items + train_test_items_for_test

    return train_items, test_items

def generate_in_train_label(data: table):
    assert "train" in data and "test" in data
    train_items = set(data["train"].map(lambda x: x[0]))
    train = data["train"]
    test = data["test"]
    if isinstance(test, vector):
        test = test.map(lambda x: (*x, int(x[0] in train_items)))
    elif isinstance(test, dict):
        for key, value in test.items():
            test[key] = value.map(lambda x: (*x, int(x[0] in train_items)))
    return table(train=train, test=test)

def generate_train_test(**kwargs):

    delta_t = kwargs.get("delta_t", 20)
    mean_delay = kwargs.get("mean_delay", 300)
    mean_rank = kwargs.get("mean_rank", 200)
    mean_last_delay = kwargs.get("mean_last_delay", 1000)
    num_items = kwargs.get("num_items", 6)
    random_seed = kwargs.get("random_seed", 1024)

    seed_torch(random_seed)

    emulater = Emulater(delta_t, mean_delay, mean_last_delay, mean_rank)

    train_items, test_items = train_test_emulate(delta_t, mean_delay, mean_last_delay, mean_rank, num_items, [3], overlap=1.0, repeat=100, train_prop=0.7)
    t1 = generate_in_train_label(table(train=train_items, test=test_items))
    t1.save(f"dataset/dataset_rank_3_num_items_{num_items}_overlap_{1}_repeat_100.db")

    train_items = emulate(delta_t, mean_delay, mean_last_delay, mean_rank, num_items, [3], repeat=70)
    test_items_r1 = emulate(delta_t, mean_delay, mean_last_delay, mean_rank, num_items, [1], repeat=100)
    test_items_r2 = emulate(delta_t, mean_delay, mean_last_delay, mean_rank, num_items, [2], repeat=20)
    test_items_r3 = emulate(delta_t, mean_delay, mean_last_delay, mean_rank, num_items, [3], repeat=10)
    t2 = generate_in_train_label(table(train=train_items, test={"r1": test_items_r1, "r2": test_items_r2, "r3": test_items_r3}))
    t2.save(f"dataset/dataset_train_r3_test_r123.db")

    items_list = (vector.range(num_items) ** 3).sample(0.5, replace=False)
    train_items = emulater.emulate(items_list, repeat=140)
    t3 = generate_in_train_label(table(train=train_items, test={"r1": test_items_r1, "r2": test_items_r2, "r3": test_items_r3}))
    t3.save(f"dataset/dataset_train_50_r3_test_r123.db")

class seq_dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, **kwargs):
        self.hyper = table(item_num=-1,
                sequence_length = -1,
                length_crop = -1,
                delta_t = 20,
                interval_range = None,
                delay_range = None,
                rank_range = None,
                unique = False,
                iter_num = 1,
                batch_repeat_num = 1,
                device = None,
                )
        self.hyper.update_exist(kwargs)
        self.property = table()

        assert self.hyper.item_num > 0
        assert self.hyper.sequence_length > 0
        assert self.hyper.delta_t > 0
        if isinstance(self.hyper.interval_range, list):
            assert len(self.hyper.interval_range) == 2
            self.hyper.interval_range = vector(self.hyper.interval_range)
        elif isinstance(self.hyper.interval_range, int):
            self.hyper.interval_range = vector(self.hyper.interval_range, self.hyper.interval_range + 1)
        else:
            raise ValueError()

        if isinstance(self.hyper.rank_range, list):
            assert len(self.hyper.rank_range) == 2
            self.hyper.rank_range = vector(self.hyper.rank_range)
        elif isinstance(self.hyper.rank_range, int):
            self.hyper.rank_range = vector(self.hyper.rank_range, self.hyper.rank_range+1)
        else:
            raise ValueError()

        if isinstance(self.hyper.delay_range, list):
            assert len(self.hyper.delay_range) == 2
            self.hyper.delay_range = vector(self.hyper.delay_range)
        elif isinstance(self.hyper.delay_range, int):
            self.hyper.delay_range = vector(self.hyper.delay_range, self.hyper.delay_range+1)
        else:
            raise ValueError()

        if self.hyper.unique:
            self.property.unique_seq_num = vector.range(self.hyper.sequence_length).map(lambda index: self.hyper.item_num - index).prod()
            self.property.unique_seq = (vector.range(self.hyper.item_num) ** self.hyper.sequence_length).filter(lambda x: len(set(x)) == self.hyper.sequence_length)
        else:
            self.property.unique_seq_num = self.hyper.item_num ** self.hyper.sequence_length
            self.property.unique_seq = vector.range(self.hyper.item_num) ** self.hyper.sequence_length

    def iter(self, **kwargs):
        hyper = table(kwargs)
        hyper.update_where(self.hyper, lambda x: x is None)
        hyper.update_notexist(self.hyper)

        for _ in range(hyper.iter_num):
            rank = vector.range(hyper.rank_range[0], hyper.rank_range[1]).sample(hyper.sequence_length)
            interval = vector.range(hyper.interval_range[0], hyper.interval_range[1]).sample(hyper.sequence_length)
            delay = vector.range(hyper.delay_range[0], hyper.delay_range[1]).sample()
            total_length = rank.sum() + interval.sum() + delay
            if hyper.batch_repeat_num < 1:
                seq_pool = self.property.unique_seq.split_random([1] * int(1/hyper.batch_repeat_num))
            else:
                seq_pool = vector([self.property.unique_seq * hyper.batch_repeat_num])

            for seqs in seq_pool:
                batch_size = len(seqs)
                input_tensor = torch.zeros([batch_size, total_length, self.hyper.item_num])
                if hyper.length_crop == -1:
                    ground_truth_tensor = torch.LongTensor(seqs, device=hyper.device)
                else:
                    ground_truth_tensor = torch.LongTensor(seqs, device=hyper.device)[:, :hyper.length_crop]
                for r in range(hyper.sequence_length):
                    on_set = interval[:(r+1)].sum() + rank[:r].sum(default=0)
                    off_set = interval[:(r+1)].sum() + rank[:(r+1)].sum()
                    for index in vector.range(len(seqs)):
                        input_tensor[index, on_set:off_set, seqs[index][r]] = 1.0
                yield input_tensor, ground_truth_tensor

    def to(self, device):
        self.hyper.device = device
        return self

class dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, data, num_items, items_crop=-1):
        self.data = data
        self.num_items = num_items
        self.items_crop = items_crop

    def __len__(self):
        return len(self.data)

    @property
    def is_train(self):
        if hasattr(self, "_dataset__is_train"):
            return self.__is_train
        self.__is_train = (len(self.data[0]) == 3)
        return self.__is_train

    def __getitem__(self, index):
        sample = self.data[index]
        items = sample[0]
        delay_time = sample[1]
        rank_time = sample[2]
        length = delay_time.sum() + rank_time.sum()
        input_encoder = torch.zeros(length, self.num_items)
        rank = len(rank_time)
        pointer = delay_time[0]
        for i in range(rank):
            input_encoder[pointer:pointer+rank_time[i], items[i]] = 1.0
            pointer += rank_time[i] + delay_time[i]
        ground_truth = torch.LongTensor(items)
        if self.items_crop > 0:
            ground_truth = ground_truth[:self.items_crop]
        if self.is_train:
            return input_encoder, ground_truth
        else:
            return input_encoder, ground_truth, sample[-1]

def data_collect_fn(batch, istrain):
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    length = torch.LongTensor([x[0].shape[0] for x in batch])
    input_encoder_list = [x[0] for x in batch]
    input_encoder = pad_sequence(input_encoder_list, batch_first=True, padding_value=0.0)
    ground_truth_length = torch.LongTensor([len(x[1]) for x in batch])
    ground_truth = pad_sequence([x[1] for x in batch], batch_first=True, padding_value=-1)
    if istrain:
        return input_encoder, length, ground_truth, ground_truth_length
    else:
        return input_encoder, length, ground_truth, ground_truth_length, torch.tensor([x[2] for x in batch])

class SimulateEmbedding:

    @overload
    def __init__(self, embedding_dim=1024, input_dim=6, method="randn", normalized=False, gain_factor=1, noise=0, item_lowrank_dim=2): ...

    def __init__(self, **kwargs):

        self.hyper = table({
            "embedding_dim": 1024,
            "input_dim": 6,
            "method": "randn",
            "normalized": False,
            "gain_factor": 1,
            "noise": 0,
            "item_lowrank_dim": 2,
            })

        self.hyper.lock_key()
        self.hyper.update_exist(kwargs)

    def simulate(self):
        assert self.hyper["method"] in ["randn", "round", "natural"]
        if self.hyper["method"] == "randn":
            embedding = torch.randn(self.hyper["input_dim"], self.hyper["embedding_dim"])
        elif self.hyper["method"] == "round":
            theta = torch.linspace(0, 2 * math.pi, steps=self.hyper["input_dim"] + 1)[:-1]
            x = torch.cos(theta)
            y = torch.sin(theta)
            p = torch.stack([x, y])
            pc = torch.randn(self.hyper["embedding_dim"], 2)
            if self.hyper["normalized"]:
                pc[:, 0] = pc[:, 0] / torch.norm(pc[:, 0], p=2)
            pc[:, 1] = pc[:, 1] - pc[:, 0] *  torch.dot(pc[:, 0], pc[:, 1]) / torch.dot(pc[:, 0], pc[:, 0])
            if self.hyper["normalized"]:
                pc[:, 1] = pc[:, 1] / torch.norm(pc[:, 1], p=2)
            embedding = torch.matmul(pc, p).T
        elif self.hyper["method"] == "natural":
            assert self.hyper.embedding_dim in [1, 2]
            if self.hyper.embedding_dim == 1:
                assert self.hyper["input_dim"] == 2
                embedding = torch.tensor([[1.0], [-1.0]])
            else:
                theta = torch.linspace(0, 2 * math.pi, steps=self.hyper["input_dim"] + 1)[:-1]
                x = torch.cos(theta)
                y = torch.sin(theta)
                embedding = torch.stack([x, y]).T
        embedding = embedding * self.hyper["gain_factor"]
        embedding = embedding + torch.randn_like(embedding) * self.hyper["noise"]
        self.embedding = embedding
        return embedding

    def save(self, filename):
        torch.save({"hyper": dict(self.hyper), "embedding": self.embedding}, filename)

    @staticmethod
    def load(filename):
        content = torch.load(filename)
        ret = SimulateEmbedding(**content["hyper"])
        ret.embedding = content["embedding"]
        return ret

    def __str__(self):
        ret_dict = dict(self.hyper)
        ret_dict["has_embedding"] = hasattr(self, "embedding")
        return str(ret_dict)

    def __repr__(self):
        return str(self)

def generate_embedding():
    # se = SimulateEmbedding(method="round", embedding_dim=1024)
    # se.simulate()
    # se.save("dataset/embedding_inputdim_6_embeddingdim_1024_round.db")
    # se = SimulateEmbedding(embedding_dim=4096, method="round", normalized=False)
    # se.simulate()
    # se.save("dataset/embedding_inputdim_6_embeddingdim_4096_round_without_normalize.db")
    # se.save("dataset/embedding_inputdim_6_embeddingdim_2048_round_without_normalize.db")

    for embedding_dim in [512, 1024, 2048, 4096]:
        se = SimulateEmbedding(embedding_dim=embedding_dim, input_dim=6, method="round", normalized=False, item_lowrank_dim=2)
        se.simulate()
        se.save(f"dataset/embedding_inputdim_2_embeddingdim_{embedding_dim}_round_without_normalize.db")

    for embedding_dim in [512, 1024, 2048, 4096]:
        se = SimulateEmbedding(embedding_dim=embedding_dim, input_dim=2, method="round", normalized=False, item_lowrank_dim=1)
        se.simulate()
        se.save(f"dataset/embedding_inputdim_2_embeddingdim_{embedding_dim}_round_without_normalize.db")

    se = SimulateEmbedding(embedding_dim=1, input_dim=2, method="natural", item_lowrank_dim=1)
    se.simulate()
    se.save("dataset/embedding_inputdim_2_natural.db")

    se = SimulateEmbedding(embedding_dim=2, input_dim=6, method="natural", item_lowrank_dim=2)
    se.simulate()
    se.save("dataset/embedding_inputdim_6_natural.db")

class SimulatedDataset:

    def __init__(self, datapath, batch_size, train_items_crop=-1, test_items_crop=-1):
        save_args(vars())
        content = table.load(datapath)
        self.raw_train, self.raw_test, hyper = content["train"], content["test"], content["hyper"]
        self.hyper.update(hyper)

        self.train_data = dataset(self.raw_train, self.hyper.num_items, train_items_crop)
        if isinstance(self.raw_test, list):
            self.test_data = dataset(self.raw_test, self.hyper.num_items, test_items_crop)
        else:
            assert isinstance(self.raw_test, dict)
            self.raw_test = table(self.raw_test)
            self.test_data = self.raw_test.map(value=lambda x: dataset(x, self.hyper.num_items, test_items_crop))
        if "info" in content:
            self.info = content.info

    @property
    def train_dataloader(self):
        if hasattr(self, "_SimulatedDataset__train_dataloader"):
            return self.__train_dataloader
        self.__train_dataloader = torch.utils.data.dataloader.DataLoader(self.train_data, self.hyper["batch_size"], shuffle=True, collate_fn=second_argument(True, data_collect_fn), num_workers=self.num_workers)
        return self.__train_dataloader

    @property
    def test_dataloader(self):
        if hasattr(self, "_SimulatedDataset__test_dataloader"):
            return self.__test_dataloader
        if isinstance(self.test_data, dataset):
            self.__test_dataloader = torch.utils.data.dataloader.DataLoader(self.test_data, self.hyper["batch_size"], shuffle=False, collate_fn=second_argument(False, data_collect_fn), num_workers=self.num_workers)
        else:
            self.__test_dataloader = self.test_data.map(value=lambda x: torch.utils.data.dataloader.DataLoader(x, self.hyper["batch_size"], shuffle=False, collate_fn=second_argument(False, data_collect_fn), num_workers=self.num_workers))
        return self.__test_dataloader

    @registered_property
    def num_workers(self) -> int:
        if torch.cuda.is_available():
            return 4
        else:
            return 0

    def __str__(self):
        ret = "Dataset: \n"
        ret += vector("\ttrain:", len(self.train_data), self.raw_train.sample(n=1)).join("\t") + "\n"
        if isinstance(self.test_data, dataset):
            ret += vector("\ttest:", len(self.test_data), self.raw_test.sample(n=1)).join("\t") + "\n"
        else:
            ret += "\ttest:\n"
            for name, td in self.raw_test.items():
                ret += "\t" + vector(name, td.length, td.sample(n=1)).join("\t") + "\n"
        return ret

    def __repr__(self):
        return str(self)
