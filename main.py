from trainer import Trainer
from zytlib import vector, path
from zytlib import Logger
from zytlib.table import table
from zytlib.wrapper import repeat_trigger
from tqdm import tqdm
import argparse
import sys
from torchfunction.inspect import get_shape
import torch

def main(parser, **kwargs):

    hyper = table({
            "max_epochs": 100,
            "timer_disable": True,
            "l2_reg": 0.01,
            "learning_rate": 1e-3,
            "embedding_size": 4096,
            "datapath": "dataset/dataset_train_rank2.db",
            "embedding": "dataset/embedding_inputdim_6_embeddingdim_4096_round_without_normalize.db"
            })
    hyper.update(kwargs)

    t = Trainer(**hyper)

    if hyper["load_model_path"] != "":
        t.load_state_dict(hyper["load_model_path"])

    t.check()

    logger = Logger(True, True)
    logger.info("running command: python " + " ".join(sys.argv))
    logger.log_parser(parser)
    logger.info(t.hyper, sep="\n")
    logger.info(t.named_parameters().map(lambda name, x: (name, x.shape, x.requires_grad)), sep="\n")

    if hyper["model_name"]:
        saved_path = path("model") / path(hyper["model_name"]).with_ext("pt")
    else:
        saved_path = path("model") / logger.f_name.with_ext("pt")

    checkpoint_dir = (path("model/checkpoint/") / logger.f_name.with_ext("")).mkdir()

    @repeat_trigger(lambda ret: t.save(checkpoint_dir / logger.f_name.with_ext("pt").name_add("_{}".format(ret.ret))), n=hyper["save_model_every_epoch_num"], start=hyper["save_model_every_epoch_num"])
    def loop(epoch):
        train_loss, train_acc = vector(), vector()

        for index, batch in tqdm(enumerate(t.train_dataloader), total=len(t.train_dataloader)):
            loss, acc = t.train_step(batch, epoch, index)
            train_loss.append(loss)
            train_acc.append(acc)

        if isinstance(t.test_dataloader, dict):
            test_loss, test_acc = table(), table()

            for name, dl in t.test_dataloader.items():
                tl, ta = vector(), vector()

                for index, batch in tqdm(enumerate(dl), total=len(dl)):
                    loss, acc = t.test_step(batch, epoch, index)
                    tl.append(loss)
                    ta.append(acc)

                test_loss[name] = tl
                test_acc[name] = ta

            logger.info(f"[{epoch}]/[{hyper['max_epochs']}], train_loss: {train_loss.mean()}, train_acc: {train_acc.mean()}, test_loss: {test_loss.map(value=lambda x: x.mean())}, test_acc: {test_acc.map(value=lambda x: x.mean())}")

            logger.variable("train_loss", train_loss.mean())
            logger.variable("train_acc", train_acc.mean())

            for name, tl in test_loss.items():
                logger.variable("test_loss[{}]".format(name), tl.mean())
            for name, tc in test_acc.items():
                logger.variable("test_acc[{}]".format(name), tc.mean())

        else:
            test_loss, test_acc = vector(), vector()

            for index, batch in tqdm(enumerate(t.test_dataloader), total=len(t.test_dataloader)):
                loss, acc = t.test_step(batch, epoch, index)
                test_loss.append(loss)
                test_acc.append(acc)

            logger.info(f"[{epoch}]/[{hyper['max_epochs']}], train_loss: {train_loss.mean()}, train_acc: {train_acc.mean()}, test_loss: {test_loss.mean()}, test_acc: {test_acc.mean()}")

            logger.variable("train_loss", train_loss.mean())
            logger.variable("train_acc", train_acc.mean())
            logger.variable("test_loss", test_loss.mean())
            logger.variable("test_acc", test_acc.mean())


        for key, value in t.inspect().items():
            logger.variable(key, value)
        return epoch

    for epoch in range(hyper["max_epochs"]):
        try:
            loop(epoch)
        except KeyboardInterrupt:
            select = vector("quit|lr|l2_reg|save|freeze".split("|")).filter(lambda x: len(x)).fuzzy_search()
            if select is None:
                continue
            elif select == "quit":
                return
            elif select == "save":
                confirm = input(f"save model to {saved_path}, [yes|no]:")
                if confirm == "yes":
                    t.save(saved_path)
                quit_or_not = input(f"quit? [yes|no]:")
                if quit_or_not:
                    return
                else:
                    continue
            elif select == "lr":
                print("current learing rate:")
                for param in t.optimizer.param_groups:
                    print(param["lr"])
                lr_mul = float(input("lr *= "))
                t.adjust_lr(lr_mul)
                print("current learing rate:")
                for param in t.optimizer.param_groups:
                    print(param["lr"])
            elif select == "l2_reg":
                print("current l2 reg", t.hyper.l2_reg)
                t.hyper.l2_reg = float(input("l2 reg="))
                print("current l2 reg", t.hyper.l2_reg)
            elif select == "freeze":
                print("current param and requires_grad")
                print(table(t.named_parameters()).map(value=lambda x: x.requires_grad))
                selected = table(t.named_parameters()).keys().append(None).fuzzy_search()
                if selected is None:
                    continue
                else:
                    table(t.named_parameters())[selected].requires_grad = False

        except Exception as e:
            logger.exception(e)
            raise e

    logger.plot_variable_dict(logger.variable_dict, saved_path=path("Log") / path("plot") / path("trainingstep_" + logger.f_name).with_ext("pdf"), hline=["bottom"])
    t.save(saved_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", default=500, type=int)
    parser.add_argument("--embedding_trainable", dest="is_embedding_fixed", action="store_false")
    parser.add_argument("--delta_t", default=20, type=int)
    parser.add_argument("--tau", default=100, type=float)
    parser.add_argument("--embedding_size", default=1024, type=int)
    parser.add_argument("--decoder_dim", default=512, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--noise_sigma", default=0.05, type=float)
    parser.add_argument("--learning_rate", default=1.0, type=float)
    parser.add_argument("--datapath", default="dataset/dataset_rank_3_num_items_6_overlap_1_repeat_100.db", type=str)
    parser.add_argument("--embedding", default="dataset/embedding_inputdim_6_embeddingdim_1024_round.db", type=str)
    parser.add_argument("--non_linear_decoder", dest="linear_decoder", action="store_false")
    parser.add_argument("--l2_reg", default=1e-5, type=float)
    parser.add_argument("--encoder_bias", action="store_true")
    parser.add_argument("--decoder_bias", action="store_true")
    parser.add_argument("--encoder_max_rank", default=-1, type=int)
    parser.add_argument("--decoder_max_rank", default=-1, type=int)
    parser.add_argument("--model_name", default="", type=str)
    parser.add_argument("--load_model_path", default="", type=str)
    parser.add_argument("--freeze_parameter", nargs="+")
    parser.add_argument("--save_model_every_epoch_num", default=100, type=int)
    parser.add_argument("--timer_enable", dest="timer_disable", action="store_false")
    parser.add_argument("--clip_grad", default=0.1, type=float)
    parser.add_argument("--order_one_init", dest="order_one_init", action="store_true")
    hyper, unknown = parser.parse_known_args()
    print("unknown", unknown)
    hyper = table(hyper)
    main(parser, **hyper)
