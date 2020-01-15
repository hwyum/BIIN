import argparse
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
# from konlpy.tag import Mecab
from tqdm import tqdm, tqdm_notebook, trange
from model.data import Corpus
from model.utils import Tokenizer, PadSequence
from utils import Config, SummaryManager, CheckpointManager
from torch.utils.tensorboard import SummaryWriter
from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer
from model.utils import Tokenizer, PreProcessor
from model.network import BIIN
from typing import Callable

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/QQP', help='directory containing data and data configuration in json format')
parser.add_argument('--model_dir', default='./experiments/base_model', help='directory containing model configuration in json format')
parser.add_argument("--type", default="bert-base-uncased", help="pretrained weights of bert")
parser.add_argument("--env", default="consol", help="execution environment", choices=["notebook", "consol"])


def evaluate(model, loss_fn, val_dl, dev, tqdm_func:Callable):
    """ calculate validation loss and accuracy """
    model.eval()
    avg_loss = 0
    correct = 0
    num_yb = 0
    for step, mb in enumerate(tqdm_func(val_dl, desc='Validation')):
        x1, x2, y = map(lambda x: x.to(dev), mb)
        output = model((x1, x2))
        loss = loss_fn(output, y)
        avg_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += torch.sum((y==predicted)).item()
        num_yb += len(y)
    else:
        avg_loss /= (step+1)
        accuracy = correct / num_yb
    return avg_loss, accuracy


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    data_config = Config(data_dir / 'config.json')
    model_config = Config(model_dir / 'config.json')

    # Vocab and Tokenizer
    ptr_dir = Path("pretrained")
    vocab_filepath = ptr_dir / "{}-vocab.pkl".format(args.type)
    with open(vocab_filepath, mode='rb') as io:
        vocab = pickle.load(io)
    ptr_tokenizer = BertTokenizer.from_pretrained(
        args.type, do_lower_case = "uncased" in args.type
    )
    ptr_tokenizer = Tokenizer(vocab, ptr_tokenizer.tokenize)

    preprocessor = PreProcessor(ptr_tokenizer, model_config.max_len)


    # Load Model
    config_filepath = ptr_dir / "{}-config.json".format(args.type)
    config = BertConfig.from_pretrained(config_filepath, output_hidden_states=False)
    model = BIIN(config, vocab, model_config.hidden_size, enc_num_layers=len(model_config.hidden_size))


    # Data Loader
    tr_ds = Corpus(data_config.tr_path, preprocessor.preprocess, sep='\t', doc_col='question1', label_col='is_duplicate',
                        is_pair=True, doc_col_second='question2')
    val_ds = Corpus(data_config.dev_path, preprocessor.preprocess, sep='\t', doc_col='question1', label_col='is_duplicate',
                        is_pair=True, doc_col_second='question2')
    tr_dl = DataLoader(tr_ds,
                       batch_size=model_config.batch_size,
                       shuffle=True,
                       num_workers=4,
                       drop_last=True)
    val_dl = DataLoader(val_ds,
                       batch_size=int(model_config.batch_size / 2),
                       shuffle=False,
                       num_workers=4,
                       drop_last=False)

    # Loss function and Optimization
    loss_func = F.cross_entropy
    opt = optim.Adam(model.parameters())
    epochs = model_config.epochs

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    # Experiments management
    writer = SummaryWriter("{}/runs".format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e10

    # Training
    tqdm_func = tqdm_notebook if args.env == 'notebook' else tqdm

    for epoch in trange(epochs, desc='epochs'):
        model.train()
        avg_tr_loss = 0
        avg_tst_loss = 0
        correct = 0
        num_y = 0
        tr_step = 0
        tst_step = 0

        for x1, x2, y in tqdm_func(tr_dl, desc='iters'):
            x1 = x1.to(dev)
            x2 = x2.to(dev)
            y = y.to(dev)
            scores = model((x1, x2))

            opt.zero_grad()
            tr_loss = loss_func(scores, y)
            tr_loss.backward()  # backprop
            opt.step()  # weight update

            avg_tr_loss += tr_loss.item()
            tr_step += 1

            # accuracy
            _, pred = torch.max(scores, 1)
            correct += torch.sum((pred == y)).item()
            num_y += len(y)

            # tensorboard
            if (epoch * len(tr_dl) + tr_step) % model_config.summary_step == 0:
                dev_loss, _ = evaluate(model, loss_func, val_dl, dev, tqdm_func)
                writer.add_scalars(
                    "loss",
                    {"train": avg_tr_loss / (tr_step + 1), "dev": dev_loss},
                    epoch * len(tr_dl) + tr_step,
                )
                tqdm.write(
                    "global_step: {:3}, tr_loss: {:.3f}, dev_loss: {:.3f}".format(
                        epoch * len(tr_dl) + tr_step, avg_tr_loss / (tr_step + 1), dev_loss
                    )
                )

            model.train()

        else:
            avg_tr_loss /= tr_step
            tr_acc = correct / num_y

            # evaluation
            # del x1, x2, y # to free up memory
            avg_dev_loss, dev_acc = evaluate(model, loss_func, val_dl, dev, tqdm_func)

            # Summary
            tr_summary = {"loss": avg_tr_loss, "acc": tr_acc}
            val_summary = {"loss": avg_dev_loss, "acc": dev_acc}
            tqdm.write('epoch : {}, tr_loss : {:.3f}, tst_loss : {:.3f}, tr_acc: {:.2f}, tst_acc : {:.2f}'
                       .format(epoch + 1, tr_summary["loss"], val_summary["loss"], tr_summary["acc"],
                               val_summary["acc"]))

            is_best = avg_dev_loss < best_val_loss

            if is_best:
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': opt.state_dict(),
                    'vocab': vocab
                }
                summary = {"train": tr_summary, "validation": val_summary}

                summary_manager.update(summary)
                summary_manager.save("summary.json")
                checkpoint_manager.save_checkpoint(state, "best.tar")

                best_val_loss = avg_dev_loss

