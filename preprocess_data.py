import pandas as pd
import argparse
from pathlib import Path
from model.utils import Vocab, Tokenizer
from utils import Config


TASKS = ["QQP", "MRPC"]
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='directory containing datasets', default='./data')
parser.add_argument('--task', help='tasks to preprocess', default='QQP')

args = parser.parse_args()
data_path = Path(args.data_dir) / args.task
data_config_path = data_path / 'config.json'
data_config = Config(data_config_path)

tr_path = data_config.tr_path
dev_path = data_config.dev_path
tst_path = data_config.tst_path

# NA 제거


