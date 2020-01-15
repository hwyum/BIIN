import pandas as pd
from utils import Config

config_path = './data/QQP/config.json'
data_config = Config(config_path)

tr_df = pd.read_csv(data_config.tr_path, sep='\t', usecols=['question1','question2','is_duplicate'])
tr_df_small = tr_df[:len(tr_df)//10]
dev_df = pd.read_csv(data_config.dev_path, sep='\t', usecols=['question1','question2','is_duplicate'])
dev_df_small = dev_df[:len(dev_df)//10]

tr_save_path = './data/QQP/train_small.tsv'
tr_df_small.to_csv(tr_save_path, sep='\t')
dev_save_path = './data/QQP/dev_small.tsv'
dev_df_small.to_csv(dev_save_path, sep='\t')
