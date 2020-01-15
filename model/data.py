from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import os
import torch
from pathlib import Path
from model.utils import Vocab, Tokenizer, PadSequence
from typing import Callable

class Corpus(Dataset):
    """ general corpus class """
    def __init__(self, filepath:str, transform_fn:Callable, sep=',',
                 doc_col:str='document', label_col:str='label', is_pair:bool=False, doc_col_second:str=None):
        """ Instantiating Corpus class
        Args:
            filepath(str): Data file path.
                           Data file should be in comma separated or tab separated format. (default: comma seperated)
                           Data should have two columns containing document and label.
            vocab: pre-defined vocab which is instance of model.utils.Vocab
            tokenizer: instance of model.utils.Tokenizer
            padder: instance of model.utils.PadSequence
            sep: separator to be used to load data
            doc_col: column name for document or sentence (Default: 'document')
            label_col: column name for label (Default: 'label')
            is_pair: True if inputs are paired sequences
            doc_col_second: (only when is_pair=True) column name for second document or sentence (Default: 'document2')
        """
        self.data = None
        self._transform_fn = transform_fn
        self._doc_col = doc_col
        self._label_col = label_col
        self._is_pair = is_pair

        if self._is_pair:
            assert doc_col_second is not None
            self._doc_col2 = doc_col_second
            self.data = pd.read_csv(filepath, sep=sep, usecols=[self._doc_col, self._doc_col2, self._label_col])
            self.data = self.data[~self.data[self._doc_col].isna()]  # Remove NA
            self.data = self.data[~self.data[self._doc_col2].isna()]  # Remove NA
            self.data = self.data[~self.data[self._label_col].isna()]  # Remove NA
            self.data = self.data[self.data[self._label_col].isin(['0','1'])]

        else:
            self.data = pd.read_csv(filepath, sep=sep, usecols=[self._doc_col, self._label_col])
            self.data = self.data[~self.data[self._doc_col].isna()]  # Remove NA
            self.data = self.data[~self.data[self._label_col].isna()]  # Remove NA
            self.data = self.data[self.data[self._label_col].isin(['0', '1'])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = int(self.data.iloc[idx][self._label_col])

        if self._is_pair:
            preprocessed_inputs = ( self._transform_fn(self.data.iloc[idx][self._doc_col])[0],
                                    self._transform_fn(self.data.iloc[idx][self._doc_col2])[0])
            sample = (torch.tensor(preprocessed_inputs[0]), torch.tensor(preprocessed_inputs[1]), torch.tensor(label))

        else:
            preprocessed_input = self._transform_fn(self.data.iloc[idx][self._doc_col])
            sample = (torch.tensor(preprocessed_input), torch.tensor(label))

        return sample