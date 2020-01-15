import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from model.utils import Vocab
from typing import Union
from model.densenet import DenseNet

class BertFeature(BertPreTrainedModel):
    def __init__(self, config, vocab: Vocab) -> None:
        super(BertFeature, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vocab = vocab
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None):
        attention_mask = input_ids.ne(
            self.vocab.to_indices(self.vocab._padding_token)
        ).float()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = self.dropout(outputs[1])
        sequence_output = self.dropout(outputs[0])

        return pooled_output, sequence_output


class Encoding(nn.Module):
    def __init__(self, input_size, hidden_size, type="GRU"):
        super(Encoding, self).__init__()
        assert type in ["GRU", "LSTM"]
        self._encoder = None
        if self.type == "GRU":
            self._encoder = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        else:
            self._encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)

    def forward(self, input):
        encoded_output = self._encoder(input)
        encoded_output = encoded_output[0]

        return encoded_output


class Interaction(nn.Module):
    def __init__(self):
        super(Interaction, self).__init__()

    def forward(self, encoded_1, encoded_2, mask=None):
        encoded_1 = encoded_1.unsqueeze(2)
        encoded_2 = encoded_2.unsqueeze(1)
        interaction_output = encoded_1 * encoded_2 # (batch, seq_1, seq_2, hidden)
        # if mask:

        return interaction_output


class FeatureExtractor(nn.Module):
    def __init__(self, input_channel):
        super(FeatureExtractor, self).__init__()
        self._densenet121 = DenseNet(input_channel=input_channel, num_classes=2)

    def forward(self, interaction_output):
        interaction_output = interaction_output.permute(0, 3, 1, 2) # to feed into conv as (N, C_{in}, H_{in}, W_{in})
        output = self._densenet121(interaction_output)

        return output
