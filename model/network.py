import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from model.utils import Vocab
from model.modules import BertFeature, Encoding, Interaction, FeatureExtractor

class BIIN(nn.Module):
    def __init__(self, config, vocab, hidden_size, enc_num_layers=1, ):
        super(BIIN, self).__init__()
        self._input = BertFeature(config, vocab)
        self._dropout = nn.Dropout(p=0.15)
        self._enc_num_layers = enc_num_layers

        assert len(hidden_size) == enc_num_layers
        if isinstance(hidden_size, list):
            input_dim = [768] + [hidden_size[0] * 2, hidden_size[1] * 2] # encoder is bidirectionL
            self._encoder = ModuleList([Encoding(i, h) for i, h in zip(input_dim, hidden_size)])
            self._extractor = FeatureExtractor(hidden_size[2] * 2) # encoder is bidirectionL
        else:
            self._encoder = Encoding(768, hidden_size)
            self._extractor = FeatureExtractor(hidden_size * 2) # encoder is bidirectionL

        self._interaction = Interaction()


    def forward(self, inputs):
        q_1, q_2 = inputs
        bert_1 = self._dropout(self._input(q_1)[1])
        bert_2 = self._dropout(self._input(q_2)[1])

        if self._enc_num_layers == 1:
            encoded_1 = self._encoder(bert_1)
            encoded_2 = self._encoder(bert_2)
        else:
            x_1 = bert_1
            x_2 = bert_2
            for i in range(len(self._encoder)):
                encoded_1 = self._encoder[i](x_1)
                encoded_2 = self._encoder[i](x_2)
                x_1 = encoded_1
                x_2 = encoded_2

        interaction_output = self._interaction(encoded_1, encoded_2)
        output = self._extractor(interaction_output)

        return output
