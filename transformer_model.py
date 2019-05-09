import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from utils import *


class PositionWiseFFN(nn.Module):
    def __init__(self, n_dim, n_head):
        super(PositionWiseFFN, self).__init__()
        d_ff = n_dim * 2
        self.conv1 = nn.Conv1d(in_channels=n_head * n_dim, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=n_head * n_dim, kernel_size=1)

    def forward(self, input):
        input = input.transpose(1, 2)

        output = F.relu(self.conv1(input))
        output = self.conv2(output)

        output = output.transpose(1, 2)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, max_len, n_dim, n_head):
        super(MultiHeadAttention, self).__init__()
        self.max_len = max_len
        self.scale_dot_product_attention = ScaleDotProductAttention(n_dim, n_head)
        self.concat_linear = nn.Linear(n_head * n_dim, n_head * n_dim)

    def forward(self, query, key, value, masking=False):
        batch_size = len(query)
        output, attention = self.scale_dot_product_attention(query, key, value, masking)

        # concat scaled dot product attention result
        length = output.size(2)
        output = output.transpose(1, 2).contiguous().view(batch_size, length, -1)
        output = self.concat_linear(output)
        return output


class ScaleDotProductAttention(nn.Module):
    def __init__(self, n_dim, n_head):
        super(ScaleDotProductAttention, self).__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        d_model = n_head * n_dim
        self.query_linear = nn.Linear(d_model, n_dim * n_head)
        self.key_linear = nn.Linear(d_model, n_dim * n_head)
        self.value_linear = nn.Linear(d_model, n_dim * n_head)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, masking=False):
        batch_size = len(query)

        # Extend (query, key, value) by the n_head
        # (batch, n_head, max_len, dimension)
        query = self.query_linear(query).view(batch_size, -1, self.n_head, self.n_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.n_head, self.n_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.n_head, self.n_dim).transpose(1, 2)

        output = torch.matmul(query, key.transpose(-1, -2))
        output /= np.power(self.n_dim, 1 / 2)

        # masking in decoder
        if masking:
            mask = get_attn_subsequent_mask(output)
            output.masked_fill_(mask, -1e9)

        attention = self.softmax(output)
        output = torch.matmul(attention, value)

        return output, attention


class Encoder(nn.Module):
    def __init__(self, vocab_size, char_dict, max_len, n_dim=128, n_layers=1, batch_size=1, n_head=8):
        super(Encoder, self).__init__()
        # parameters
        self.n_dim = n_dim
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_layers = n_layers

        self.emb_layer = nn.Embedding(self.vocab_size, n_dim * n_head)
        self.pos_emb_layer = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.vocab_size, n_dim * n_head),
                                                          freeze=True)

        self.multi_head_attention = MultiHeadAttention(max_len, n_dim, n_head)
        self.pos_ffn = PositionWiseFFN(n_dim, n_head)
        self.layer_norm = nn.LayerNorm(n_head * n_dim)

    def forward(self, input_data):
        # positional encoding
        embedded = self.emb_layer(input_data) + self.pos_emb_layer(input_data)

        for _ in range(self.n_layers):
            # Self Multi Head attention
            output = self.multi_head_attention(embedded, embedded, embedded)

            # Add & Norm
            embedded = self.layer_norm(output + embedded)

            # Feed Forward
            output = self.pos_ffn(output)

            # Add & Norm
            embedded = self.layer_norm(output + embedded)

        output = embedded

        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size, char_dict, max_len, n_dim=128, n_layers=1, batch_size=1, n_head=8):
        super(Decoder, self).__init__()
        # parameters
        self.n_dim = n_dim
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.char_dict = char_dict

        self.emb_layer = nn.Embedding(self.vocab_size, n_dim * n_head)
        self.pos_emb_layer = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.vocab_size, n_dim * n_head),
                                                          freeze=True)

        self.multi_head_attention = MultiHeadAttention(max_len, n_dim, n_head)
        self.pos_ffn = PositionWiseFFN(n_dim, n_head)
        self.layer_norm = nn.LayerNorm(n_head * n_dim)

    def forward(self, input_data, encoder_output, is_eval=False):
        input_len = len(input_data)
        if not is_eval:
            # add Start of sentence for training
            input_data = torch.cat((torch.LongTensor([[self.char_dict.char2idx['SOS']]] * input_len), input_data[:, :-1]), dim=-1)

            # positional encoding
            embedded = self.emb_layer(input_data) + self.pos_emb_layer(input_data)
            for _ in range(self.n_layers):
                # Self Multi Head attention (masking=True)
                output = self.multi_head_attention(embedded, embedded, embedded, masking=True)

                # Add & Norm
                embedded = self.layer_norm(output + embedded)

                # Encoder Multi Head attention
                output = self.multi_head_attention(embedded, encoder_output, encoder_output)

                # Add & Norm
                embedded = self.layer_norm(output + embedded)

                # Feed Forward
                output = self.pos_ffn(output)

                # Add & Norm
                embedded = self.layer_norm(output + embedded)

            output = embedded

            return output
        else:
            # positional encoding
            embedded = self.emb_layer(input_data) + self.pos_emb_layer(input_data)
            for _ in range(self.n_layers):
                # Self Multi Head attention (masking=True)
                output = self.multi_head_attention(embedded, embedded, embedded)

                # Add & Norm
                embedded = self.layer_norm(output + embedded)

                # Encoder Multi Head attention
                output = self.multi_head_attention(embedded, encoder_output, encoder_output)

                # Add & Norm
                embedded = self.layer_norm(output + embedded)

                # Feed Forward
                output = self.pos_ffn(output)

                # Add & Norm
                embedded = self.layer_norm(output + embedded)

            output = embedded

            return output


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, char_dict, loss_function, batch_size, max_len, n_layers=1, n_dim=128,
                 teacher_forcing_ratio=0.5, n_head=8):
        super(TransformerSeq2Seq, self).__init__()
        self.char_dict = char_dict
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = Encoder(vocab_size, char_dict, max_len, n_dim, n_layers, batch_size, n_head)
        self.decoder = Decoder(vocab_size, char_dict, max_len, n_dim, n_layers, batch_size, n_head)
        self.optim = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)

        self.linear = nn.Linear(n_dim * n_head, vocab_size)

    def forward(self, data):
        # Encoder
        encoder_output = self.encoder(data[:, 0])

        # Decoder
        decoder_output = self.decoder(data[:, 1], encoder_output)

        output = self.linear(decoder_output)
        return output

    def train_model(self, data):
        self.optim.zero_grad()

        output = self.forward(data)

        loss = self.loss_function(output.view(-1, self.vocab_size), data[:, 1].contiguous().view(-1))

        loss.backward()
        self.optim.step()
        return loss.data / self.max_len

    def eval_model(self, data):
        loss = 0
        decoder_outputs = []

        # Encoder
        encoder_output = self.encoder(data[:, 0])

        # Decoder
        decoder_input = torch.LongTensor([[self.char_dict.char2idx['SOS']]] * len(data))
        for c in range(self.max_len):
            decoder_result = self.decoder(decoder_input, encoder_output, is_eval=True)
            decoder_output = self.linear(decoder_result).squeeze(1)
            # print(decoder_output.shape)
            top_value, top_idx = F.softmax(decoder_output, dim=-1).data.topk(1)
            # print(top_idx.shape)
            decoder_outputs.append([self.char_dict.idx2char[top_i.data.tolist()[0]] for top_i in top_idx])
            decoder_input = top_idx

            loss += self.loss_function(decoder_output, data[:, 1, c])
        print(np.array(decoder_outputs).T[0])
        return loss.data / self.max_len
