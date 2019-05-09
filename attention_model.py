import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class RNNEncoder(nn.Module):
    def __init__(self, input_size, n_dim=128, batch_size=1):
        super(RNNEncoder, self).__init__()
        # parameters
        self.n_dim = n_dim
        self.input_size = input_size
        self.batch_size = batch_size

        self.embedding_layer = self.make_embedding_layer()
        self.hidden_layer = self.make_hidden_layer()

    def make_embedding_layer(self):
        embedding_layer = nn.Embedding(self.input_size, self.n_dim)

        return embedding_layer

    def make_hidden_layer(self):
        hidden_layer = nn.GRU(self.n_dim, self.n_dim)
        return hidden_layer

    def forward(self, input_data, hidden_state):
        embedded = self.embedding_layer(input_data)
        embedded = embedded.view(1, self.batch_size, self.n_dim)
        output, hidden_state = self.hidden_layer(embedded, hidden_state)
        return output, hidden_state

    def init_state(self):
        return torch.zeros(1, self.batch_size, self.n_dim)


class AttentionDecoder(nn.Module):
    def __init__(self, input_size, n_dim=128, n_layers=1, batch_size=1):
        super(AttentionDecoder, self).__init__()
        # parameters
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.input_size = input_size
        self.batch_size = batch_size

        self.embedding_layer = self.make_embedding_layer()
        self.hidden_layer = self.make_hidden_layer()
        self.concat_linear = nn.Linear(n_dim * 2, n_dim)
        self.linear = nn.Linear(n_dim, input_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def make_embedding_layer(self):
        embedding_layer = nn.Embedding(self.input_size, self.n_dim)

        return embedding_layer

    def make_hidden_layer(self):
        hidden_layer = nn.GRU(self.n_dim, self.n_dim)
        return hidden_layer

    def forward(self, input_data, hidden_state, encoder_outputs):
        embedded = self.embedding_layer(input_data)
        output = embedded.view(1, self.batch_size, self.n_dim)

        # calculate attention
        dot_product = torch.tensordot(torch.cat(encoder_outputs), hidden_state.permute(0, 2, 1))
        softmax_dot_product = F.softmax(dot_product, 0)
        attention_value = (softmax_dot_product.unsqueeze(2) * torch.cat(encoder_outputs)).sum(0, keepdim=True)

        hidden_state = self.concat_linear(torch.cat([attention_value, hidden_state], -1))

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.hidden_layer(output, hidden_state)
        output = self.linear(output[0])
        output = self.softmax(output)
        return output, hidden_state

    def init_state(self):
        return torch.zeros(1, self.batch_size, self.n_dim)


class AttentionSeq2Seq(object):
    def __init__(self, vocab_size, char_dict, loss_function, batch_size, max_len, n_layers=1, n_dim=128,
                 teacher_forcing_ratio=0.5):
        self.char_dict = char_dict
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.max_len = max_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = RNNEncoder(vocab_size, n_dim, batch_size)
        self.decoder = AttentionDecoder(vocab_size, n_dim, n_layers, batch_size)
        self.optim = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)

    def train_model(self, data):
        loss = 0
        self.optim.zero_grad()
        encoder_outputs = []

        # Encoder
        encoder_state = self.encoder.init_state()
        for c in range(self.max_len):
            encoder_output, encoder_state = self.encoder(data[:, 0, c], encoder_state)
            encoder_outputs.append(encoder_output)

        # Decoder
        decoder_input = torch.LongTensor([self.char_dict.char2idx['SOS']] * self.batch_size)
        decoder_state = encoder_state

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for c in range(self.max_len):
                decoder_result, decoder_state = self.decoder(decoder_input, decoder_state, encoder_outputs)
                decoder_input = data[:, 1, c]
                loss += self.loss_function(decoder_result, data[:, 1, c])
        else:
            for c in range(self.max_len):
                decoder_result, decoder_state = self.decoder(decoder_input, decoder_state, encoder_outputs)
                top_value, top_idx = decoder_result.data.topk(1)
                decoder_input = top_idx
                loss += self.loss_function(decoder_result, data[:, 1, c])

        loss.backward()
        self.optim.step()
        return loss.data / self.max_len

    def eval_model(self, data):
        loss = 0
        decoder_outputs = []
        encoder_outputs = []

        # Encoder
        encoder_state = self.encoder.init_state()
        for c in range(self.max_len):
            encoder_output, encoder_state = self.encoder(data[:, 0, c], encoder_state)
            encoder_outputs.append(encoder_output)

        # Decoder
        decoder_input = torch.LongTensor([self.char_dict.char2idx['SOS']] * self.batch_size)
        decoder_state = encoder_state
        for c in range(self.max_len):
            decoder_result, decoder_state = self.decoder(decoder_input, decoder_state, encoder_outputs)
            top_value, top_idx = decoder_result.data.topk(1)
            decoder_outputs.append([self.char_dict.idx2char[top_i.data.tolist()[0]] for top_i in top_idx])
            decoder_input = top_idx
            if c == self.max_len - 1:
                loss += self.loss_function(decoder_result,
                                           torch.LongTensor([self.char_dict.char2idx['EOS']] * self.batch_size))
            else:
                loss += self.loss_function(decoder_result, data[:, 1, c])
        print(np.array(decoder_outputs).T)
        return loss.data / self.max_len
