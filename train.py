import torch
import torch.nn as nn
import torch.utils.data

from model import Seq2Seq
from attention_model import AttentionSeq2Seq
from utils import CharDict
from pre_processing import *

MAX_LEN = 10
BATCH_SIZE = 1
EPOCH = 1000
HIDDEN_SIZE = 64
HIDDEN_LAYERS = 1
TEACHER_FORCING_RATIO = 0.0

# read data
input_data, output_data = read_data('data/data_set.txt')

# make vocab dict
char_dict = CharDict(input_data, output_data)

# preprocessing data
preproc_input_data, preproc_output_data, pairs = preproc_data(input_data, output_data, char_dict, MAX_LEN)

# TODO
# training set
pairs = torch.LongTensor(pairs)
train_loader = torch.utils.data.DataLoader(pairs, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

# TODO
# valid set

# make model
loss_function = nn.NLLLoss()

# Attention model
seq2seq = AttentionSeq2Seq(len(char_dict.char2idx), char_dict, loss_function, batch_size=BATCH_SIZE, max_len=MAX_LEN,
                           n_layers=HIDDEN_LAYERS, n_dim=HIDDEN_SIZE, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

# Basic model
# seq2seq = Seq2Seq(len(char_dict.char2idx), char_dict, loss_function, batch_size=BATCH_SIZE, max_len=MAX_LEN,
#                   n_layers=HIDDEN_LAYERS, n_dim=HIDDEN_SIZE, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

# train
for e in range(EPOCH):
    print('----------------')
    print(e, 'epoch')
    for b, data in enumerate(train_loader):
        # data[0] = input
        # data[1] = output

        print('Train Loss:', seq2seq.train(data).item())
    print('Valid Loss:', seq2seq.eval(data).item())
