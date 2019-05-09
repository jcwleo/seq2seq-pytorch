import torch
import numpy as np


class CharDict(object):
    def __init__(self, input_data, output_data):
        print('Making character dict...')
        self.char2idx = self.make_char2index(input_data, output_data)
        self.idx2char = self.make_index2char(self.char2idx)
        print('Finished making!')

    def make_char2index(self, input_data, output_data):
        total_data = input_data + output_data
        char_list = list(set(''.join(total_data)))
        char2idx = {}
        char2idx['SOS'] = 0
        char2idx['EOS'] = 1
        char2idx['PAD'] = 2
        for idx, char in enumerate(char_list):
            char2idx[char] = idx + 3

        return char2idx

    def make_index2char(self, char2idx):
        return {int(y): x for x, y in char2idx.items()}


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def get_attn_subsequent_mask(output):
    attn_shape = [output.size(1), output.size(2), output.size(2)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask
