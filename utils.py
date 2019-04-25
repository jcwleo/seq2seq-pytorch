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
