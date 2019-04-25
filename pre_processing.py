def preproc_data(input_data, output_data, char_dict, max_length):
    print('Preprocessing data...')
    preproc_input_data = []
    preproc_output_data = []
    pairs = []
    for input, output in zip(input_data, output_data):
        preproc_input = []
        preproc_output = []
        for i in range(max_length):
            if len(input) > i:
                input_idx = char_dict.char2idx.get(input[i])
            else:
                # add padding
                input_idx = 2
            if len(output) > i:
                output_idx = char_dict.char2idx.get(output[i])
            elif len(output) == i:
                # add EOS
                output_idx = 1
            else:
                # add padding
                output_idx = 2

            preproc_input.append(input_idx)
            preproc_output.append(output_idx)

        preproc_input_data.append(preproc_input)
        preproc_output_data.append(preproc_output)
        pairs.append([preproc_input, preproc_output])
    print('Finished preprocessing!')

    return preproc_input_data, preproc_output_data, pairs


def read_data(data_path):
    print('Reading data...')

    # Read the file and split into lines
    lines = open('%s' % (data_path), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('\t')] for l in lines]

    input_data = []
    output_data = []

    for input, output in pairs:
        input_data.append(input)
        output_data.append(output)
    print('Finished reading!')

    return input_data, output_data
