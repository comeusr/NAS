import torch.nn as nn
from genotype import PRIMITIVES
from utils import *

class LSTMController(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_nodes = args.num_nodes
        self.num_operations = len(PRIMITIVES)
        self.hid_size = args.controller_hid
        self.temperature = args.temp
        self.num_blocks = sum(1 for i in range(self.num_nodes) for n in range(2 + i)) * 2

        self.decoders = []
        self.lstm = nn.LSTMCell(1, self.hid_size)
        for block_idx in range(self.num_blocks):
            decoder = nn.Linear(self.hid_size, self.num_operations)
            self.decoders.append(decoder)
            # lstm_cell = nn.LSTMCell(1, self.hid_size)
            # self.lstm.append(lstm_cell)

        # self.lstm_cell =

        self._decoders = torch.nn.ModuleList(self.decoders)
        # self._lstm = torch.nn.ModuleList(self.lstm)

        self.reset_parameters()
        self.static_init_hidden = keydefaultdict(self.init_hidden)

        # Debug purpose set LSTM weights to 0
        # nn.init.zeros_(self._lstm.parameters())
        # Debug purpose set decoders weight zero
        # nn.init.zeros_(self._decoders.parameters())

        self.last_bias = nn.ParameterList([x.bias for x in self._decoders])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def _get_default_hidden(key):
            return get_variable(
                torch.zeros(key, self.hid_size),
                self.device,
            )

        def _get_default_input(key):
            return get_variable(
                torch.zeros(key, 1),
                self.device,
            )

        self.static_inputs = keydefaultdict(_get_default_input)

        # self.encoder = nn.Embedding(self.hid_size, self.hid_size)
        # self.lstm = nn.LSTMCell(self.num_operations, self.hid_size)

    def _last_bias(self):
        return self.last_bias

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
            # param.data.fill_(0)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self):
        '''
        :return: A soft architecture.
        '''
        Probs = []

        # initial the input and first
        inputs = self.static_inputs[1]
        # print(inputs)
        hidden = self.static_init_hidden[1]

        for idx in range(self.num_blocks):
            # print('fail at index '+str(idx))
            hx, cx = self.lstm(inputs, hidden)
            hx = hx  # .cuda()
            cx = cx  # .cuda()
            hidden = (hx, cx)
            loggit = self._decoders[idx](hx)
            # loggit = loggit/self.temperature

            inputs = get_variable(torch.tensor(idx + 1.0), self.device)
            inputs = inputs.unsqueeze(0).unsqueeze(0)

            # print('inputs: '+str(inputs))
            # print('Input shape: '+str(inputs.shape))
            # print('Hidden Shape: '+ str())

            Probs.append(loggit)

        temp_result = torch.cat(Probs, 0)
        # result = torch.FloatTensor(2, int(self.num_blocks/2), len(OPS))
        # result[0] = temp_result[:int(self.num_blocks/2)]
        # result[1] = temp_result[int(self.num_blocks/2):]
        result = []
        result.append(temp_result[:int(self.num_blocks / 2)])
        result.append(temp_result[int(self.num_blocks / 2):])

        return result

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.hid_size)
        return (get_variable(zeros, self.device, ),
                get_variable(zeros.clone(), self.device))