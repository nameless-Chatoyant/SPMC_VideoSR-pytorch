import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import same_padding_conv

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.conv_layers = [nn.Conv2d(ch_in, ch_out, kernel_size, stride) for ch_in, ch_out, kernel_size, stride in zip(args.ch_in, args.ch_out, args.kernel_size, args.stride, args.types)]
        self.skip_connections = []
    def forward(self, x):
        for layer_idx, conv in enumerate(self.conv_layers):
            skip_connections = []
            x = same_padding_conv(x, conv)
            x = F.relu(x)
            if layer_idx in [0, 2]:
                skip_connections.append(x)
        self.skip_connections.append(skip_connections)
        return x


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels, self.kernel_size, 1,
                              self.padding)

    def forward(self, input, h, c):

        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, A.size()[1] / self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        return (Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])).cuda())
class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = ConvLSTMCell.init_hidden(bsize, self.hidden_channels[i], (height, width))
                    internal_state.append((h, c))
                # do forward
                name = 'cell{}'.format(i)
                (h, c) = internal_state[i]

                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.conv_layers = [dynamic_conv(ch_in, ch_out, kernel_size, stride) for ch_in, ch_out, kernel_size, stride, dynamic_conv in zip(args.ch_in, args.ch_out, args.kernel_size, args.stride, args.types)]
    def forward(self, x):
        for layer_idx, conv in enumerate(self.conv_layers):
            x = same_padding_conv(x, conv)
            x = F.relu(x)
        return x

class DetailFusionNet(nn.Module):
    def __init__(self, args):
        super(DetailFusionNet, self).__init__()
        self.encoder = Encoder(args.encoder)
        self.convlstm = ConvLSTM()
        self.decoder = Decoder(args.decoder)
    def forward(self, hr_sparses, lr):
        """
        # Arguments
            hr_sparses: [(b, c, h*scale, w*scale)] *t
            lr: (b, c, h, w)
        """
        encoded = []
        for i in hr_sparses:
            feature = self.encoder(i)
            print(self.encoder.skip_connections)
        # ConvLSTM(ch_in, ch_out, ch_hid, kernel_size)
        decoded = []
        for i in encoded:
            self.decoder(i)
        
        return decoded

if __name__ == '__main__':
    pass