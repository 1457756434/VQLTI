#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn


class conv_leaky(nn.Module):
    def __init__(self, conv_config) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=conv_config[0],
                        out_channels=conv_config[1],
                        kernel_size=conv_config[2],
                        stride=conv_config[3],
                        padding=conv_config[4])
        self.leaky = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):

        x = self.conv2d(x)
        x = self.leaky(x)
        return x

class deconv_leaky(nn.Module):
    def __init__(self, deconv_config) -> None:
        super().__init__()
        self.convtranspose2d = nn.ConvTranspose2d(in_channels=deconv_config[0],
                        out_channels=deconv_config[1],
                        kernel_size=deconv_config[2],
                        stride=deconv_config[3],
                        padding=deconv_config[4])
        self.leaky = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        x = self.convtranspose2d(x)
        x = self.leaky(x)
        return x


class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, seq_len=4, ):
        super().__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

        self.seq_len = seq_len
        
      
    def forward(self, inputs=None, hidden_state=None,):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).to(inputs.device, non_blocking=True)
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(self.seq_len):
            if inputs is None:
                
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).to(htprev.device, non_blocking=True)
            else:
                x = inputs[index, ...]
       
            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext



class ConvGRU_encoder_block(nn.Module):

    def __init__(self, conv_config, GRU_config,) -> None:
        super().__init__()
        #conv_config:[in_channels, out_channels, kernel_size, stride, padding]
        #GRU_config :[shape, input_channels, filter_size, num_features, seq_le]
        shape = GRU_config[0]
        input_channels = GRU_config[1]
        filter_size = GRU_config[2]
        num_features = GRU_config[3]
        seq_len = GRU_config[4]
        
        self.subnet = conv_leaky(conv_config=conv_config)
        self.GRU_block = CGRU_cell(shape, input_channels, filter_size, num_features, seq_len=seq_len, )

    def forward(self, x):
        seq_number, batch_size, input_channel, height, width = x.size()
        x = torch.reshape(x, (-1, input_channel, height, width))
        
        x = self.subnet(x)
        x = torch.reshape(x, (seq_number, batch_size, x.size(1), x.size(2), x.size(3)))
        outputs_stage, state_stage = self.GRU_block(x, None)
        return outputs_stage, state_stage

class ConvGRU_decoder_block(nn.Module):

    def __init__(self, deconv_config, GRU_config, ) -> None:
        super().__init__()
        #conv_config:[in_channels, out_channels, kernel_size, stride, padding]
        #GRU_config :[shape, input_channels, filter_size, num_features, seq_le]
        shape = GRU_config[0]
        input_channels = GRU_config[1]
        filter_size = GRU_config[2]
        num_features = GRU_config[3]
        seq_len = GRU_config[4]
        
        self.subnet = deconv_leaky(deconv_config=deconv_config)
        self.GRU_block = CGRU_cell(shape, input_channels, filter_size, num_features, seq_len=seq_len, )

    def forward(self, x, h_state):
        x, state_stage = self.GRU_block(x, h_state)
        seq_number, batch_size, input_channel, height, width = x.size()
        x = torch.reshape(x, (-1, input_channel, height, width))
        x = self.subnet(x)
        x = torch.reshape(x, (seq_number, batch_size, x.size(1), x.size(2), x.size(3)))
        # T,B,C,H,W
        return x




class CLSTM_cell(nn.Module):
    """
    ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1])
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1])
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1])
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)
