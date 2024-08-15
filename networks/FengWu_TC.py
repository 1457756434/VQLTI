import torch
import torch.nn as nn
import numpy as np
import math
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from petrel_client.client import Client
import os
import io
from collections import OrderedDict
import matplotlib.pyplot as plt
import json

#networks.
from networks.FPN_TC_block import FPN_TC_block

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout,):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
     
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_





class Encoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1,2,4,8), num_res_blocks=2,
                 attn_resolutions=[2], dropout=0.0, resamp_with_conv=True, in_channels=69,
                 resolution=32, z_channels=128, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                    
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)


        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], )
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, )
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, )

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

    

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, )
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, )

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h




class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)
    
    def forward(self, latents: Tensor) -> Tensor:
        """
        latents: [B x N x D]
        """
        # latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BN x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BN x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BN, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BN x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BN, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x N x D]

        #############################################################################
        # # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        
        vq_loss = commitment_loss * self.beta + embedding_loss

        #############################################################################
        # vq_loss = None


        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents, vq_loss  # [B x N x D]


    
class img_encoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1,2,2,4), num_res_blocks=2,
                 attn_resolutions=[2], dropout=0.0, resamp_with_conv=True, in_channels=69,
                 resolution=32, z_channels=128, double_z=True, ) -> None:
        super().__init__()
        self.encoder = Encoder(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                 attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv, in_channels=in_channels,
                 resolution=resolution, z_channels=z_channels, double_z=double_z, )

    def forward(self, x):
        x = self.encoder(x)
        return x




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TC_info_Embedding(nn.Module):
    def __init__(self, in_channel, embed_dim, num_embed=200, beta=0.25, dropout=0.1):
        super().__init__()
        self.num_embed = num_embed
        self.beta = beta
        self.embed_layer = nn.Linear(in_channel, embed_dim*in_channel)  # Convert (x, y) to a specified embedding sizei
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        # self.embed_layers = nn.ModuleList()
        # for _ in range(in_channel):
        #     self.embed_layers.append(nn.Embedding(num_embeddings=num_embed, embedding_dim=embed_dim))
        self.linear_embed1 = nn.Linear(embed_dim*in_channel, embed_dim*in_channel*2)
        self.linear_embed2 = nn.Linear(embed_dim*in_channel*2, embed_dim*in_channel*4)
        self.dropout = nn.Dropout(dropout)

        patch_size = in_channel*4

        fussion_patch_dim = embed_dim
        self.to_patch = nn.Sequential(
            Rearrange('b (p c) -> b p c', p=patch_size),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
    def Reverse_embedding(self, embed_pre, vq_loss_all, input):
        """
        input: B, N
        D: embed_dim
        """
        B, N, D = embed_pre.shape
        device = embed_pre.device
        
        target_pre_list = []

        

        for i in range(self.in_channel):
            input_i = F.one_hot(input[:,i].to(torch.int64), num_classes=self.num_embed).float()
            
            dist = torch.sum(embed_pre[:,i,:] ** 2, dim=1, keepdim=True) + \
                   torch.sum(self.embed_layers[i].weight **2, dim=1) - \
                   2 * torch.matmul(embed_pre[:,i,:], self.embed_layers[i].weight.t()) #[B, K]
            encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)
            # print(f"encoding_inds:{encoding_inds.shape}")
            encoding_one_hot = torch.zeros(encoding_inds.size(0), self.num_embed, device=device)
            encoding_one_hot.scatter_(1, encoding_inds, 1) #B, K

            quantized_latents = torch.matmul(encoding_one_hot, self.embed_layers[i].weight)  #[B, D]

            commitment_loss = F.mse_loss(quantized_latents.detach(), embed_pre[:, i, :])     
            embedding_loss = F.mse_loss(quantized_latents, embed_pre[:, i, :].detach())    

            vq_loss = commitment_loss * self.beta + embedding_loss

            encoding_inds = encoding_inds.squeeze(-1)
            # print(f"encoding_inds:{encoding_inds.shape}")
            encoding_inds = encoding_inds.detach()
            target_pre = F.softmax(1/dist, dim=-1)
            target_pre_list.append(target_pre)
            target_loss = F.cross_entropy(target_pre, input_i)
            vq_loss_all = vq_loss_all +  vq_loss + target_loss
            
        vq_loss_all = vq_loss_all/B
        target_pre_array = torch.stack(target_pre_list, dim=1)
        
        
        return target_pre_array, target_loss, vq_loss_all
    

    def forward(self, src):
        
        embed_array = self.embed_layer(src)
        embed_array = self.linear_embed1(embed_array)
        embed_array = nonlinearity(embed_array)
        embed_array = self.linear_embed2(embed_array)
        embed_array = self.dropout(embed_array)
        embed_array = self.to_patch(embed_array)
        
        return embed_array

def singel_num(t):
    return t[0] if isinstance(t, tuple) else t





class CrossAttention(nn.Module):
    def __init__(self,
                 
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.query = nn.Linear(n_embd, n_embd)

        self.key = nn.Linear(condition_embd, n_embd)
        
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        

    def forward(self, x, encoder_output, mask=None):
        B, N, C = x.size()
        
        B, N_token, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        q = self.query(x).view(B, N, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs)

        k = self.key(encoder_output).view(B, N_token, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs)
        
        v = self.value(encoder_output).view(B, N_token, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, N, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, N, N)

        att = F.softmax(att, dim=-1) # (B, nh, N, N)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, N, N) x (B, nh, N, hs) -> (B, nh, N, hs)
        y = y.transpose(1, 2).contiguous().view(B, N, C) # re-assemble all head outputs side by side, (B, N, C)
        att = att.mean(dim=1, keepdim=False) # (B, N, N)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y



class TC_info_Decoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, patch_num=8,in_channel=2, patch_dim=256, n_head=4) -> None:
        super().__init__()
        
        self.cross_att = CrossAttention(n_embd=in_features, condition_embd=in_features, n_head=n_head)
        patch_num = patch_num
        fussion_patch_dim = patch_num*patch_dim
        self.patch_2_line = nn.Sequential(
            Rearrange('b p c -> b (p c)', p=patch_num),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, fussion_patch_dim),
            nn.LayerNorm(fussion_patch_dim),
        )
        self.fc1 = nn.Linear(fussion_patch_dim, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        # self.back_2_info = nn.Sequential(
        #     Rearrange('b (n c) -> b n c', n=in_channel),
        # )

    def forward(self, x, img_token):
        #(Q, KV)
        x = self.cross_att(x, img_token)
        
        x = self.patch_2_line(x)
        x = self.fc1(x)
        x2 = self.fc2(x)
        x = x + x2
        x = self.fc3(x)
        # x = self.back_2_info(x)
        return x



class TC_info_Decoder_2(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, patch_num=8,in_channel=2, patch_dim=256, n_head=4) -> None:
        super().__init__()
        
        self.cross_att = CrossAttention(n_embd=in_features, condition_embd=in_features, n_head=n_head)
        patch_num = patch_num
        fussion_patch_dim = patch_num*patch_dim
        self.patch_2_line = nn.Sequential(
            Rearrange('b p c -> b (p c)', p=patch_num),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, hidden_features),
            nn.LayerNorm(hidden_features),
        )
        self.fc1 = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.back_2_info = nn.Sequential(
        #     Rearrange('b (n c) -> b n c', n=in_channel),
        # )

    def forward(self, x, img_token):
        #(Q, KV)
        x = self.cross_att(x, img_token)
        
        x = self.patch_2_line(x)
        x2 = self.fc1(x)
        x = x + x2
        x = self.fc2(x)
        # x = self.back_2_info(x)
        return x





class VQ_TC_Model_ERA5(nn.Module):
    def __init__(self, ERA5_image_size=(40, 40), ERA5_channels=69, \
                dim=256, in_channel=2, Cross_att_qkv_mode="Q_TC_info",
                num_embeddings: int = 1024, embedding_dim: int =128, beta: float = 0.25, n_head=4, \
                is_choose_channel=False, choose_channel_hid_dim=16, double_z=True, ch_mult=(1,2,2,4),\
                encoder_mode="Resnet", decoder_mode="Cross_atten", dropout=0.1) -> None:
        super().__init__()
        self.encoder_mode = encoder_mode
        self.beta = beta
        # self.wind_embed = nn.Embedding(n_wind_embed, dim)
        # self.mslp_embed = nn.Embedding(n_mslp_embed, dim)
        self.Cross_att_qkv_mode = Cross_att_qkv_mode
        if Cross_att_qkv_mode == "Q_TC_info":
            patch_num=in_channel*4
        elif Cross_att_qkv_mode == "Q_img_token":
            patch_num=1166
        else:
            raise ValueError("no such Cross_att_qkv_mode")
        self.is_choose_channel = is_choose_channel
        if is_choose_channel:
            self.ERA5_choose_channel_module = channel_choose_module(inp_dim=ERA5_channels, hid_dim=choose_channel_hid_dim)
            self.future_ERA5_choose_channel_module = channel_choose_module(inp_dim=ERA5_channels, hid_dim=choose_channel_hid_dim)

        self.tc_info_embedding = TC_info_Embedding(in_channel=in_channel, embed_dim=dim, dropout=dropout)
        if double_z:
            z_channels = int(dim//2)
        else:
            z_channels = dim

        if encoder_mode=='Resnet':
            self.ERA5_encoder = img_encoder(in_channels=ERA5_channels, z_channels=z_channels, double_z=double_z, ch_mult=ch_mult,)
            self.future_img_encoder = img_encoder(in_channels=ERA5_channels, z_channels=z_channels, double_z=double_z, ch_mult=ch_mult,)
        elif encoder_mode=='FPN':
 
            self.ERA5_encoder = FPN_TC_block(FPN_mode='ERA5', patch_size=(5, 5), in_chancel=ERA5_channels, dim=dim)
            self.future_img_encoder = FPN_TC_block(FPN_mode='ERA5', patch_size=(5, 5), in_chancel=ERA5_channels, dim=dim)

        ERA5_image_size = singel_num(ERA5_image_size)
        self.img_token_size = (ERA5_image_size//8)**2
        
        self.cross_att = CrossAttention(n_embd=dim, condition_embd=dim, n_head=n_head)


        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)
        
        if decoder_mode=="Cross_atten":
            self.tc_info_decoder = TC_info_Decoder(in_features=dim, hidden_features=dim*in_channel*2, patch_num=patch_num,\
                                                out_features=in_channel, in_channel=in_channel, n_head=n_head, patch_dim=dim)
        else:
            self.tc_info_decoder = TC_info_Decoder_2(in_features=dim, hidden_features=dim*in_channel*2, patch_num=patch_num,\
                                                out_features=in_channel, in_channel=in_channel, n_head=n_head, patch_dim=dim)
        
        
    def forward(self, new_era5_data, inp_lable, label):
        """
        new_era5_data: B, C, H, W
        inp_lable: B, 2
        """
        # wind = inp_lable[:, 0].long()
        # mslp = inp_lable[:, 1].long()
        
        # embed_wind = self.wind_embed(wind)
        # embed_mslp = self.mslp_embed(mslp)
       
        tc_info_feature = self.tc_info_embedding(inp_lable)
        
        if self.is_choose_channel:

            new_era5_data = torch.permute(new_era5_data, (0, 2, 3, 1))
            new_era5_data = self.ERA5_choose_channel_module(new_era5_data)
            new_era5_data = torch.permute(new_era5_data, (0, 3, 1, 2))
        if self.encoder_mode=="Resnet":
            era5_feature = self.ERA5_encoder(new_era5_data).flatten(-2)
            
            img_token = era5_feature
            img_token = img_token.transpose(1, 2)
        elif self.encoder_mode=="FPN":
           
            era5_feature = self.ERA5_encoder(new_era5_data)
            img_token = era5_feature
            # print(f"img_token: {img_token.shape}")

     
        
        if self.Cross_att_qkv_mode == "Q_TC_info":
            hidden_state = self.cross_att(tc_info_feature, img_token)
        elif self.Cross_att_qkv_mode == "Q_img_token":
            hidden_state = self.cross_att(img_token, tc_info_feature)
        # print(hidden_state.shape)
        quantized_hidden_state, vq_loss = self.vq_layer(hidden_state)
        
        # print(img_token.shape)
        # self.tc_info_decoder()

        if self.is_choose_channel:
            new_era5_data = torch.permute(new_era5_data, (0, 2, 3, 1))
            new_era5_data_future = self.future_ERA5_choose_channel_module(new_era5_data)
            new_era5_data_future = torch.permute(new_era5_data_future, (0, 3, 1, 2))
        else:
            new_era5_data_future = new_era5_data

        if self.encoder_mode=="Resnet":
            img_future = self.future_img_encoder(new_era5_data_future).flatten(-2)
            img_future = img_future.transpose(1, 2)
        elif self.encoder_mode=="FPN":
            img_future = self.future_img_encoder(new_era5_data_future)
            #print(f"img_future: {img_future.shape}")
        #(Q, KV)
        tc_info_out = self.tc_info_decoder(quantized_hidden_state, img_future)
 
        
        recons_loss = F.mse_loss(tc_info_out, label)
        # loss = recons_loss + vq_loss
        #print(f"recons_loss: {recons_loss.item()}, vq_loss: {vq_loss.item()}")
        
        quantized_hidden_state = torch.reshape(quantized_hidden_state, (quantized_hidden_state.shape[0], -1))
        return tc_info_out, recons_loss, vq_loss, quantized_hidden_state
        
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}
    

class hidden_state_LSTM(nn.Module):
    def __init__(self, input_size=256*8, hidden_size=256*8*2, num_layers=2, output_size=256*8,\
                 input_length=4, out_length = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 

        self.lstm_in = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.lstm_out = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        
        self.input_length = input_length
        self.out_length = out_length
    

    def forward(self, input_seq):
        #input_seq: [B, T, N, C]
        # output(batch_size, seq_len, num_directions * hidden_size)
        # print(input_seq.shape)
        batch_size, seq_len, token_size, channel_size = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2], input_seq.shape[3]
        input_seq = torch.reshape(input_seq, (input_seq.shape[0], input_seq.shape[1], -1))
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        # print(input_seq.shape)
        _, (h_n, c_n) = self.lstm_in(input_seq) # 


        ##########################################
        # print(h_n.shape, c_n.shape)
        de_inp = input_seq[:, -1, :]
        
        
        de_inp = torch.unsqueeze(de_inp, 1)
        de_output, (h_n, c_n) = self.lstm_out(de_inp, (h_n, c_n))
        # print(de_output.shape, h_n.shape, c_n.shape)
        de_output = self.linear(torch.squeeze(de_output, 1))
        
        
        de_output = torch.reshape(de_output, (batch_size, token_size, -1))
        ##########################################
        # de_inputs = torch.zeros(batch_size, self.out_length, self.input_size).to(input_seq.device)
        # de_output, _ = self.lstm_out(de_inputs, (h_n, c_n))
        # outputs = self.linear(de_output)

        #print(outputs.shape)
        return de_output



class checkpoint_ceph(object):
    def __init__(self, conf_path="~/petreloss.conf", checkpoint_dir="cephnew:s3://myBucket/my_checkpoint") -> None:
        self.client = Client(conf_path=conf_path)
        self.checkpoint_dir = checkpoint_dir

    def load_checkpoint(self, url):
        url = os.path.join(self.checkpoint_dir, url)
        # url = self.checkpoint_dir + "/" + url
        if not self.client.contains(url):
            return None
        with io.BytesIO(self.client.get(url, update_cache=True)) as f:
            checkpoint_data = torch.load(f, map_location=torch.device('cpu')) 
        return checkpoint_data

class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Cross_Transformer(nn.Module):

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttention(                 
                 n_embd=dim, # the embed dim
                 condition_embd=dim, # condition dim
                 n_head=heads, # the number of heads
                 attn_pdrop=dropout, # attention dropout prob
                 resid_pdrop=dropout, # residual attention dropout prob
                 causal=True),
                FeedForward(dim=dim, hid_dim=mlp_dim, dropout=dropout),
            ]))
            
                
    def forward(self, Q, KV):
        for att_seq, feed_seq, in self.layers:
            Q = att_seq(Q, KV) + Q
            Q = feed_seq(Q) + Q
            #print(seq_embedding.shape, img_embedding.shape)
        return self.norm(Q)

class ST_transformer(nn.Module):

    def __init__(self, dim, depth, heads, mlp_dim, inp_len=4, patch_num=8, dropout=0.) -> None:
        super().__init__()
        self.Cross_Transformer = Cross_Transformer(dim, depth, heads, mlp_dim, dropout)
        self.causal_dim = int(patch_num * (1+inp_len)*inp_len/2)
        causal_num = int((1+inp_len)*inp_len/2)
        #print(self.causal_num)
        if self.causal_dim!=11660:
            self.linear_block = nn.Linear(dim*self.causal_dim, dim*patch_num)
        else:
            self.linear_block = nn.Linear(dim*causal_num, dim)
    def forward(self, Q):
        """
        Q: [B, T, n, dim]
        """
        B, T, n, dim = Q.shape
        KV = Q
        
        # print(f"fusion_embedding: {fusion_embedding.shape}")
        att_score = self.Cross_Transformer(Q[:, 0, :, :], KV[:, 0, :, :])
        # print(f"att_score: {att_score.shape}")
        for t in range(1, T):
            for t_of_t in range(0, t+1):
                
                att_score_t = self.Cross_Transformer(Q[:, t_of_t, :, :], KV[:, t, :, :])
                if self.causal_dim!=11660:
                    att_score = torch.concat((att_score, att_score_t), dim=-2)
                else:
                    att_score = torch.concat((att_score, att_score_t), dim=-1)
        #         print(att_score_t.shape)
        # print(f"att_score: {att_score.shape}")
        if self.causal_dim!=11660:
            att_score = torch.reshape(att_score, (att_score.shape[0], -1))
            att_score = self.linear_block(att_score)
            att_score = torch.reshape(att_score, (B, n, -1))
        else:
            att_score = self.linear_block(att_score)
            #print(f"att_score: {att_score.shape}")
        # print(f"att_score: {att_score.shape}")
        return att_score


class SELF_transformer(nn.Module):

    def __init__(self, dim, depth, heads, mlp_dim, inp_len=4, patch_num=8, dropout=0.) -> None:
        super().__init__()
        self.Cross_Transformer = Cross_Transformer(dim, depth, heads, mlp_dim, dropout)
        self.linear_block = nn.Linear(patch_num*inp_len, patch_num)
    def forward(self, Q):
        """
        Q: [B, T, n, dim]
        """
        B, T, n, dim = Q.shape
        Q = Q.reshape(B, -1, dim)
        KV = Q
        

        # att_score: [B, T*n, dim]
        att_score = self.Cross_Transformer(Q, KV)
        # print(f"att_score: {att_score.shape}")
       
        att_score = torch.permute(att_score, (0, 2, 1))
        att_score = self.linear_block(att_score)
        att_score = torch.permute(att_score, (0, 2, 1))
        #print(f"att_score: {att_score.shape}")
        return att_score



class physics_SINDy_module(nn.Module):
    def __init__(self, in_channel=2, polyorder=2, pre_dim=2, dim=128) -> None:
        super().__init__()
        C = in_channel 
        self.in_channel = in_channel
        self.in_dim = int(((C+1)*C/2)*C)
        self.img_encoder = FPN_TC_block(FPN_mode='ERA5', patch_size=(5, 5), in_chancel=in_channel, dim=dim)
        
        self.spare_linear = nn.Sequential(
            nn.Linear(88*dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, pre_dim),
        )
        
        self.polyorder = polyorder
        # self.out_linear = nn.Linear(dim*88, pre_dim)
        
    def matrices_data(self, data_now):
        """
        B,C,H,W
        """
        B,C,H,W = data_now.shape
        polyorder = self.polyorder
        index = 0
        if(polyorder>=2):
            num_C = int(((C+1)*C/2)*self.in_channel)
            data_poly = torch.zeros((B,num_C,H,W), device=data_now.device,)#device=data_now.shape
            for i in range(C):
                for j in range(i, C):
                    data_poly[:, index] = torch.mul(data_now[:, i], data_now[:, j])
                    index = index + 1 
        else:
            raise ValueError("no such polyorder")
        return data_poly
    

    def forward(self, add_data):
        """
        B,T,C,H,W
        """
        B,T,C,H,W = add_data.shape
        add_data = add_data.to(torch.float32)
        # add_data = add_data.reshape(-1,H,W)
        add_data = add_data.reshape(-1,C,H,W)
        
        # print(add_data.shape)
        add_data = self.img_encoder(add_data)
        add_data = add_data.reshape(B, T, add_data.shape[-2], add_data.shape[-1])
      
        return add_data


class physics_guid_module(nn.Module):
    def __init__(self, era5_channel=69, polyorder=2, out_dim = 69, dim = 128, pre_dim=2) -> None:
        super().__init__()
        C = era5_channel 
        self.in_dim = era5_channel + int((C+1)*C/2)
        self.spare_linear = nn.Linear(self.in_dim, out_dim)
        self.polyorder = polyorder

        self.img_encoder = FPN_TC_block(FPN_mode='ERA5', patch_size=(5, 5), in_chancel=out_dim, dim=dim)
       
        self.out_linear = nn.Linear(dim*88, pre_dim)
        
    def pool_data(self, data_last, data_now):
        """
        B,C,H,W
        """
        B,C,H,W = data_now.shape
        polyorder = self.polyorder
        data_diff = data_now - data_last

        index = 0
        if(polyorder>=2):
            num_C = int((C+1)*C/2)
            data_poly = torch.zeros((B,num_C,H,W), device=data_now.device,)#device=data_now.shape
            for i in range(C):
                for j in range(i, C):
                    data_poly[:, index] = torch.mul(data_now[:, i], data_now[:, j])
                    index = index + 1 
        else:
            raise ValueError("no such polyorder")
        data_return = torch.concat((data_diff, data_poly), dim=1)

        return data_return
    

    def forward(self, era5_data_last, era5_data_now, inp_lable_last):
        """
        B,C,H,W
        """
        data_return = self.pool_data(era5_data_last, era5_data_now)
        data_return = torch.permute(data_return, dims=(0,2,3,1))
        data_return = self.spare_linear(data_return)
        data_return = torch.permute(data_return, dims=(0,3,1,2))
        era5_feature = self.img_encoder(data_return)
        # print(era5_feature.shape)
        era5_feature = torch.reshape(era5_feature, shape=(era5_feature.shape[0], -1))
        pre_data = self.out_linear(era5_feature) + inp_lable_last
        # print(pre_data.shape)
        return pre_data

class FengWu_TC(nn.Module):
    def __init__(self, backboon="LSTM", residual=False, hid_residual=False,\
                is_load_checkpoint=True, is_freeze_model=True, is_freeze_future=True, cmp_mode="Many_to_Many",
                decoder_backboon="Origin", is_physics=False, is_hid_vq=False,
                **kwargs
                ) -> None:
        super().__init__()
        
        self.residual = residual
        self.hid_residual = hid_residual
        self.decoder_backboon = decoder_backboon
        #del self.VQ_TC.tc_info_decoder
        # for key in checkpoint_dict["model"]:
        #     new_state_dict = OrderedDict()
        #     for k, v in checkpoint_dict["model"][key].items():
        #         print(k)
        self.cmp_mode = cmp_mode
        self.is_use_vq= kwargs.get("is_use_vq", True)
        vq_tc_config= kwargs.get('vq_tc_config',\
                                 {"model_path":"VQ_VAE_TC_ERA5/world_size4-VQ_VAE_TC_ERA5/checkpoint_best.pth",
                                  "ERA5_image_size":40,
                                  "ERA5_channels":69,
                                  "dim":128,
                                  "in_channel":2,
                                  "num_embeddings":1024,
                                  "embedding_dim":128,
                                  "beta":0.25,
                                  "n_head":4,
                                  "is_choose_channel":False,
                                  "choose_channel_hid_dim":128,
                                  "double_z": False,
                                  "ch_mult":[1,2,4,8],
                                  "Cross_att_qkv_mode":"Q_TC_info",
                                  "encoder_mode":"FPN",
                                  "decoder_mode":"Cross_atten",
                                  "dropout": 0.0
                                  })
        self.Cross_att_qkv_mode = vq_tc_config["Cross_att_qkv_mode"] #kwargs.get(vq_tc_config["Cross_att_qkv_mode"], "Q_TC_info")
        if vq_tc_config["Cross_att_qkv_mode"] == "Q_TC_info":
            patch_num=vq_tc_config["in_channel"]*4
        elif vq_tc_config["Cross_att_qkv_mode"] == "Q_img_token":
            patch_num=1166
        else:
            raise ValueError("no such Cross_att_qkv_mode")
        dim = vq_tc_config["dim"]
        
        self.encoder_mode = vq_tc_config["encoder_mode"]
        self.is_choose_channel=vq_tc_config["is_choose_channel"]
        self.VQ_TC = VQ_TC_Model_ERA5(
                                ERA5_image_size=vq_tc_config["ERA5_image_size"],
                                ERA5_channels=vq_tc_config["ERA5_channels"],
                                dim=vq_tc_config["dim"],
                                in_channel=vq_tc_config["in_channel"],
                                num_embeddings=vq_tc_config["num_embeddings"],
                                embedding_dim=vq_tc_config["embedding_dim"],
                                beta=vq_tc_config["beta"],
                                n_head=vq_tc_config["n_head"],
                                is_choose_channel=vq_tc_config["is_choose_channel"],
                                choose_channel_hid_dim=vq_tc_config["choose_channel_hid_dim"],
                                ch_mult=vq_tc_config["ch_mult"],
                                Cross_att_qkv_mode=vq_tc_config["Cross_att_qkv_mode"],
                                encoder_mode=vq_tc_config["encoder_mode"],
                                decoder_mode=vq_tc_config["decoder_mode"],
                                dropout=vq_tc_config["dropout"],
                                )
        
        if is_load_checkpoint:
            model_path=vq_tc_config["model_path"]
            self.checkpoint_ceph = checkpoint_ceph()
            self.load_checkpoint(model_path)
        if is_freeze_model:
            self.VQ_TC = self.freeze_model(self.VQ_TC, is_freeze_future)


        if self.is_choose_channel:
            self.ERA5_choose_channel_module = self.VQ_TC.ERA5_choose_channel_module
            self.future_ERA5_choose_channel_module = self.VQ_TC.future_ERA5_choose_channel_module


        self.tc_info_embedding = self.VQ_TC.tc_info_embedding

        self.ERA5_encoder = self.VQ_TC.ERA5_encoder
    
        self.cross_att = self.VQ_TC.cross_att
        if self.is_use_vq:
            self.vq_layer = self.VQ_TC.vq_layer

        self.future_img_encoder = self.VQ_TC.future_img_encoder
        if decoder_backboon=="Origin":
            self.tc_info_decoder = self.VQ_TC.tc_info_decoder
        
        del self.VQ_TC
        if backboon=="LSTM":
            backboon_config = kwargs.get('backboon_config',\
                                 {
                                  
                                  "hidden_size":2048,
                                  "num_layers":2,
                                  "input_length":4,
                                  "out_length":1,

                                  })
            
            hidden_size = backboon_config["hidden_size"]
            num_layers = backboon_config["num_layers"]
            input_length = backboon_config["input_length"]
            out_length = backboon_config["out_length"]

            input_size = dim*patch_num
            output_size= dim*patch_num
            self.h_state_iteration = hidden_state_LSTM(
                    input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size,\
                    input_length=input_length, out_length = out_length)
        elif backboon=="ST_transformer":
            backboon_config = kwargs.get('backboon_config',\
                                 {
                                  "input_length":4,
                                  
                                  "depth":3,
                                  "heads":4,
                                  "mlp_dim":128,
                                  "dropout":0.1,

                                  })
            input_length = backboon_config["input_length"]
            
            depth = backboon_config["depth"]
            heads = backboon_config["heads"]
            mlp_dim = backboon_config["mlp_dim"]
            dropout = backboon_config["dropout"]



            self.h_state_iteration = ST_transformer(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, inp_len=input_length, 
                                                    patch_num=patch_num, dropout=dropout)
        elif backboon=="future_transformer":
            backboon_config = kwargs.get('backboon_config',\
                                 {
                                  "input_length":4,
                                  
                                  "depth":3,
                                  "heads":4,
                                  "mlp_dim":256,
                                  "dropout":0.1,

                                  })
            input_length = backboon_config["input_length"]
            
            depth = backboon_config["depth"]
            heads = backboon_config["heads"]
            mlp_dim = backboon_config["mlp_dim"]
            dropout = backboon_config["dropout"]



            self.h_state_iteration = ST_transformer(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, inp_len=input_length, 
                                                    patch_num=patch_num, dropout=dropout)
        elif backboon=="SELF_transformer":
            backboon_config = kwargs.get('backboon_config',\
                                 {
                                  "input_length":4,
                                 
                                  "depth":3,
                                  "heads":4,
                                  "mlp_dim":256,
                                  "dropout":0.1,

                                  })
            input_length = backboon_config["input_length"]
            
            depth = backboon_config["depth"]
            heads = backboon_config["heads"]
            mlp_dim = backboon_config["mlp_dim"]
            dropout = backboon_config["dropout"]
            self.h_state_iteration = SELF_transformer(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, inp_len=input_length, 
                                                    patch_num=patch_num, dropout=dropout)
        else:
            raise ValueError(f"No {backboon} backboon")

        self.is_physics = is_physics
        
        if is_physics:
            
            physics_config= kwargs.get('physics_config',\
                                    {
                                    "is_cmp_PI":True,
                                    "ERA5_channels":69,
                                    "polyorder":2,
                                    "out_dim":1,
                                    "dim":128,
                                    "pre_dim":2,
                                    "gate_rate":0.5,
                                    })
            self.is_cmp_PI = physics_config["is_cmp_PI"]
            if self.is_cmp_PI:
                # from networks.physic_TC import cmp_TC_Potential_Intensity
              
                self.intensity_mean, self.intensity_std = self.get_intensity_meanstd()
               
                
                self.physics_model = physics_SINDy_module(in_channel=2, polyorder=2, pre_dim=2)
                self.Hid_Cross_Transformer = Cross_Transformer(dim=dim, depth=3, heads=4, mlp_dim=128, dropout=0.1)
               
            else:
                self.physics_model = physics_guid_module(era5_channel=physics_config["ERA5_channels"],
                                                        polyorder=physics_config["polyorder"],
                                                        out_dim=physics_config["out_dim"], dim=physics_config["dim"], 
                                                        pre_dim=physics_config["pre_dim"])
                
                self.gate_rate = nn.Parameter(torch.Tensor([physics_config["gate_rate"]]), requires_grad=True)
        

        self.is_hid_vq = is_hid_vq


    def get_intensity_meanstd(self):
        intensity_mean = np.array([49.040946869297855, 988.9592162921715])
        intensity_std = np.array([28.33988190313529,  21.844814128245602])
        return torch.from_numpy(np.array(intensity_mean)), torch.from_numpy(np.array(intensity_std))


    

    def hid_vq_loss(self, pre_hidden_state, quantized_hidden_state):
        commitment_loss = F.mse_loss(quantized_hidden_state.detach(), pre_hidden_state)
        embedding_loss  = F.mse_loss(quantized_hidden_state, pre_hidden_state.detach())
        
        vq_loss = commitment_loss * 0.25 + embedding_loss
        return vq_loss
    def forward(self, new_era5_data, inp_lable, fengwu_pre, pre_len=None, PI_data=None, pre_lable=None, is_train_mode=False):
        if self.cmp_mode=="Many_to_Many":
                
            """
        
            new_era5_data: [B, T, C, H, W]
            inp_lable: [B, T, C]
            fengwu_pre: [B, T_pre, C, H, W]

            
            """
            B, T, _, _, _ = new_era5_data.shape
            if pre_len==None:
                B, T_pre, _, _, _ = fengwu_pre.shape
            else:
                T_pre = pre_len
                fengwu_pre = fengwu_pre[:,:pre_len]
                if PI_data != None:
                    PI_data = PI_data[:,:pre_len]

            if self.is_physics:
                if self.is_cmp_PI:
                    physics_pre = PI_data[:,:,:2]
                    # print(physics_pre[0,0,:,0,0])
                    physics_pre = (physics_pre - self.intensity_mean.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(inp_lable.device)) / self.intensity_std.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(inp_lable.device)
                    
                    physics_hid_pre = self.physics_model(physics_pre)
                    # print(physics_pre.shape)
                else:
                    era5_data_last = new_era5_data[:, -1]
                    inp_lable_last = inp_lable[:, -1]
                    physics_pre = torch.zeros((B, T_pre, inp_lable.shape[-1]), device=inp_lable.device)# requires_grad=True
                    for i in range(T_pre):
                        era5_data_now = fengwu_pre[:, i]
                        physics_pre_t = self.physics_model(era5_data_last, era5_data_now, inp_lable_last)
                        physics_pre[:, i] = physics_pre_t
                        inp_lable_last = physics_pre_t
                        era5_data_last = fengwu_pre[:, i]
                
                    

           
            new_era5_data = torch.reshape(new_era5_data, (-1, new_era5_data.shape[2], new_era5_data.shape[3], new_era5_data.shape[4]))
            inp_lable_reshape = torch.reshape(inp_lable, (-1, inp_lable.shape[2]))
            tc_info_feature = self.tc_info_embedding(inp_lable_reshape)
            
            


            if self.is_choose_channel:
               
                new_era5_data = torch.permute(new_era5_data, (0, 2, 3, 1))
                new_era5_data = self.ERA5_choose_channel_module(new_era5_data)
                new_era5_data = torch.permute(new_era5_data, (0, 3, 1, 2))



            if self.encoder_mode=="Resnet":
                
                era5_feature = self.ERA5_encoder(new_era5_data).flatten(-2)
                
                img_token = era5_feature
                img_token = img_token.transpose(1, 2)
            elif self.encoder_mode=="FPN":
                
                era5_feature = self.ERA5_encoder(new_era5_data)
                
                img_token = era5_feature



            if self.Cross_att_qkv_mode == "Q_TC_info":
                hidden_state = self.cross_att(tc_info_feature, img_token)
            elif self.Cross_att_qkv_mode == "Q_img_token":
                hidden_state = self.cross_att(img_token, tc_info_feature)



            #hidden_state = self.cross_att(tc_info_feature, img_token) 
            if self.is_use_vq:
                quantized_hidden_state, vq_loss_label = self.vq_layer(hidden_state)
            else:
                vq_loss_label = None
                quantized_hidden_state = hidden_state
            quantized_hidden_state_copy = quantized_hidden_state
            # print(vq_loss.shape)
            #print(img_token.shape)
            # self.tc_info_decoder()
            quantized_hidden_state = torch.reshape(quantized_hidden_state, 
                                                   (B, T, quantized_hidden_state.shape[1], quantized_hidden_state.shape[2]))
           
            
            #pre_len = fengwu_pre.shape[1]
            quantized_hidden_state_pre_list = []

            for i in range(pre_len):
                # print(f"quantized_hidden_state:{quantized_hidden_state.shape}")
                quantized_hidden_state_pre = self.h_state_iteration(quantized_hidden_state)
                # print(f"quantized_hidden_state_pre:{quantized_hidden_state_pre.shape}")
                if self.is_physics and self.is_cmp_PI:
                    quantized_hidden_state_pre = self.Hid_Cross_Transformer(quantized_hidden_state_pre, physics_hid_pre[:, i])
                if self.hid_residual:
                    quantized_hidden_state_pre = quantized_hidden_state_pre + quantized_hidden_state[:, -1]

                    # print(f"quantized_hidden_state_pre:{quantized_hidden_state_pre.shape}")



                quantized_hidden_state_pre = torch.unsqueeze(quantized_hidden_state_pre, dim=1)
                quantized_hidden_state_pre_list.append(quantized_hidden_state_pre)
                #print(quantized_hidden_state_pre.shape)
                quantized_hidden_state = torch.concat((quantized_hidden_state, quantized_hidden_state_pre), dim=1)
                #print(quantized_hidden_state.shape)
                quantized_hidden_state = quantized_hidden_state[:, 1:]

                #print(quantized_hidden_state.shape)
            # print(quantized_hidden_state_pre.shape)
            quantized_hidden_state = torch.cat(quantized_hidden_state_pre_list, dim=1)
            quantized_hidden_state = torch.reshape(quantized_hidden_state, (-1, quantized_hidden_state.shape[-2], quantized_hidden_state.shape[-1]))
            if self.is_hid_vq and is_train_mode and self.is_use_vq:
                
                fengwu_pre_reshape = torch.reshape(fengwu_pre, (-1, fengwu_pre.shape[2], fengwu_pre.shape[3], fengwu_pre.shape[4]))
                pre_lable_reshape  = torch.reshape(pre_lable, (-1, pre_lable.shape[2]))
                pre_lable_feature = self.tc_info_embedding(pre_lable_reshape)


                if self.encoder_mode=="Resnet":
                    
                    era5_feature_pre = self.ERA5_encoder(fengwu_pre_reshape).flatten(-2)
                    img_token_pre = era5_feature_pre
                    img_token_pre = img_token_pre.transpose(1, 2)
                elif self.encoder_mode=="FPN":
                    
                    era5_feature_pre = self.ERA5_encoder(fengwu_pre_reshape)
                    img_token_pre = era5_feature_pre
                if self.Cross_att_qkv_mode == "Q_TC_info":
                    hidden_state_pre = self.cross_att(pre_lable_feature, img_token_pre)
                elif self.Cross_att_qkv_mode == "Q_img_token":
                    hidden_state_pre = self.cross_att(img_token_pre, pre_lable_feature)
                #hidden_state = self.cross_att(tc_info_feature, img_token) 
            
                quantized_hidden_state_label, vq_loss_label = self.vq_layer(hidden_state_pre)

                # print(quantized_hidden_state.shape, quantized_hidden_state_label.shape)
                vq_loss_label = self.hid_vq_loss(quantized_hidden_state, quantized_hidden_state_label)
                # print(vq_loss_label.shape)


            

            fengwu_pre = torch.reshape(fengwu_pre, (-1, fengwu_pre.shape[2], fengwu_pre.shape[3], fengwu_pre.shape[4]))
            
            if self.is_choose_channel:
                fengwu_pre = torch.permute(fengwu_pre, (0, 2, 3, 1))
                new_era5_data_future = self.future_ERA5_choose_channel_module(fengwu_pre)
                new_era5_data_future = torch.permute(new_era5_data_future, (0, 3, 1, 2))
                # self.future_ERA5_choose_channel_module.plt_mask_channel.plt_pic("era5_future")
                # self.future_ERA5_choose_channel_module.plt_origin_channel.plt_pic("era5_future_ori")
            else:
                new_era5_data_future = fengwu_pre

            if self.encoder_mode=="Resnet":
                era5_feature_pre = self.future_img_encoder(new_era5_data_future).flatten(-2)
                era5_feature_pre = era5_feature_pre.transpose(1, 2)
            elif self.encoder_mode=="FPN":
                era5_feature_pre = self.future_img_encoder(new_era5_data_future)


            # print(quantized_hidden_state.shape)
            # print(era5_feature_pre.shape)
            if self.decoder_backboon == "Origin":
                tc_info_out = self.tc_info_decoder(quantized_hidden_state, era5_feature_pre)
                tc_info_out = torch.reshape(tc_info_out, (B, T_pre, tc_info_out.shape[-1]))
           

            # print(tc_info_out.shape)
        
            if self.residual:
            
                tc_info_out = tc_info_out + inp_lable[:, -1, :].unsqueeze(1)
            quantized_hidden_state_copy = torch.reshape(quantized_hidden_state_copy, (quantized_hidden_state_copy.shape[0], -1))
            
            
            return tc_info_out, vq_loss_label, quantized_hidden_state_copy


    def load_checkpoint(self, checkpoint_path, load_model=True, ):
        
        checkpoint_dict = self.checkpoint_ceph.load_checkpoint(checkpoint_path)
        if checkpoint_dict is None:
            self.logger.info("checkpoint is not exist")
            return
        
        checkpoint_model = checkpoint_dict['model']
        print(f"load epoch {checkpoint_dict['epoch']}")
        if load_model:
            for key in checkpoint_model:
                new_state_dict = OrderedDict()
                for k, v in checkpoint_model[key].items():
                    if "module" == k[:6]:
                        name = k[7:]
                    else:
                        name = k
                    
                    new_state_dict[name] = v

                self.VQ_TC.load_state_dict(new_state_dict, strict=False)
                # self.model[key].load_state_dict(checkpoint_model[key])


    def freeze_model(self, model, is_freeze_future=True):

        for (name, param) in model.named_parameters():
            module_name = name.split(".")[0]
            
            if (module_name=="future_img_encoder" or module_name=="tc_info_decoder") and (not is_freeze_future):
                # print("**********************************************************")
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model



class channel_choose_module(nn.Module):
    def __init__(self, inp_dim=1, hid_dim=16,) -> None:
        super().__init__()
        
        self.proj1 = nn.Linear(inp_dim, hid_dim)
        self.elu = nn.ELU()
        self.proj2 = nn.Linear(hid_dim, inp_dim)
        # self.plt_mask_channel = plt_mask_channel(inp_dim)
        # self.plt_origin_channel = plt_mask_channel(inp_dim)
    def forward(self, x):
        x_ori = x
        x = self.proj1(x)
        x = self.elu(x)
        x = self.proj2(x)
        x = self.elu(x)
        # self.plt_mask_channel.add_mask(x)
        # self.plt_origin_channel.add_mask(x_ori)
        x = torch.mul(x, x_ori)
        return x

