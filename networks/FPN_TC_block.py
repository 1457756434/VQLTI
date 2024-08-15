


import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
import math

import torch.nn.functional as F

#  nn.SyncBatchNorm

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes=64, kernel_size=3, stride=1, padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.SyncBatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.SyncBatchNorm(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out




class FPN_mudule(nn.Module):
    def __init__(self, in_chancel, block, layers, patch_size=5, dim=128, FPN_mode='ERA5'):
        super(FPN_mudule, self).__init__()
        # self.inplanes = 64
        patch_H, patch_W = pair(patch_size)
        self.FPN_mode = FPN_mode
        if FPN_mode == "ERA5":
            ERA5_chancel = in_chancel
            self.ERA5_conv1 = nn.Conv2d(ERA5_chancel, 128, kernel_size=7, stride=1, padding=3, bias=False)
            self.ERA5_bn1 = nn.SyncBatchNorm(128)
   
            #self.ERA5_layer = self._make_layer(block, 128, 128, 1, kernel_size=6, stride=1, padding=0, downsample_kernel_size=6, down_stride=1)
            self.ERA5_layer1 = self._make_layer(block, 128, 64, layers[0])
            self.ERA5_layer2 = self._make_layer(block, 128, 128, layers[1], stride=2, down_stride=2)
            self.ERA5_layer3 = self._make_layer(block, 256, 128, layers[2], stride=2, down_stride=2)
        # Top layer
        # fussion
        self.fussion = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.lateral_layer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer2 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer3 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        


        fussion_patch_dim = 256 * patch_H * patch_W
        self.to_patch_p2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_p3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_p4 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_p5 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, planes, blocks, kernel_size=3, stride=1, padding=1, downsample_kernel_size=1, down_stride=1):
        downsample  = None
        if stride != 1 or inplanes != block.expansion * planes:
            downsample  = nn.Sequential(
                nn.Conv2d(inplanes, block.expansion * planes, kernel_size=downsample_kernel_size, stride=down_stride,bias=False),
                #nn.SyncBatchNorm(block.expansion * planes)
            )
        layers = []
        layers.append(block(inplanes, planes, kernel_size, stride,  padding, downsample))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, padding=padding))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, img):
        if self.FPN_mode == "ERA5":
            img = self.ERA5_conv1(img)
            img = self.ERA5_bn1(img)
            c1 = img
            # print("c1:{}".format(c1.shape))
            c2 = self.ERA5_layer1(c1)
            # print("c2:{}".format(c2.shape))
            c3 = self.ERA5_layer2(c2)
            # print("c3:{}".format(c3.shape))
            c4 = self.ERA5_layer3(c3)
            # print("c4:{}".format(c4.shape))
            c5 = c4
        #print("IR_c5:{}".format(IR_c5.shape))
        p5 = self.fussion(c5)
        #print("IR_p5:{}".format(IR_p5.shape))
        p4 = self._upsample_add(p5, self.lateral_layer1(c4))
        p4 = self.smooth1(p4)
        #print("IR_p4:{}".format(IR_p4.shape))
        p3 = self._upsample_add(p4, self.lateral_layer2(c3))
        p3 = self.smooth2(p3)
        # print("IR_p3:{}".format(IR_p3.shape))
        p2 = self._upsample_add(p3, self.lateral_layer3(c2))
        p2 = self.smooth3(p2)
        #print("IR_p2:{}".format(IR_p2.shape))
      
        
        p2 = self.to_patch_p2(p2)
        p3 = self.to_patch_p3(p3)
        p4 = self.to_patch_p4(p4)
        p5 = self.to_patch_p5(p5)

        
       
        fussion_token = torch.concat((p2, p3, p4, p5), dim=1)
      

        return  fussion_token



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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




class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=256, dropout=0.) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        
        project_out = not (heads ==1 and dim_head == dim)

        self.heads =heads
        self.scale = dim_head ** -0.5

        #self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        self.qlv_embedding = nn.Linear(dim, heads*dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    
    def forward(self, img_embedding, seq_embedding):
        # x = self.norm(x)
        q = self.qlv_embedding(seq_embedding)
        k = self.qlv_embedding(img_embedding)
        v = self.qlv_embedding(img_embedding)
      

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
       
        return self.to_out(out)



class Cross_Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim=dim, hid_dim=mlp_dim, dropout=dropout),
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim=dim, hid_dim=mlp_dim, dropout=dropout),
            ]))
            
        self.last_att = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.last_feed = FeedForward(dim=dim, hid_dim=mlp_dim, dropout=dropout)
            
                
    def forward(self, img_embedding, seq_embedding):
        for att_seq, feed_seq, att_img, feed_img, in self.layers:
            seq_embedding = att_seq(img_embedding, seq_embedding) + seq_embedding
            seq_embedding = feed_seq(seq_embedding) + seq_embedding
            img_embedding = att_img(seq_embedding, img_embedding, ) + img_embedding
            img_embedding = feed_img(img_embedding) + img_embedding
            #print(seq_embedding.shape, img_embedding.shape)

        seq_embedding = self.last_att(img_embedding, seq_embedding) + seq_embedding
        seq_embedding = self.last_feed(seq_embedding) + seq_embedding 

        return self.norm(seq_embedding)

class ST_transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.) -> None:
        super().__init__()
        self.Cross_Transformer = Cross_Transformer(dim, depth, heads, dim, mlp_dim, dropout)
    
    def forward(self, img_embedding, seq_embedding):
        """
        img_embedding: [B, T, n, dim]
        seq_embedding: [B, T, 1, dim]
        """
        B, T, n, dim = img_embedding.shape
        
        fusion_embedding = torch.concat((img_embedding[:, 0, :, :], seq_embedding[:, 0, :, :]), dim=-2)
        # print(f"fusion_embedding: {fusion_embedding.shape}")
        att_score = self.Cross_Transformer(fusion_embedding, seq_embedding[:, 0, :, :])
        # print(f"att_score: {att_score.shape}")
        for t in range(1, T):
            for t_of_t in range(0, t+1):
                fusion_embedding = torch.concat((img_embedding[:, t_of_t, :, :], seq_embedding[:, t_of_t, :, :], seq_embedding[:, t, :, :]), dim=-2)
                # print(f"fusion_embedding: {fusion_embedding.shape}")
                att_score_t = self.Cross_Transformer(img_embedding[:, t_of_t, :, :], seq_embedding[:, t, :, :])
                att_score = torch.concat((att_score, att_score_t), dim=-2)
        # print(f"att_score: {att_score.shape}")
        return att_score
    



class AR_decoder(nn.Module):
    def __init__(self,dim, depth, heads, mlp_dim, seq_len, pre_len, hid_dim , seq_dim=2, dropout=0.) -> None:
        super().__init__()
        self.Cross_Transformer = Cross_Transformer(dim, depth, heads, dim, mlp_dim, dropout)
        self.seq_att_num = int((1 + seq_len)*seq_len/2)
        self.pre_len = pre_len
        self.Pre_Head = nn.Sequential(
            nn.LayerNorm(dim*self.seq_att_num),
            nn.Linear(dim*self.seq_att_num, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, seq_dim),
        )
        self.seq_embedding = nn.Linear(seq_dim, dim)
    def forward(self, hid_state, final_seq, step):

        
        final_seq_embedding = self.seq_embedding(final_seq)
        final_seq_embedding = torch.unsqueeze(final_seq_embedding, dim=1)
        final_seq_embedding = torch.concat((final_seq_embedding, hid_state), dim=-2)
        #print(f"final_seq_embedding {final_seq_embedding.shape}")
        hid_state = self.Cross_Transformer(final_seq_embedding, hid_state)

        hid_state_pre = torch.reshape(hid_state, (hid_state.shape[0], -1))
        pre_seq = self.Pre_Head(hid_state_pre) + final_seq
        
        final_seq = pre_seq
        
        for _ in range(1, step):
            #save_fig(hid_state[0], f"hid_state{step}")
            final_seq_embedding = self.seq_embedding(final_seq)
            final_seq_embedding = torch.unsqueeze(final_seq_embedding, dim=1)
            final_seq_embedding = torch.concat((final_seq_embedding, hid_state), dim=-2)
            # print(final_seq_embedding.shape)
            # print(hid_state.shape)
            hid_state = self.Cross_Transformer(final_seq_embedding, hid_state)
            hid_state_pre = torch.reshape(hid_state, (hid_state.shape[0], -1))
            pre_seq_t = self.Pre_Head(hid_state_pre) + final_seq
            
            final_seq = pre_seq_t
        final_seq = torch.unsqueeze(final_seq, dim=1)
        return final_seq


    
class FPN_TC_block(nn.Module):
    def __init__(self, FPN_mode='ERA5', patch_size=(5, 5), in_chancel=1, dim=512,
                ) -> None:
        super().__init__()
       
      
        patch_H, patch_W = pair(patch_size)

        self.patch_H = patch_H
        self.patch_W = patch_W
       
        self.FPN = FPN_mudule(in_chancel=in_chancel, patch_size=patch_size, dim=dim, block=Bottleneck, layers=[2,2,2], FPN_mode=FPN_mode)
       
    def forward(self, img):

        img = torch.reshape(img, (-1, img.shape[-3],  img.shape[-2], img.shape[-1]))
        
        fussion_token = self.FPN(img)
        
      
        return fussion_token
    



