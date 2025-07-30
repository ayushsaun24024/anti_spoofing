import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Union
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        x = self.input_drop(x)

        att_map = self._derive_att_map(x)

        x = self._project(x, att_map)

        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)

        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        
        x = self.input_drop(x)

        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)

        att_map = torch.tanh(self.att_proj(att_map))
        
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h




class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        return out


class AASIST(nn.Module):
    def __init__(self, temperatures):
        super().__init__()
        
        # filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        filts = [256, [1, 32], [32, 32], [32, 64], [64, 64]] # matching original dimension in paper
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  temperatures


        # self.LL = nn.Linear(1024, 128)
        self.LL = nn.Linear(1024, 256) # matching original dimension in paper

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
            
        )

        # self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.pos_S = nn.Parameter(torch.randn(1, 85, filts[-1][-1]))  # matching original dimension in paper
        
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        x = self.LL(x)
        
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)
        
        w1 = F.softmax(w,dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S 
        
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        
        w2 = F.softmax(w,dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
     
        e_T = m1.transpose(1, 2)
       
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        return output

class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=4, width=26, stride=1):
        super(Res2NetBlock, self).__init__()
        self.scale = scale
        self.width = width
        
        self.conv1 = nn.Conv1d(in_channels, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        
        self.convs = nn.ModuleList([])
        for i in range(scale - 1):
            self.convs.append(nn.Conv1d(width, width, kernel_size=3, padding=1, bias=False))
            
        self.conv3 = nn.Conv1d(width * scale, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, dim=1)
        sp = []
        
        for i in range(self.scale):
            if i == 0:
                sp.append(spx[i])
            elif i == 1:
                sp.append(self.relu(self.convs[i-1](spx[i])))
            else:
                sp.append(self.relu(self.convs[i-1](spx[i] + sp[i-1])))
        
        out = torch.cat(sp, dim=1)
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class LightRes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=4, width=26):
        super(LightRes2NetBlock, self).__init__()
        self.scale = scale
        self.width = width
        
        self.conv1 = nn.Conv1d(in_channels, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        
        self.depthwise_convs = nn.ModuleList([])
        self.pointwise_convs = nn.ModuleList([])
        for i in range(scale - 1):
            self.depthwise_convs.append(
                nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width, bias=False)
            )
            self.pointwise_convs.append(
                nn.Conv1d(width, width, kernel_size=1, bias=False)
            )
            
        self.conv3 = nn.Conv1d(width * scale, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, dim=1)
        sp = []
        
        for i in range(self.scale):
            if i == 0:
                sp.append(spx[i])
            elif i == 1:
                temp = self.depthwise_convs[i-1](spx[i])
                temp = self.pointwise_convs[i-1](temp)
                sp.append(self.relu(temp))
            else:
                temp = self.depthwise_convs[i-1](spx[i] + sp[i-1])
                temp = self.pointwise_convs[i-1](temp)
                sp.append(self.relu(temp))
        
        out = torch.cat(sp, dim=1)
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1)
        
        return x * attention

class DynamicConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DynamicConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.base_conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        
        self.dynamic_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        base_output = self.base_conv(x)
        
        dynamic_scaling = self.dynamic_mlp(x)
        dynamic_scaling = dynamic_scaling.unsqueeze(-1)
        
        output = base_output * dynamic_scaling
        
        return output

class MFARes2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=4, width=32):
        super(MFARes2NetBlock, self).__init__()
        
        self.light_res2net = LightRes2NetBlock(in_channels, out_channels, scale, width)
        
        self.lff = ChannelAttention(out_channels)
        
        self.dynamic_conv = DynamicConvolution(in_channels, out_channels)
        
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        res2net_out = self.light_res2net(x)
        
        lff_out = self.lff(res2net_out)
        
        dynamic_out = self.dynamic_conv(x)
        
        fused = torch.cat([lff_out, dynamic_out], dim=1)
        output = self.fusion(fused)
        
        return output

class GlobalFeatureFusion(nn.Module):
    def __init__(self, num_blocks, channels):
        super(GlobalFeatureFusion, self).__init__()
        self.num_blocks = num_blocks
        
        self.fusion_weights = nn.Parameter(torch.ones(num_blocks))
        
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=1, bias=False) 
            for _ in range(num_blocks)
        ])
        
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, block_outputs):
        weights = self.softmax(self.fusion_weights)
        
        fused_output = 0
        for i, (output, conv) in enumerate(zip(block_outputs, self.conv_blocks)):
            fused_output += weights[i] * conv(output)
        
        return fused_output

class MFARes2Net(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2, num_blocks=4, channels=256, scale=4, width=64):
        super(MFARes2Net, self).__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.mfa_blocks = nn.ModuleList([
            MFARes2NetBlock(channels, channels, scale, width)
            for _ in range(num_blocks)
        ])
        
        self.gff = GlobalFeatureFusion(num_blocks, channels)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        
        x = self.input_proj(x)
        
        x = x.transpose(1, 2)
        
        block_outputs = []
        current_x = x
        for block in self.mfa_blocks:
            current_x = block(current_x)
            block_outputs.append(current_x)
        
        fused = self.gff(block_outputs)
        
        pooled = self.global_pool(fused).squeeze(-1)
        
        output = self.classifier(pooled)
        
        return output

class ENSEMBLE_MODEL(nn.Module):
    def __init__(self, num_classes=2):
        super(ENSEMBLE_MODEL, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
        wav2vec_hidden_size = self.wav2vec2.config.hidden_size
        
        # Truncate to use only 15 transformer layers
        self.wav2vec2.encoder.layers = self.wav2vec2.encoder.layers[:15]

        # Freeze first 9 layers, fine-tune remaining 6 layers
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < 9:  # Freeze first 9 layers
                for param in layer.parameters():
                    param.requires_grad = False
            else:  # Fine-tune layers 9-14 (remaining 6 layers)
                for param in layer.parameters():
                    param.requires_grad = True

        # Keep feature extractor and position embeddings frozen
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False

        self.aasist_1 = AASIST(temperatures=[2.0, 2.0, 100.0])
        self.aasist_2 = AASIST(temperatures=[1.5, 1.5, 50.0])   
        self.aasist_3 = AASIST(temperatures=[3.0, 3.0, 150.0])
        self.aasist_4 = AASIST(temperatures=[2.5, 1.5, 75.0])
        
        self.mfa_res2net_1 = MFARes2Net(wav2vec_hidden_size, num_classes, 4, 256, 4, 64)
        self.mfa_res2net_2 = MFARes2Net(wav2vec_hidden_size, num_classes, 6, 320, 8, 80)
        self.mfa_res2net_3 = MFARes2Net(wav2vec_hidden_size, num_classes, 3, 192, 4, 48)
        self.mfa_res2net_4 = MFARes2Net(wav2vec_hidden_size, num_classes, 5, 384, 6, 96)
        
        self.lfcc_extractor = torchaudio.transforms.LFCC(
            sample_rate=16000, n_filter=60, n_lfcc=60,
            speckwargs={'n_fft': 512, 'win_length': 400, 'hop_length': 160, 'center': False}
        )
        
        self.lfcc_align_1 = nn.Sequential(
            nn.Conv1d(60, 128, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm1d(128), nn.ReLU(inplace=True)               
        )
        self.lfcc_align_2 = nn.Sequential(
            nn.Conv1d(60, 128, kernel_size=5, stride=2, padding=2),  
            nn.BatchNorm1d(128), nn.ReLU(inplace=True)               
        )
        self.lfcc_align_3 = nn.Sequential(
            nn.Conv1d(60, 128, kernel_size=7, stride=2, padding=3),  
            nn.BatchNorm1d(128), nn.ReLU(inplace=True)               
        )
        self.lfcc_align_4 = nn.Sequential(
            nn.Conv1d(60, 128, kernel_size=3, stride=1, padding=1),  
            nn.MaxPool1d(2), nn.BatchNorm1d(128), nn.ReLU(inplace=True)  
        )
        
        self.lfcc_mfa_1 = MFARes2Net(1152, num_classes, 4, 256, 4, 64)
        self.lfcc_mfa_2 = MFARes2Net(1152, num_classes, 6, 320, 8, 80)
        self.lfcc_mfa_3 = MFARes2Net(1152, num_classes, 3, 192, 4, 48)
        self.lfcc_mfa_4 = MFARes2Net(1152, num_classes, 5, 384, 6, 96)
        
        # Keep your score calibrations
        self.score_calibrations = nn.ModuleList([
            nn.Linear(2, 2) for _ in range(12)
        ])
        
        weights = torch.tensor([0.12, 0.10, 0.11, 0.09, 0.08, 0.09, 0.08, 0.07, 0.09, 0.08, 0.09, 0.10])
        self.register_buffer('fusion_weights', weights / weights.sum())
        
        self.final_calibration = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2)
        )
        
    def extract_wav2vec_features(self, audio):
        # Remove torch.no_grad() since we want gradients for fine-tuning
        wav2vec_output = self.wav2vec2(audio, output_hidden_states=True)
        # Use the final layer (layer 14, 0-indexed) instead of layer 5
        features = wav2vec_output.hidden_states[-1]  # or wav2vec_output.last_hidden_state
        return features
    
    def extract_lfcc_features(self, audio, align_module):
        lfcc = self.lfcc_extractor(audio)
        lfcc_aligned = align_module(lfcc)
        return lfcc_aligned.transpose(1, 2)
    
    def temporal_alignment(self, wav2vec_features, lfcc_features):
        if wav2vec_features.size(1) != lfcc_features.size(1):
            lfcc_features = F.adaptive_avg_pool1d(
                lfcc_features.transpose(1, 2), wav2vec_features.size(1)
            ).transpose(1, 2)
        return lfcc_features
    
    def forward(self, audio):
        wav2vec_features = self.extract_wav2vec_features(audio)
        
        subsystem_outputs = []
        
        subsystem_outputs.append(self.aasist_1(wav2vec_features))
        subsystem_outputs.append(self.aasist_2(wav2vec_features))
        subsystem_outputs.append(self.aasist_3(wav2vec_features))
        subsystem_outputs.append(self.aasist_4(wav2vec_features))
        
        subsystem_outputs.append(self.mfa_res2net_1(wav2vec_features))
        subsystem_outputs.append(self.mfa_res2net_2(wav2vec_features))
        subsystem_outputs.append(self.mfa_res2net_3(wav2vec_features))
        subsystem_outputs.append(self.mfa_res2net_4(wav2vec_features))
        
        lfcc_1 = self.extract_lfcc_features(audio, self.lfcc_align_1)
        lfcc_1 = self.temporal_alignment(wav2vec_features, lfcc_1)
        combined_1 = torch.cat([wav2vec_features, lfcc_1], dim=2)
        subsystem_outputs.append(self.lfcc_mfa_1(combined_1))
        
        lfcc_2 = self.extract_lfcc_features(audio, self.lfcc_align_2)
        lfcc_2 = self.temporal_alignment(wav2vec_features, lfcc_2)
        combined_2 = torch.cat([wav2vec_features, lfcc_2], dim=2)
        subsystem_outputs.append(self.lfcc_mfa_2(combined_2))
        
        lfcc_3 = self.extract_lfcc_features(audio, self.lfcc_align_3)
        lfcc_3 = self.temporal_alignment(wav2vec_features, lfcc_3)
        combined_3 = torch.cat([wav2vec_features, lfcc_3], dim=2)
        subsystem_outputs.append(self.lfcc_mfa_3(combined_3))
        
        lfcc_4 = self.extract_lfcc_features(audio, self.lfcc_align_4)
        lfcc_4 = self.temporal_alignment(wav2vec_features, lfcc_4)
        combined_4 = torch.cat([wav2vec_features, lfcc_4], dim=2)
        subsystem_outputs.append(self.lfcc_mfa_4(combined_4))
        
        calibrated_outputs = []
        for i, output in enumerate(subsystem_outputs):
            calibrated_output = self.score_calibrations[i](output)
            calibrated_outputs.append(calibrated_output)
        
        fused_logits = sum(
            weight * output for weight, output in zip(self.fusion_weights, calibrated_outputs)
        )
        
        final_output = self.final_calibration(fused_logits)
        
        return final_output
