import torch
import numpy as np
import torch.nn as nn
from torch.nn import LayerNorm
import math
import inspect
import torch.nn.functional as F
from dataclasses import dataclass
import h5py
from torchsummary import summary


# @dataclass
class Config:

    def __init__(self, block_size, v_size, n_layer, n_head, n_embd, n_embd_temp, dropout, bias, dtype, batch_size, n_r,
                 n_z,max_input_len,max_rz_len):
        self.block_size = block_size
        self.v_size = v_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_embd_temp = n_embd_temp
        self.dropout = dropout
        self.bias = bias
        self.dtype = dtype
        self.batch_size = batch_size
        self.n_r = n_r
        self.n_z = n_z
        self.max_input_len = max_input_len
        self.max_rz_len = max_rz_len

def region_embedding_func(region_raw, config):
    # reg_embd = torch.from_numpy(region_raw).cuda(0)
    reg_embd = region_raw
    # 补零
    # reg_embd = F.pad(reg_embd, (0, config.n_embd - reg_embd.size(0)))
    # 补全tensor
    # reg_embd = reg_embd.repeat(config.v_size, 1)
    # 复制张量以适应新的维度
    # reg_embd_expanded = reg_embd.unsqueeze(0).repeat(config.batch_size, 1, 1)
    # reg_embd_expanded = reg_embd_expanded.unsqueeze(1)
    return reg_embd


# 进行Position embedding,将data_raw中的position文件读入，拍平并生成对应的Position embedding
def position_embedding_func(position_raw, config):
    position_raw[position_raw > 1] = 1
    # pos_embd = torch.from_numpy(position_raw).cuda(0)
    pos_embd = position_raw
    # 补零
    # padding_amount = (config.n_embd - pos_embd.size(1), 0)
    # pos_embd = F.pad(pos_embd, pad=padding_amount, mode='constant', value=0)
    # 计算需要复制的次数以达到至少40行
    # replication_times = (config.v_size // pos_embd.size(0)) + (1 if config.v_size % pos_embd.size(0) != 0 else 0)
    # 复制张量
    # repeated_tensor = pos_embd.repeat(replication_times, 1)
    # 计算还需补多少行才能达到40行config.v_size
    # remaining_rows = config.v_size - repeated_tensor.size(0)
    # 如果需要，创建剩余的全零行张量并拼接到已复制的张量后面
    # if remaining_rows > 0:
    #     zero_tensor = torch.zeros((remaining_rows, config.n_embd))
    #     padded_tensor = torch.cat((repeated_tensor, zero_tensor), dim=0)
    # else:
    #     padded_tensor = repeated_tensor
    # assert padded_tensor.size() == (config.v_size, config.n_embd), "Final shape should be (40, 1152)"

    return pos_embd


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim).float())
        self.bias = nn.Parameter(torch.zeros(ndim).float()) if bias else None

    def forward(self, input):
        if input.dtype == torch.float64:
            input = input.float()
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class DynamicLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(DynamicLayerNorm, self).__init__()
        self.eps = eps
        self.sgmoid = nn.Sigmoid()
        # 注意：不预先定义weight和bias，因为它们的大小取决于输入的特征维度
    def forward(self, input_tensor):
        # 确保weight和bias在与input_tensor相同的设备上创建
        device = input_tensor.device
        feature_dim_size = input_tensor.shape[-1]
        weight = nn.Parameter(torch.ones(feature_dim_size, device=device), requires_grad=True)
        bias = nn.Parameter(torch.zeros(feature_dim_size, device=device), requires_grad=True)
        mean = input_tensor.mean(dim=-1, keepdim=True)
        var = input_tensor.var(dim=-1, unbiased=False, keepdim=True)
        output = (input_tensor - mean) / torch.sqrt(var + self.eps)
        output = output * weight + bias
        return self.sgmoid(output)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Conv_embd(nn.Module):
# todo 模型初始化有问题
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device  # Explicitly initialize device
        self.sigmoid = nn.Sigmoid()  # Corrected typo in the activation function name
        self.conv = None
    def _initialize_conv(self, out_channels):
        # 根据需要的out_channels动态初始化卷积层
        if self.conv is None or self.conv.out_channels != out_channels:
            self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=1)
    def forward(self, x,out_channels=None):
        if out_channels is not None:
            self._initialize_conv(out_channels)
        x = self.conv(x).transpose(0, 1)
        output_tensor = self.sigmoid(x)
        # todo 补零
        return output_tensor


class Encoder_Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Out_head(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias, dtype=torch.float64)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.float64)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def custom_padding(input_tensor, target_shape, pad_value=0):
    """
    将输入张量padding到指定的形状。
    参数:
    - input_tensor: 输入的张量，任意形状。
    - target_shape: 目标形状，如torch.Size([100, 10000])。
    - pad_value: 用于填充的值，默认为0。
    返回:
    - padded_tensor: 经过padding处理后符合目标形状的张量。
    """
    # 获取输入和目标形状的尺寸
    # input_shape = input_tensor.shape
    # target_rows, target_cols = target_shape
    # # 处理行维度的扩展
    # num_repeats = (target_rows // input_shape[0]) + 1  # 计算重复次数以接近但不超过目标行数
    # repeated_tensor = torch.cat([input_tensor] * num_repeats)[:target_rows]  # 重复并裁剪到目标行数
    # # 处理列维度的padding
    # current_cols = repeated_tensor.shape[1]
    # pad_width = (0, target_cols - current_cols)  # 定义在列方向上的padding宽度
    # padded_tensor = F.pad(repeated_tensor, pad_width, "constant", pad_value)  # 使用指定的值进行padding
    # 创建一个全0的目标形状张量
    padded_tensor = torch.zeros(target_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    # 计算原张量可以放入新张量的起始位置，假设填充在左上角
    start_row = 0
    start_col = 0
    # 假设输入张量的形状能够完全放入目标形状内
    end_row = min(input_tensor.shape[0], target_shape[0])
    end_col = min(input_tensor.shape[1], target_shape[1])
    # 将原张量数据复制到新张量对应位置
    padded_tensor[start_row:end_row, start_col:end_col] = input_tensor[:end_row, :end_col]
    return padded_tensor


class Onion(nn.Module):
    def __init__(self, config):
        super(Onion, self).__init__()
        self.config = config
        self.block_stack = nn.ModuleList([Encoder_Block(config) for _ in range(config.n_layer)])
        self.upsample = Conv_embd()# todo 不固定
        self.LN = DynamicLayerNorm()
        self.out_head = Out_head(config)

    def _init_weihts(self):
        # 实现权重初始化逻辑
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # 举例使用Xavier初始化，根据需要调整
            elif 'bias' in name and param is not None:
                nn.init.zeros_(param)

    def forward(self, x):
        tok_embd_list = []
        for i in range(self.config.batch_size):
            self.config.n_r = x[i,:,(int(x[i,:,-1])-2)] # train 测试时用
            self. config.n_z = x[i,:,(int(x[i,:,-1])-1)]
            output_channals= (self.config.n_r * self.config.n_z)
            self.config.v_size = x[i, :, :(int(x[i,:,-1])-3)].shape[1]# N的长度
            # 上采样之前做LN
            input_LN = self.LN(x[i, :, :(int(x[i,:,-1])-3)])
            # todo upsample不固定，修改input_LN，有问题
            self.upsample._initialize_conv(int(output_channals.item()))
            self.upsample.to("cuda")
            tok_embd_i = self.upsample(input_LN)
            target_shape = torch.Size([100, self.config.n_embd])
            padded_x = custom_padding(tok_embd_i, target_shape)
            # todo BatchNorm/LayerNorm/MidNorm
            index = x[i, :, int(x[i,:,-1])-3].to(torch.long)

            # self.config.n_r = x[i, :, -2]# model 测试时用
            # self.config.n_z = x[i, :, -1]
            # output_channals= (self.config.n_r * self.config.n_z)
            # self.config.v_size = x[i, :, :-3].shape[1]
            # # 上采样之前做LN
            # input_LN = self.LN(x[i, :, :-3])
            # # 根据输入做上采样，输出维度为 m = n_r*n_z
            # self.upsample._initialize_conv(int(output_channals.item()))
            # self.upsample.to("cuda")
            # tok_embd_i = self.upsample(input_LN)
            # # 将测量通道数padding到100，m padding到10000
            # target_shape = torch.Size([100, self.config.n_embd])
            # padded_x = custom_padding(tok_embd_i, target_shape)
            # # todo BatchNorm/LayerNorm/MidNorm
            # index = x[i, :, -3].to(torch.long)
            # posi = torch.tensor(self.config.posi).to('cuda')[index][0]

            posi = torch.tensor(self.config.posi[index]).to('cuda')
            regi = torch.tensor(self.config.regi[index].flatten()).to('cuda')
            # .cpu().detach().numpy()
            # 将regi重复，扩充到和初始测量通道数相同
            regi_repeat = regi.unsqueeze(0).repeat(self.config.v_size, 1)
            # 将两个位置编码补零至100*10000
            padded_posi = custom_padding(posi, target_shape)
            padded_regi = custom_padding(regi_repeat, target_shape)
            #todo 简单的相加是不是不好，堆叠成3通道？
            #todo posi加在attention之后
            padded_x = padded_x + padded_posi + padded_regi
            tok_embd_list.append(padded_x)
        tok_embd = torch.stack(tok_embd_list)
        # config.n_embd = len(tok_embd)
        for block in self.block_stack:
            tok_embd = block(tok_embd)
        x_in = torch.sum(tok_embd, dim=1, dtype=torch.float64) #todo 直接加不合适
        output = self.out_head(x_in).unsqueeze(1)
        return output

    # Example usage:


if __name__ == '__main__':
    config = Config(1024, 101, 6, 10, 2500, 1, 0.0, True, torch.float64, 3, 1, 1)

    # aa = np.loadtxt('data_HL_2A/0_region_list.txt')
    # bb = np.loadtxt('data_HL_2A/0_cMatrix.txt')
    # reg_embd = region_embedding_func(aa, config)
    # pos_embd = position_embedding_func(bb, config)
    # config.reg_embd = reg_embd
    # config.pos_embd = pos_embd
    # pth_ = "./data_HL_2A/test_database.h5"
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with h5py.File(pth_, 'r') as database:
    #     regi_list = []
    #     posi_list = []
    #     regi = database['regi']
    #     posi = database['posi']
    #     for dkey in regi.keys():
    #         regi_list.append(regi[dkey][:])  # 使用[:]确保数据被复制到列表中
    #         posi_list.append(posi[dkey][:])
    #     regi = np.array(regi_list)
    #     posi = np.array(posi_list)

    with h5py.File("./data_Phantom/phantomdata/mini_train_database.h5", 'r') as database:
        regi = database['regi']
        posi = database['posi']
        regi_list = [regi[dkey][:] for dkey in regi.keys()]  # 收集regi信息
        posi_list = [posi[dkey][:] for dkey in posi.keys()]  # 收集posi信息
    config.regi = regi_list
    config.posi = posi_list
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    onion = Onion(config).to(device)
    print(onion)

    # a = torch.randn(config.batch_size, 1, config.v_size)
    # padding = torch.tensor([[[0, 36, 32]]]).repeat(config.batch_size, 1, 1)
    # aa = torch.cat((a, padding), dim=-1).to(device)
    # summary(onion, (1, 43), device="cuda")
    a = np.loadtxt('./test_input.txt')
    aa_numpy = a.reshape(a.shape[0],a.shape[1])
    aa_tensor = torch.tensor(np.expand_dims(aa_numpy, axis=1))
    aa = aa_tensor.to(device).to(torch.float32)
    print(aa.shape)
    out = onion(aa)
    print('输出形状={}', out.shape)
    print('out ={}', out)
