from dataset import OnionDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchinfo import summary

class Config:

    def __init__(self, n_layer, n_head, dropout, bias, dtype, batch_size, max_input_len, max_rz_len):
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_rz_len = max_rz_len


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim).float())
        self.bias = nn.Parameter(torch.zeros(ndim).float()) if bias else None

    def forward(self, input):
        if input.dtype == torch.float32:
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
        assert config.max_rz_len % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.max_rz_len, 3 * config.max_rz_len, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.max_rz_len, config.max_rz_len, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.max_rz_len = config.max_rz_len
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.max_rz_len, config.max_rz_len))
                                 .view(1, 1, config.max_rz_len, config.max_rz_len))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (max_rz_len)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.max_rz_len, dim=2)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimention must be divisible by number of heads"

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores += attn_mask
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.out_proj(out)
        return out, attn_weights


def self_attention(embed_input, multihead_attn):
    return multihead_attn(query=embed_input, key=embed_input, value=embed_input)


def cross_attention(query, posi, multihead_attn):
    return multihead_attn(query=query, key=posi, value=posi)


def process_attention(multihead_atten,embed_input, posi):
    for i in range(4):
        embed_input, _ = self_attention(embed_input, multihead_atten)
    self_attn_output = embed_input
    # cross_attn_output, _ = cross_attention(self_attn_output, posi, multihead_atten)
    cross_attn_output, _ = cross_attention(posi,self_attn_output, multihead_atten)

    vec = cross_attn_output
    return vec


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.max_rz_len, 4 * config.max_rz_len, bias=config.bias)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.max_rz_len, config.max_rz_len, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Encoder_Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.max_rz_len, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.max_rz_len, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        # x = self.ln_1(x + self.attn(x))
        x = x + self.mlp(self.ln_1(x))
        # x = self.ln_2(x + self.mlp(x))
        return x


class Out_head(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.max_rz_len, config.max_rz_len, bias=config.bias, dtype=torch.float32)
        self.relu = nn.ReLU()
        self.sgmoid = nn.Sigmoid()
        self.c_proj = nn.Linear(config.max_rz_len, config.max_rz_len, bias=config.bias, dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        # x = self.relu(x)
        x = self.sgmoid(x)
        x = self.dropout(x)
        # x = self.relu(x)
        return x

class vec_compress(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c1= nn.Conv1d(in_channels = config.max_input_len, out_channels = config.max_input_len//2, kernel_size = 3, padding = 1,dtype=torch.float32)
        self.relu = nn.ReLU()
        self.c2= nn.Conv1d(in_channels = config.max_input_len//2, out_channels = 1, kernel_size = 3, padding = 1, dtype=torch.float32)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = x.permute(0,2,1)
        return x

class ConvEmbModel(nn.Module):
    def __init__(self, input_len=100, out_channels=10000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device  # Explicitly initialize device
        # self.sigmoid = nn.Sigmoid()  # Corrected typo in the activation function name
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=1)
        # self.batch_norm = nn.BatchNorm1d(input_len)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).transpose(1, 2)
        # x_norm = self.batch_norm(x)
        # output_tensor = self.sigmoid(x_norm)
        # return output_tensor
        return x


class Onion(nn.Module):
    def __init__(self, config):
        super(Onion, self).__init__()
        self.config = config
        self.conv_emb = ConvEmbModel(out_channels=config.max_rz_len)
        self.block_stack = nn.ModuleList([Encoder_Block(config) for _ in range(config.n_layer)])
        # self.LN = DynamicLayerNorm()
        self.LN = nn.LayerNorm(config.max_rz_len)
        self.posi_embedding = nn.Linear(config.max_rz_len, config.max_rz_len)
        # self.vec_compress = nn.Linear(config.max_input_len*config.max_rz_len, config.max_rz_len)
        self.vec_compress = vec_compress(config)
        self.out_head = Out_head(config)
        self.max_input_len = config.max_input_len
        self.n = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # 添加一个可学习的权重 n
        self.n_head = config.n_head
        self.max_rz_len = config.max_rz_len
        self.multihead_atten = MultiHeadAttention(config.max_rz_len,config.n_head)
        # self.process_attention = process_attention(self.multihead_atten,embeded_input, posi)

    def _init_weihts(self):
        # 实现权重初始化逻辑
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # 举例使用Xavier初始化，根据需要调整
            elif 'bias' in name and param is not None:
                nn.init.zeros_(param)

    def forward(self, input, regi, posi):
        input_embeded = self.conv_emb(input)
        posi_embeded = posi
        # posi_embeded = self.posi_embedding(posi)
        # vec = embeded_input + regi + posi
        # vec = (embeded_input + posi) * regi
        vec = process_attention(self.multihead_atten,input_embeded, posi_embeded)
        for block in self.block_stack:
            vec = block(vec)
        # x_in = torch.sum(vec, dim=1, dtype=torch.float32)##很不合适！！！展平过MLP合适
        # vec_out =self.vec_compress(vec.view(4, -1).unsqueeze(1))
        #output = self.out_head(vec_out).unsqueeze(1) * regi[:, 0, :].unsqueeze(1)
        vec_out =self.vec_compress(vec) * regi[:, 0, :].unsqueeze(1)
        output = self.out_head(vec_out)
        return output

# max_input_len, max_rz_len = 100, 2500
# train_path = '/media/congwang/data/python_code/Onion/data_Phantom/phantomdata/mini_train_database.h5'
# train_dataset = OnionDataset(train_path, max_input_len=max_input_len, max_rz_len=max_rz_len)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
#
# loss_fn = nn.MSELoss().to('cuda')
# conv_emb = ConvEmbModel(out_channels=max_rz_len)
# config = Config(1024, 40, 6, 10, max_rz_len, 1, 0.0, True, torch.float32, 3, 1, 1)
# model = Onion(config)
# for (input, regi, posi, info), label in train_loader:
#     input = conv_emb(input)
#     regi = regi.unsqueeze(1).repeat(1, max_input_len, 1)
#     vec = input + regi + posi
#     pred = model(vec).squeeze(1)
#     loss = loss_fn(pred, label)
if __name__ =="__main__":
    config = Config(2, 4, 0.5, True, torch.float32, 1, 24, 25*36)
    input_shapes = [(1,24),(1,24,25*36),(1,24,25*36)]
    inputs = [torch.randn(*shape) for shape in input_shapes]
    onion = Onion(config)
    summary(onion,input_data=inputs)