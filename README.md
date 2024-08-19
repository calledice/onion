# 数据
## 幽灵场数据
Phantomdata_generation.py
### 用途
用于构造onion模型的预训练数据，数据符合线积分原理，且归一化。
### 实现方式
运行Phantomdata_generation.py文件 <br />
定义构建数据集的类别：train，valid，test三类 <br />
定义每一类生成不同计算区域的数量 <br />
定义每一种计算区域下生成的case数量（主要是region中最大值和分布的不同）
### 关键参数含义
size_ratio定义计算区域的rz比，一般r小于等于z <br />
numgridz定义z方向上的网格数 <br />
randn_indexr和randn_indexz定义了最大值所在的索引位置 <br />
rdm_point_num定义了诊断装置的所在位置，上中下三个位置<br />
rdm_chord_num定义了诊断装置的观测弦数量<br />
c_matrix为各观测弦的贡献矩阵<br />
region_generate()用于随机生成不同大小的场区域<br />
value为场的最大值<br />
label_generate()是基于不同大小的场区域，给场中的网格赋值<br />
input是诊断信号，由贡献矩阵和label相乘获得<br />
返回的input最后三位还包括对应场区域的索引和场区域的大小r和z <br />
## East数据
### 来源
论文：Application of deep learning to soft x-ray tomography at EAST
## HL-2A数据
### 来源
论文：Deep learning based surrogate model a fast soft x-ray(SXR) tomography on HL-2A Tokamak
## XL-50U数据
### 来源
新奥新能源研究院
# 模型
## 模型结构
### onion_model
作为baseline模型，模型的输入包括了input，posi，regi三部分，采用embedding的方式对齐维度，然后vec = (embeded_input + posi) * regi <br />
vec过数个encoder模块后，简单的第二个维度相加合并，进入输出头
```
class Onion(nn.Module):
    def __init__(self, config):
        super(Onion, self).__init__()
        self.config = config
        self.conv_emb = ConvEmbModel(out_channels=config.max_rz_len)
        self.block_stack = nn.ModuleList([Encoder_Block(config) for _ in range(config.n_layer)])
        # self.LN = DynamicLayerNorm()
        self.LN = nn.LayerNorm(config.max_rz_len)
        self.scale_linear = nn.Linear(config.max_rz_len,config.max_rz_len)
        self.out_head = Out_head(config)
        self.max_input_len = config.max_input_len
        self.n = nn.Parameter(torch.tensor(1.0,requires_grad=True)) # 添加一个可学习的权重 n

    def _init_weihts(self):
        # 实现权重初始化逻辑
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # 举例使用Xavier初始化，根据需要调整
            elif 'bias' in name and param is not None:
                nn.init.zeros_(param)

    def forward(self, input, regi, posi):
        embeded_input = self.conv_emb(input)
        posi = self.scale_linear(posi)
        # vec = embeded_input + regi + posi
        vec = (embeded_input + posi) * regi
        for block in self.block_stack:
            vec = block(vec)
        x_in = torch.sum(vec, dim=1, dtype=torch.float32) #todo 直接加不合适
        # x_in = self.LN(x_in)
        output = self.out_head(x_in).unsqueeze(1)
        return output
```
### onion_model_crossatten
作为baseline模型，模型的输入包括了input，posi，regi三部分，采用embedding的方式对齐维度，然后vec = process_attention(self.multihead_atten,input_embeded, posi_embeded) <br />
这里将input做了自注意力然后和posi做交叉注意力输出vec<br />
vec过数个encoder模块后，vec_out =self.vec_compress(vec)通过1d卷积压缩第二个维度，
最后进入输出头
```
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
        posi_embeded = self.posi_embedding(posi)
        # vec = embeded_input + regi + posi
        # vec = (embeded_input + posi) * regi
        vec = process_attention(self.multihead_atten,input_embeded, posi_embeded)
        for block in self.block_stack:
            vec = block(vec)
        # x_in = torch.sum(vec, dim=1, dtype=torch.float32)##很不合适！！！展平过MLP合适
        # vec_out =self.vec_compress(vec.view(4, -1).unsqueeze(1))
        #output = self.out_head(vec_out).unsqueeze(1) * regi[:, 0, :].unsqueeze(1)
        vec_out =self.vec_compress(vec)
        output = self.out_head(vec_out)
        return output
```
## 模型训练
onion_train.py
需要配置路径参数和config参数
```
    paser = argparse.ArgumentParser()
    # paser.add_argument("--model_name", help="选择模型", default="expert_mmoe")
    paser.add_argument("--model_name", help="选择模型", default="Onion")
    paser.add_argument("--train_input_path", help="训练集输入数据路径", default="./data_Phantom/phantomdata/mini_1_train_database.h5")
    paser.add_argument("--val_input_path", help="验证集输入数据路径", default="./data_Phantom/phantomdata/mini_1_valid_database.h5")
    paser.add_argument("--num_train_epochs", help="num_train_epochs", type=int, default=5)
    paser.add_argument("--weight_decay", help="weight_decay", type=float, default=0.005)
    paser.add_argument("--learning_rate", help="learning_rate", type=float, default=5e-4)
    paser.add_argument("--scheduler_step", help="lr更新步长", type=int, default=500)
    paser.add_argument("--check_every", help="每多少步validate一次", type=int, default=200)
    # paser.add_argument("--out_path", help="输出路径", default="./model_data")
    paser.add_argument("--out_path", help="输出路径", default="./model_attn_data")
    paser.add_argument("--tb_save_path", help="TensorBoard 保存路径", default="TensorBoard_logs")
    args = paser.parse_args()
```
```

    config = Config(4, 32, 0.0, True, torch.float32, 4,100,2048)
    config_dict = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "dropout": config.dropout,
        "bias": config.bias,
        "dtype": str(config.dtype),
        "batch_size": config.batch_size,
        "max_input_len": config.max_input_len,
        "max_rz_len": config.max_rz_len
    }
```
模型最终保存pt文件，模型结构文件，模型config文件,loss曲线等
## 结果评估与可视化
output_show_compare.py和see_dataset.py
### 用途
output_show_compare.py用于可视化test结果，包括output和label的对比，input和result的对比 <br />
see_dataset.py用于检查数据是否合理
### 实现方式
#### output_show_compare.py
运行Phantomdata_generation.py文件 <br />
输入模型结构路径，配置文件路径，测试数据集路径<br />
predict()会生成多个csv文件并且返回各文件的路径 <br />
plot_save()读入个csv文件并且保存画图
若已获得输出的csv文件，则test = False
#### see_dataset.py
输入文件路径后可以直接运行

## merge code
git checkout cong
git add .
git commit -m "commit"
git push
git checkout main
git pull
git merge cong
解决冲突，重新提交
git push
git checkout cong
git merge main
push