import torch
class Config:
    def __init__(self, n_layer=None, n_head=None, dropout=None, bias=True, dtype=torch.float32, batch_size=64, 
                 max_input_len=100, max_rz_len=10000, max_n=100, max_r=100, max_z=100, lr=0.001, epochs=20, early_stop=5):
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_input_len = max_input_len
        self.max_rz_len = max_rz_len
        self.max_n = max_n
        self.max_r = max_r
        self.max_z = max_z
        self.lr = lr
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
print(config.device)