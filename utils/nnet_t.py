import torch
import torch.nn as nn
import torch.nn.functional as F

class TModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.batch_norm = batch_norm

        # 第一层的主分支
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)


        # 分支2，包含两层， 输出剩余时间
        self.branch2_fc1 = nn.Linear(h1_dim, 512)
        self.branch2_fc2 = nn.Linear(512, 1)


    def forward(self, states_nnet):

        x = states_nnet

        # 预处理输入
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # 第一层
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)

        branch2_out = F.relu(self.branch2_fc1(x))
        branch2_out = F.relu(self.branch2_fc2(branch2_out))

        
        # 返回主输出和两个分支输出
        return branch2_out

    
    def parameters_branch2(self):
        params = []
        for n, p in self.named_parameters():
            if 'branch2' in n:
                params.append(p)

        return params