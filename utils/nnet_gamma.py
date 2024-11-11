import torch
import torch.nn as nn
import torch.nn.functional as F

class GammaModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # 第一层的主分支
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # 分支1，包含两层， 输出gamma值
        self.branch1_fc1 = nn.Linear(h1_dim + out_dim, 512)
        self.branch1_fc2 = nn.Linear(512, 1)


        # ResNet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # 输出层
        self.fc_out = nn.Linear(resnet_dim, out_dim)

        #self.batch_norm = False

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

        # 存储branch的输入
        y = x


        # 主分支继续前向传播
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)

        # ResNet块
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)
            x = F.relu(x + res_inp)

        # 输出层
        main_out = self.fc_out(x)

        y = torch.concat([y, main_out], dim=-1)

        # 两个分支
        branch1_out = F.relu(self.branch1_fc1(y))
        branch1_out = F.relu(self.branch1_fc2(branch1_out))

        
        # 返回主输出和两个分支输出
        return main_out, branch1_out

    def parameters_branch1(self):
        params = []
        for n, p in self.named_parameters():
            if 'branch1' in n:
                params.append(p)

        return params
