from torch.nn import functional as F, ModuleList, Module, Dropout, Linear
from torch_geometric.nn import GCNConv, GATConv


class GCN(Module):
    def __init__(self, convs):
        super(GCN, self).__init__()
        self.convs = ModuleList()
        self.dropouts = ModuleList()
        lens = len(convs)
        classs = convs[f'conv{lens-1}'][1]
        self.linear1 = Linear(256, 512)
        self.linear2 = Linear(512, 1024)
        self.linear3 = Linear(1024, int(classs))
        for key, (inputDim, outputDim, dropout) in convs.items():
            # 定义图卷积层
            if key == f'conv{lens-1}':
                self.convs.append(GCNConv(int(inputDim), 256))
                # 定义dropout
                self.dropouts.append(Dropout(float(dropout)))
            else:
                self.convs.append(GCNConv(int(inputDim), int(outputDim)))
                # 定义dropout
                self.dropouts.append(Dropout(float(dropout)))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))  # 卷积
            x = self.dropouts[i](x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return F.softmax(x, dim=1)


class GAT(Module):
    def __init__(self, convs):
        super(GAT, self).__init__()
        self.convs = ModuleList()
        lens = len(convs)
        classs = convs[f'conv{lens-1}'][1]
        self.linear1 = Linear(256, 512)
        self.linear2 = Linear(512, 1024)
        self.linear3 = Linear(1024, int(classs))
        for key, (inputDim, outputDim, numHeads, dropout, concat) in convs.items():
            # 定义图注意层
            if key == f'conv{lens-1}':
                self.convs.append(GATConv(int(inputDim), 256, heads=int(numHeads), dropout=float(dropout),
                                          concat=(concat == '拼接')))
            else:
                self.convs.append(GATConv(int(inputDim), int(outputDim), heads=int(numHeads), dropout=float(dropout),
                                          concat=(concat == '拼接')))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return F.softmax(x, dim=1)
