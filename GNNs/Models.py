from torch.nn import functional as F, ModuleList, Module, Dropout
from torch_geometric.nn import GCNConv, GATConv


class GCN(Module):
    def __init__(self, convs):
        super(GCN, self).__init__()
        self.convs = ModuleList()
        self.dropouts = ModuleList()
        for key, (inputDim, outputDim, dropout) in convs.items():
            # 定义图卷积层
            self.convs.append(GCNConv(int(inputDim), int(outputDim)))
            # 定义dropout
            self.dropouts.append(Dropout(float(dropout)))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))  # 卷积
            x = self.dropouts[i](x)
        return F.softmax(x, dim=1)


class GAT(Module):
    def __init__(self, convs):
        super(GAT, self).__init__()
        self.convs = ModuleList()
        for key, (inputDim, outputDim, numHeads, dropout, concat) in convs.items():
            # 定义图注意层
            self.convs.append(GATConv(int(inputDim), int(outputDim), heads=int(numHeads), dropout=float(dropout),
                                      concat=(concat == '拼接')))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return F.softmax(x, dim=1)
