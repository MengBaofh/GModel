# GModel(GCN\GAT)
一款窗口交互式图神经网络模型（GCN、GAT）软件，同时提供强大的数据可视化分析功能。  
可用于半监督（或监督）预测分类问题（二分类或多分类）。  
· 支持两种数据格式的导入（Excel/pth）；  
· 支持已训练模型和已有数据的直接导入；  
· 支持绘制导入数据的散点图和等值图；  
· 支持可视化训练集和验证集的划分并提供两种数据集划分方式（竖向划分和横向划分）；  
· 支持对指定数据进行可视化展示（绘制散点图和等值图）。
# 注意
1.输入的Excel表格，默认第一列为FID(序号)，第二列为X坐标，第三列为Y坐标，最后一列为标签，中间为节点特征。  
2.Excel表中的标签必须从0开始编号，并且必须连续，未知类型的保持空值；Excel表中的点从序号为0开始，对应在图上的位置必须先从左到右，再从上到下。
