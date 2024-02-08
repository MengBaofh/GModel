# GModel(GCN\GAT)
一款窗口交互式图神经网络模型（GCN、GAT）软件，同时提供强大的数据可视化分析功能。  
可用于半监督（或监督）预测分类问题（二分类或多分类）。  
· 支持两种数据格式的导入（Excel/pth）；  
· 支持直接导入已训练的模型和已构建好的数据集；  
· 支持绘制导入数据的散点图和等值图；  
· 支持可视化训练集和验证集的划分并提供三种数据集划分方式（竖向划分、横向划分和自定义划分）；  
· 支持对指定数据进行可视化展示（绘制散点图和等值图）。
# 注意
1.输入的Excel表格（记为表1），默认第一列为FID(序号)，第二列为X坐标，第三列为Y坐标，最后一列为标签，中间为节点特征；  
2.Excel表（表1）中的标签必须从0开始编号，并且必须连续，未知类型的保持空值；  
3.使用常规数据集划分方法（纵向或横向划分）时，Excel表（表1）中的点需要按照一定的顺序排列（例如：表中从序号为0开始，对应在图上的位置先从左到右，再从上到下）；  
4.使用自定义数据集划分方法时，可以直接导入需要作为训练集的数据在表1中对应的索引（按列排布，即此处导入的Excel表的第一列为训练集索引，其中第一行不计入），然后会自动进行训练集和验证集的划分（8：2）。    
