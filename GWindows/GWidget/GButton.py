import os
import sys
import numpy as np
from tkinter import *
from tkinter.messagebox import showerror, showwarning, showinfo
from tkinter.ttk import *
from GWindows.GWidget.publicMember import PublicMember
from GWindows.GWidget.GTopLevel import ParaMulSelectTop, ParaSelectTop, GraphTextShowTop
from GNNs.Models import GCN, GAT
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, NLLLoss
from torch import tensor, save, load, long as torchLong, no_grad, eq, cuda


class CustomButton(Button, PublicMember):
    """
    自定义按钮类
    """

    def __init__(self, master, image_path, command):
        super().__init__(master)
        PublicMember().__init__()
        self.update_id = None  # 用于存储延时更新的标识符
        self.image = None
        self.master = master
        self.image_path = image_path
        self['command'] = command
        self.bind("<Configure>", self.scheduleAdjustImage)

    def scheduleAdjustImage(self, event):
        """
        延时更新调整图像大小，防止卡顿
        :param event: 事件对象
        """
        if self.update_id is None:
            self.update_id = self.after(100, self.adjustImage)

    def adjustImage(self):
        # 图片自适应按钮大小
        self.image = self.imageLoader(self.image_path, (self.winfo_width() - 10, self.winfo_height() - 10))[2]
        self['image'] = self.image
        self.update_id = None  # 重置延时更新的标识符


class AddDataButton(CustomButton):
    """
    添加图(excel)按钮
    """

    def __init__(self, master):
        super().__init__(master, 'GImage/open.png',
                         lambda: self.openFile('打开图文件',
                                               [('图数据文件(Excel)', '*.xlsx'), ('图数据文件(pth)', '*.pth')],
                                               self.addData))

    def addData(self, f):
        # 可能会有两种类型的文件重复加载的情况
        fileName = f.name.split('/')[-1]
        fileSuffix = fileName.split('.')[-1]
        if fileSuffix == 'pth':
            try:
                data = load(f.name)
                x = data.x
            except:
                showerror('GModel', '导入的数据为空或格式有误！')
                return
        try:
            self.master.master.getFrame()[1].getTreeView().insert('', '0', f.name, text=fileName)
        except TclError:
            showwarning('GModel', f'<{f.name}>文件已导入！')
        else:
            if fileSuffix == 'xlsx':
                self.loadedData[f.name] = self.dataLoader(f.name)  # 将数据加入字典
            else:
                self.partitionedDataSets[f.name] = data
                # print(self.partitionedDataSets[f.name])
                self.loadedData[f.name] = self.dataLoader(data)
            self.master.master.getFrame()[1].getTreeView().updateText()


class DivideDataButton(CustomButton):
    """
    划分数据集按钮
    """

    def __init__(self, master):
        super().__init__(master, 'GImage/divideData.png', self.check_train_val_test)


class GNNsButton(CustomButton):
    """
    图神经网络按钮
    """

    def __init__(self, master, image_path, command):
        super().__init__(master, image_path, command)
        self.min_loss = sys.maxsize  # 最低训练损失
        self.bestModelState = None  # 最佳模型
        self.lastModel = None  # 最后一轮训练的模型
        self.gTextShowTop = None
        # 超参数设置窗口
        self.hyperParameterTop = lambda typeName: ParaSelectTop(self, f'{typeName}超参数设置', f'{typeName}',
                                                                {'优化器': ['Adam', 'SGD'],
                                                                 '损失函数': ['交叉损失函数',
                                                                              '负对数似然损失函数'],
                                                                 '学习率': [i / 100 for i in range(10)],
                                                                 '正则化系数': [i / 100 for i in range(10)],
                                                                 '运行设备': ['CPU', 'GPU'],
                                                                 '临界距离': ['请手动输入'],
                                                                 '迭代训练次数': ['请手动输入']},
                                                                self.once)
        self.master = master

    @staticmethod
    def setOptimizer(optimizerType, lr, weight_decay, model):
        if optimizerType == 'Adam':
            return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizerType == 'SGD':
            return SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    @staticmethod
    def getAcc(model, dataSet, mask):
        # 获取标记集合的训练准确度
        model.eval()
        with no_grad():
            x = model(dataSet.x, dataSet.edge_index)  # 重新获取预测结果
            mask_out = x[mask]
            predict_y = mask_out.max(1)[1]
            # print(predict_y)
            acc = eq(predict_y, dataSet.y[mask]).float().mean()  # 将标记集合的预测结果与实际比较
            return acc, mask_out.cpu().numpy(), dataSet.y[mask].cpu().numpy()

    def closeShowTop(self):
        self.gTextShowTop.destroy()
        self.gTextShowTop = None

    def saveTorchModel(self, modelType):
        if modelType:
            if not self.bestModelState:
                showerror('GModel', '还未训练模型，保存失败！')
                return
            save(self.bestModelState, self.torch_root + 'bestModel.pth')
            showinfo('GModel', f'最佳模型成功保存至{os.path.abspath(self.torch_root + "bestModel.pth")}！')
        else:
            if not self.lastModel:
                showerror('GModel', '还未训练模型，保存失败！')
                return
            save(self.lastModel, self.torch_root + 'lastModel.pth')
            showinfo('GModel', f'最终模型成功保存至{os.path.abspath(self.torch_root + "lastModel.pth")}！')

    def saveTorchData(self):
        try:
            data = self.partitionedDataSets[self.targetData]
        except:
            showerror('GModel', '未处理数据或未选中数据！')
        else:
            if data.edge_index is not None:
                save(data, self.torch_root + 'data.pth')
                showinfo('GModel', f'数据成功保存至{os.path.abspath(self.torch_root + "data.pth")}！')
            else:
                showerror('GModel', '未处理数据！')

    def once(self, modelType, vars, convs):
        # 一次实验过程
        # print(convs)
        try:
            data = self.partitionedDataSets[self.targetData]
        except:
            showerror('GModel', '数据集未划分，请先划分数据集！')
            return
        self.gTextShowTop = GraphTextShowTop(self, '训练历程', 'Loss', 'Accuracy',
                                             'Training Loss&Acc And Validation Accuracy',
                                             'Epoch', {'保存最优模型(训练损失最低)': lambda: self.saveTorchModel(1),
                                                       '保存最终模型(最后一轮训练结束)': lambda: self.saveTorchModel(0),
                                                       '保存pth格式数据集': self.saveTorchData})
        self.gTextShowTop.protocol("WM_DELETE_WINDOW", self.closeShowTop)
        if self.partitionedDataSets[self.targetData].edge_index is None:
            self.gTextShowTop.textbox.showProcess(f'! 未检测到邻接矩阵，即将新建邻接矩阵。\n')
            critical_distance = float(vars['临界距离'].get())
            self.gTextShowTop.textbox.showProcess(f'* 临界距离：{critical_distance}\n')
            self.gTextShowTop.textbox.showProcess('√ 正在生成邻接矩阵(每5000节点需要3-5分钟)...\n')
            # data.edge_index = self.constructEdgeIndex(self.gTextShowTop.textbox, data.ini_points, critical_distance)
            edge_index = []
            num_points = len(data.ini_points)
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    if not self.gTextShowTop:
                        return
                    self.gTextShowTop.textbox.update()
                    x1, y1 = data.ini_points[i + 1]
                    x2, y2 = data.ini_points[j + 1]
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if distance <= critical_distance:
                        edge_index.append([i, j])
                        edge_index.append([j, i])
            data.edge_index = tensor(edge_index, dtype=torchLong).t().contiguous()
            # torch.save(data, 'data.pth')
        else:
            self.gTextShowTop.textbox.showProcess(f'* 检测到邻接矩阵，将使用已有的邻接矩阵。\n')
        # 设置模型
        self.gTextShowTop.textbox.showProcess('√ 正在设置模型...\n')
        try:
            if modelType == 'GCN':
                model = GCN(convs)
            elif modelType == 'GAT':
                model = GAT(convs)
            else:
                showerror('GModel', f'× 无法识别的模型类型：{modelType}')
                self.gTextShowTop.textbox.showProcess(f'× 无法识别的模型类型：{modelType}\n')
                return
        except Exception as e:
            showerror('GModel', f'× 模型参数有误，请检查后重试！\n{e}')
            self.gTextShowTop.textbox.showProcess(f'× 模型参数有误，请检查后重试！\n{e}\n')
            return
        self.gTextShowTop.textbox.showProcess(f'* 模型类型：{modelType}\n* 模型参数：\n{model}\n')
        # 设置学习率、正则化系数、迭代训练次数
        self.gTextShowTop.textbox.showProcess('√ 正在设置模型超参数...\n')
        try:
            lr = float(vars['学习率'].get())
            weight_decay = float(vars['正则化系数'].get())
            num_epochs = int(vars['迭代训练次数'].get())
        except:
            showerror('GModel', '× 学习率/正则化系数/迭代训练次数类型有误，请检查后重试！')
            self.gTextShowTop.textbox.showProcess('× 学习率/正则化系数/迭代训练次数类型有误，请检查后重试！\n')
            return
        self.gTextShowTop.textbox.showProcess(
            f'* 学习率：{lr}\n* 正则化系数：{weight_decay}\n* 训练次数：{num_epochs}\n')
        # 设置优化器
        self.gTextShowTop.textbox.showProcess('√ 正在设置模型优化器及损失函数...\n')
        if vars['优化器'].get() == 'Adam':
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif vars['优化器'].get() == 'SGD':
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            showerror('GModel', f'× 未知优化器：{vars["优化器"].get()}，请修改后重试！')
            self.gTextShowTop.textbox.showProcess(f'× 未知优化器：{vars["优化器"].get()}，请修改后重试！\n')
            return
        # 设置损失函数
        if vars['损失函数'].get() == '交叉损失函数':
            criterion = CrossEntropyLoss()
        elif vars['损失函数'].get() == '负对数似然损失函数':
            criterion = NLLLoss()
        else:
            showerror('GModel', f'× 未知损失函数：{vars["损失函数"].get()}，请修改后重试！')
            self.gTextShowTop.textbox.showProcess(f'× 未知损失函数：{vars["损失函数"].get()}，请修改后重试！\n')
            return
        self.gTextShowTop.textbox.showProcess(f'* 优化器：\n{optimizer}\n* 损失函数：{criterion}\n')
        # 设置运行设备
        PublicMember.device = 'cuda' if vars['运行设备'].get() == 'GPU' and cuda.is_available() else 'cpu'
        self.gTextShowTop.textbox.showProcess(f'* 运行设备：{self.device}\n')
        # 数据迁移
        model.to(self.device)
        data.to(self.device)
        # 迭代训练
        self.gTextShowTop.textbox.showProcess('√ 准备就绪！\n')
        loss_history, train_acc_history, val_acc_history = [], [], []
        for epoch in range(num_epochs):
            self.gTextShowTop.update()
            model.train()  # 模型训练
            optimizer.zero_grad()  # 梯度清零
            x = model(data.x, data.edge_index)
            train_loss = criterion(x[data.train_mask], data.y[data.train_mask])  # 根据训练集进行优化
            train_loss.backward()
            optimizer.step()  # 更新模型参数
            if train_loss < self.min_loss:  # 保存训练损失最低的一次的模型参数
                self.min_loss = train_loss
                self.bestModelState = model
                # print(self.bestModelState)
            train_acc_training, _, _ = self.getAcc(model, data, data.train_mask)
            val_acc_training, _, _ = self.getAcc(model, data, data.val_mask)
            loss_history.append(train_loss.item())
            train_acc_history.append(train_acc_training.item())
            val_acc_history.append(val_acc_training.item())
            # print(f'Epoch [{epoch + 1}/{num_epochs}], '
            #       f'Train Loss: {train_loss.item():.4f}, '
            #       f'Train Acc: {train_acc_training.item():.4f}, '
            #       f'Test Acc: {val_acc_training.item():.4f}')
            self.gTextShowTop.textbox.showProcess(f'Epoch [{epoch + 1}/{num_epochs}], '
                                                  f'Train Loss: {train_loss.item():.4f}, '
                                                  f'Train Acc: {train_acc_training.item():.4f}, '
                                                  f'Test Acc: {val_acc_training.item():.4f}\n')
            if not epoch % 20:  # 每20轮画一次
                self.gTextShowTop.drawCanvas(loss_history, train_acc_history, val_acc_history)
        self.lastModel = model
        # 数据迁移
        if self.device == 'cuda':
            model.to('cpu')
            data.to('cpu')


class GCNButton(GNNsButton):
    """
    GCN模型按钮
    """

    def __init__(self, master):
        self.convs = {}
        super().__init__(master, 'GImage/gcn.png',
                         lambda: ParaMulSelectTop(self, 'GCN模型参数配置', ['输入维度', '输出维度', 'dropout'],
                                                  [['请手动输入'], ['请手动输入'], [i / 10 for i in range(10)]],
                                                  lambda: self.hyperParameterTop('GCN'))
                         )


class GATButton(GNNsButton):
    """
    GAT模型按钮
    """

    def __init__(self, master):
        self.convs = {}
        super().__init__(master, 'GImage/gat.png',
                         lambda: ParaMulSelectTop(self, 'GAT模型参数配置',
                                                  ['输入维度', '输出维度', '注意力头数', 'dropout', '多头处理机制'],
                                                  [['请手动输入'], ['请手动输入'], [i for i in range(1, 6)],
                                                   [i / 10 for i in range(10)], ['拼接', '取平均']],
                                                  lambda: self.hyperParameterTop('GAT'))
                         )
