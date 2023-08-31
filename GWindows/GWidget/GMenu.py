import pandas as pd
from tkinter import *
from tkinter.messagebox import showinfo, showerror
from GWindows.GWidget.publicMember import PublicMember
from GWindows.GWidget.GTopLevel import GraphSliderTop
from torch import no_grad


class MenuBar(Menu):
    """
    主窗口菜单栏
    """

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.master = master
        self.addMenu()

    def addMenu(self):
        about_menu = AboutMenu(self, tearoff=False)
        dataSet_menu = DataSetMenu(self, tearoff=False)
        modelPre_menu = ModelPreMenu(self, tearoff=False)
        self.add_cascade(label='关于', menu=about_menu)
        self.add_cascade(label='数据处理', menu=dataSet_menu)
        self.add_cascade(label='模型预测', menu=modelPre_menu)


class DataSetMenu(Menu, PublicMember):
    """
    '数据处理'菜单
    """

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        PublicMember().__init__()
        self.master = master  # menubar
        self.add_command(label='散点式可视化',
                         command=lambda: self.openFile('打开文件', [('Excel文件', '*.xlsx')], self.showAsScatterGraph))
        self.add_command(label='区域式可视化',
                         command=lambda: self.openFile('打开文件', [('Excel文件', '*.xlsx')], self.showAsCounterGraph))
        self.add_command(label='划分数据集', command=self.train_val_test)


class ModelPreMenu(Menu, PublicMember):
    """
    '模型预测'菜单
    """

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        PublicMember().__init__()
        self.label = None
        self.z = None
        self.y = None
        self.x = None
        self.master = master  # menubar
        self.add_command(label='导入模型',
                         command=lambda: self.openFile('导入模型', [('模型文件', '*.pth')], self.setCurrentModel))
        self.add_command(label='模型预测', command=self.predict)

    def predict(self):
        # 使用导入的模型进行预测
        try:
            model = self.currentModel.to(self.device)
            data = self.partitionedDataSets[self.targetData].to(self.device)
            df = self.loadedData[self.targetData]
        except:
            showerror('GModel', '模型或数据未导入！')
            return
        # data = torch.load('data.pth').to(self.device)
        model.eval()
        with no_grad():
            predict_probability = model(data.x, data.edge_index).cpu()  # 预测，并将数据转回到cpu
        predict_probability_one = predict_probability[:, 1]  # 取第二列(标签为1)的概率
        self.x = df.iloc[:, 1].tolist()  # XX
        self.y = df.iloc[:, 2].tolist()
        self.z = predict_probability_one.numpy().tolist()
        self.label = predict_probability.max(1)[1].numpy().tolist()
        model.to(self.device)
        data.to(self.device)
        GraphSliderTop(self, '模型预测', '正样本的概率分布图', 'X', 'Y', self.x, self.y, self.z, '输出预测结果',
                       lambda: self.saveFile('保存预测结果', '新建文件', '预测结果文件', 'xlsx', self.saveResultExcel))
        model.to('cpu')
        data.to('cpu')

    def saveResultExcel(self, f):
        df = pd.DataFrame()
        df['XX'] = self.x
        df['YY'] = self.y
        df['P_of_Positive'] = self.z
        df['Label'] = self.label
        df.to_excel(f.name, index=True)
        showinfo('GModel', '成功保存预测结果！')


class AboutMenu(Menu):
    """
    '关于'菜单
    """

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.master = master  # menubar
        self.add_command(label='软件信息', command=lambda: showinfo('GModel',
                                                                    '作者：方豪\n'
                                                                    '导师：刘岳\n'
                                                                    '版本：1.0.0\n'
                                                                    '联系方式1：825585398@qq.com\n'
                                                                    '联系方式2：mengbaofh@cug.edu.cn'))  # 绑定事件


class ContextMenu(Menu, PublicMember):
    """
    左侧treeview的右键菜单
    """

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        PublicMember().__init__()
        self.master = master  # MainLeftTreeView
        self.add_command(label='移除数据', command=self.deleteData)  # 绑定事件

    def deleteData(self):
        # 移除数据
        pass
