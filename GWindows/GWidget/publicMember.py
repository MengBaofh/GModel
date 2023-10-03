import random
import pandas as pd
from tkinter import *
from tkinter.filedialog import *
from torch_geometric.data import Data
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
from tkinter.messagebox import showerror, askyesno, showinfo
from torch import tensor, save, load, BoolTensor, long as torchLong, float as torchFloat

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


class PublicMember:
    # 所有PublicNumber类及其子类共享,仅在最初类创建时执行一次
    loadedData = {}  # 导入的全部数据（图）{f.name:DataFrame(),}
    selectedData = []  # 右侧按钮列表
    targetData = None  # 当前选中的数据（图）
    partitionedDataSets = {}  # 划分数据集后的Data类型数据{f.name:Data(),}
    partitionedDataFrames = {}  # 划分数据集后的DataFrame类型数据{f.name:DataFrame(),}
    currentModel = None  # 当前使用的模型
    train_level = 0.8  # 训练集占比
    device = 'cpu'  # 默认使用的设备
    torch_root = 'Torch_Output/'

    def __init__(self):
        self.image_open = None
        self.image_open_origin = None
        self.image_load = None

    def imageLoader(self, path: str, size: tuple):
        """
        图片加载器
        :param path: 路径
        :param size: 显示的大小
        :return:
        """
        self.image_open_origin = Image.open(path)  # 原始图片
        self.image_open = Image.open(path).resize(size)  # 裁剪后的图片
        self.image_load = ImageTk.PhotoImage(self.image_open)  # 加载后的已裁剪图片
        return self.image_open_origin, self.image_open, self.image_load

    @staticmethod
    def dataLoader(path):
        """
        excel/Data数据加载器，一键处理数据为dataframe格式
        :param path: 待处理的数据路径或data类型数据
        :return:
        """
        if isinstance(path, str):
            df = pd.read_excel(path)
        else:
            x = path.x.numpy()
            y = path.y.numpy()
            FID = [i for i in range(x.shape[0])]
            columns = [f'feature_{i}' for i in range(x.shape[1])]
            df = pd.DataFrame(x, columns=columns)
            df['label'] = y
            df.insert(0, 'FID', FID)
            try:
                ini_points = path.ini_points
                xx = []
                yy = []
                for value in ini_points.values():
                    xx.append(float(value[0]))
                    yy.append(float(value[1]))
                df.insert(1, 'XX', xx)
                df.insert(2, 'YY', yy)
            except AttributeError:
                pass
        return df

    @staticmethod
    def openFile(title, fileTypeName, func):
        """
        打开某类型文件
        :param title: 对话框标题
        :param fileTypeName: 文件类型名及后缀列表[('模型文件', '*.pth')]
        :param func: 读取内容后执行的函数
        :return:
        """
        file_dialog = askopenfile(title=title, filetypes=fileTypeName)
        if file_dialog is not None:  # 检查返回值是否为 None
            with file_dialog as f:
                func(f)

    @staticmethod
    def saveFile(title, iniFileType, fileTypeName, fileType, func):
        """
        保存某类型文件
        :param title: 对话框标题
        :param iniFileType: 默认文件名
        :param fileTypeName: 文件类型名
        :param fileType: 文件类型后缀
        :param func: 读取内容后执行的函数
        :return:
        """
        file_dialog = asksaveasfile(title=title, initialfile=iniFileType, defaultextension=f'*.{fileType}',
                                    filetypes=[(fileTypeName, f'*.{fileType}')])
        if file_dialog is not None:  # 检查返回值是否为 None
            with file_dialog as f:
                func(f)

    def saveAsExcel(self, f):
        # 将当前选中的dataframe保存为Excel
        df = self.partitionedDataFrames[self.targetData]
        df.to_excel(f.name, index=False)  # 输出为excel，不包含索引列

    @staticmethod
    def saveAsTorch(x, f):
        # torch的保存
        save(x, f.name)

    def showAsScatterGraph(self, f):
        # 将选择的excel文件图形化散点式输出，x轴为第二列，y轴为第三列，z轴为最后一列
        self.plt.close()
        df = self.dataLoader(f.name)
        from GWindows.GWidget.GTopLevel import ScatterDiagramTop
        ScatterDiagramTop(self, '图形化输出', f'{f.name.split("/")[-1]}图形化输出结果', 'X', 'Y',
                          df.iloc[:, 1].tolist(), df.iloc[:, 2].tolist(), df.iloc[:, -1].tolist(), '', None)
        # # 使用 Seaborn 调色板自动生成颜色
        # sns.set_palette('husl')  # 使用 husl 调色板
        # ax = sns.scatterplot(data=df, x=df.columns[1], y=df.columns[2], hue=df.columns[-1])
        # self.plt.xlabel('X')
        # self.plt.ylabel('Y')
        # self.plt.axis('equal')  # x\y轴间隔相同
        # self.plt.title(f'{f.name.split("/")[-1]}图形化输出结果')
        # self.plt.legend()
        # # 在 y = (y_max + y_min) / 2 处画一条辅助横线
        # y_min = df[df.columns[2]].min()
        # y_max = df[df.columns[2]].max()
        # y_middle = (y_max + y_min) // 2
        # ax.axhline(y=y_middle, color="red", linestyle="-", label="y Middle Line")
        # x_min = df[df.columns[1]].min()
        # x_max = df[df.columns[1]].max()
        # x_middle = (x_max + x_min) // 2
        # ax.axvline(x=x_middle, color="red", linestyle="-", label="x Middle Line")
        # self.plt.show()

    def showAsCounterGraph(self, f):
        # 将选择的excel文件图形化区域式输出，x轴为第二列，y轴为第三列，z轴为最后一列
        df = self.dataLoader(f.name)
        x = df.iloc[:, 1].tolist()  # XX
        y = df.iloc[:, 2].tolist()
        z = df.iloc[:, -1].tolist()
        # 填充等值线
        from GWindows.GWidget.GTopLevel import GraphSliderTop
        GraphSliderTop(self, '图形化输出', f'{f.name.split("/")[-1]}图形化输出结果', 'X', 'Y', x, y, z, '', None)
        # plt.tricontourf(x, y, z, levels=10, cmap='hot_r')
        # plt.title(f'{f.name.split("/")[-1]}图形化输出结果')
        # plt.ylabel('Y')
        # plt.xlabel('X')
        # plt.axis('equal')  # x\y轴间隔相同
        # plt.show()

    def check_train_val_test(self):
        try:
            dataFrame = self.loadedData[self.targetData]
        except KeyError:
            showerror('GModel', '未选中图数据，请选中后重试！')
            return
        try:
            partitionedData = self.partitionedDataSets[self.targetData]
        except KeyError:
            pass
        else:
            if partitionedData.train_mask is not None:
                if not askyesno('GModel',
                                '检测到数据集已划分，是否重新划分数据集？\n（重新划分数据集后，需要重新生成邻接矩阵）'):
                    return
        from GWindows.GWidget.GTopLevel import TypeSelectTop
        self.topLevel = TypeSelectTop(self, '选择数据集划分方式',
                                      ['横向划分', '纵向划分'],
                                      ['GImage/horizontal.png', 'GImage/vertical.png'],
                                      [lambda: self.init_train_val_test(dataFrame),
                                       lambda: self.init_train_val_test(dataFrame, 1)])

    def init_train_val_test(self, dataFrame, divideType=0):
        self.topLevel.destroy()
        df_adj = dataFrame.iloc[:, 0:3]  # 用于构建邻接关系-前三列
        df_xx = dataFrame.iloc[:, 1]  # x坐标
        df_yy = dataFrame.iloc[:, 2]  # y坐标
        xxList = df_xx.tolist()
        yyList = df_yy.tolist()
        df_x = dataFrame.iloc[:, 3:-1]
        df_y = dataFrame.iloc[:, -1]
        num_of_y = df_y.nunique()  # 标签种数（不包括空值）
        # print(num_of_y)
        xx = tensor(df_xx.values, dtype=torchFloat)  # x坐标
        yy = tensor(df_yy.values, dtype=torchLong)  # y坐标
        x = tensor(df_x.values, dtype=torchFloat)  # 特征值-中间列
        y = tensor(df_y.values, dtype=torchLong)  # 分类标签-最后一列
        row = xxList.count(xxList[0])  # x相同时，y的个数 = 行数
        col = yyList.count(yyList[0])  # y相同时，x的个数 = 列数
        data = Data(x=x, y=y, xx=xx, yy=yy, row=row, col=col)
        data.ini_points = {int(i[0] + 1): list(i[1:]) for i in df_adj.values}  # pointIndex(1~n):[x, y]
        from GWindows.GWidget.GTopLevel import GraphSliderTop2
        self.graphSliderTop2 = GraphSliderTop2(self, '数据集划分', '数据集可视化', 'X', 'Y', dataFrame, divideType, row,
                                               col, '确定',
                                               lambda: self.train_val_test(data, dataFrame, num_of_y, divideType))

    def train_val_test(self, data, dataFrame, num_of_y, divideType):
        # 划分数据集并保存到partitionedDataSets\partitionedDataFrames
        self.graphSliderTop2.destroy()
        num_of_point = len(data.ini_points)
        labels = [i for i in range(num_of_y)]  # 标签
        # print(labels)
        upper_samples = {i: [] for i in labels}  # 上/左图的点
        lower_sample = []  # 下/右
        unKnown_sample = []  # 未知
        allPoints_y = data.y.numpy().tolist()
        if divideType:  # 纵向
            train_col = int(data.col * self.train_level)
            for i in range(num_of_point):
                if i % data.col < train_col:  # 左图的点
                    label = int(allPoints_y[i])
                    if label in labels:
                        upper_samples[label].append(i)
                    else:
                        unKnown_sample.append(i)  # 未知
                else:  # 右图点
                    lower_sample.append(i)
        else:
            train_row = int(data.row * self.train_level)
            # print(train_row)
            for i in range(num_of_point):
                if i // data.col < train_row:  # 上图的点
                    label = int(allPoints_y[i])
                    if label in labels:
                        upper_samples[label].append(i)
                    else:
                        unKnown_sample.append(i)  # 未知
                else:  # 下部分点
                    lower_sample.append(i)
        upper_num = len(upper_samples[0])
        for k, v in upper_samples.items():
            if not v:
                showerror('GModel', f'检测到训练集未包含标签：{k}。\n请重新划分数据集，确保训练集包含所有标签类型！')
                return
            if len(v) < upper_num:
                upper_num = len(v)  # 取数量少的为标准
        # print(upper_samples)
        selected_upper_sample = {k: random.sample(v, int(upper_num * 0.9))
                                 for k, v in upper_samples.items()}
        selected_train_points = [0] * num_of_point  # 上/左图选择的点
        val_points = [0] * num_of_point  # 验证集的点
        test_points = [0] * num_of_point  # 测试集的点(训练集的未选中的点)
        for v in selected_upper_sample.values():
            for point_index in v:
                selected_train_points[point_index] = 1
        for point_index in lower_sample:
            val_points[point_index] = 1
        for point_index in unKnown_sample:
            test_points[point_index] = 1
        data.train_mask = BoolTensor(selected_train_points)  # 转换为张量，上部分为训练集
        data.val_mask = BoolTensor(val_points)
        data.test_mask = BoolTensor(test_points)
        self.partitionedDataSets[self.targetData] = data
        self.partitionedDataFrames[self.targetData] = dataFrame.copy()
        ttype = [0] * data.x.shape[0]  # 默认为测试集，标记为0
        for num, i in enumerate(data.train_mask):
            if i:
                ttype[num] = 1  # 训练集标记为1
        for num, i in enumerate(data.val_mask):
            if i:
                ttype[num] = 2  # 验证集标记为2
        self.partitionedDataFrames[self.targetData]['Type'] = ttype
        if askyesno('GModel', '成功划分数据集。\n划分的数据集已缓存，是否另存为副本？'):
            self.saveFile('保存数据集', '新建数据集', '数据集文件', 'xlsx', self.saveAsExcel)

    def setCurrentModel(self, f):
        try:
            self.currentModel = load(f.name)
        except:
            showerror('GModel', '模型导入失败！')
        else:
            showinfo('GModel', '模型导入成功！')
        # print(self.currentModel)
