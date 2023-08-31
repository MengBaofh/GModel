import random
# import torch
import seaborn as sns
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
    selectedData = []  # 右侧按钮列
    targetData = None  # 当前选中的数据（图）
    partitionedDataSets = {}  # Data类型数据{f.name:Data(),}
    currentModel = None  # 当前使用的模型
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
        # 将当前选中的Data()类型的数据保存为Excel
        data = self.partitionedDataSets[self.targetData]
        ttype = [0] * data.x.shape[0]  # 默认为测试集，标记为0
        for num, i in enumerate(data.train_mask):
            if i:
                ttype[num] = 1  # 训练集标记为1
        for num, i in enumerate(data.val_mask):
            if i:
                ttype[num] = 2  # 验证集标记为2
        df = self.loadedData[self.targetData]
        df['Type'] = ttype
        df.to_excel(f.name, index=False)  # 输出为excel，不包含索引列

    @staticmethod
    def saveAsTorch(x, f):
        # torch的保存
        save(x, f.name)

    def showAsScatterGraph(self, f):
        # 将选择的excel文件图形化散点式输出，x轴为第二列，y轴为第三列，z轴为最后一列
        plt.close()
        df = self.dataLoader(f.name)
        # 使用 Seaborn 调色板自动生成颜色
        sns.set_palette('husl')  # 使用 husl 调色板
        ax = sns.scatterplot(data=df, x=df.columns[1], y=df.columns[2], hue=df.columns[-1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')  # x\y轴间隔相同
        plt.title(f'{f.name.split("/")[-1]}图形化输出结果')
        plt.legend()
        # 在 y = (y_max + y_min) / 2 处画一条辅助横线
        y_min = df[df.columns[2]].min()
        y_max = df[df.columns[2]].max()
        y_middle = (y_max + y_min) // 2
        ax.axhline(y=y_middle, color="red", linestyle="-", label="Middle Line")
        plt.show()

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

    def train_val_test(self):
        # 划分数据集并保存到partitionedDataSets
        try:
            dataFrame = self.loadedData[self.targetData]
        except:
            showerror('GModel', '未选中图数据，请选中后重试！')
            return
        df_adj = dataFrame.iloc[:, 0:3]  # 用于构建邻接关系-前三列
        df_x = dataFrame.iloc[:, 3:-1]
        df_y = dataFrame.iloc[:, -1]
        x = tensor(df_x.values, dtype=torchFloat)  # 特征值-中间列
        y = tensor(df_y.values, dtype=torchLong)  # 分类标签-最后一列
        data = Data(x=x, y=y)
        ini_points = {int(i[0] + 1): list(i[1:]) for i in df_adj.values}  # pointIndex(1~n):[x, y]
        num_of_point = len(ini_points)
        upper_sample1 = []  # 阴
        upper_sample2 = []  # 阳
        lower_sample1 = []  # 阴
        lower_sample2 = []  # 阳
        unKnown_sample = []  # 未知
        allPoints_y = data.y.numpy().tolist()
        for i in range(num_of_point):
            if i < num_of_point // 2:  # 上半图的点
                if allPoints_y[i] == 0:  # 样本点1(阴性)
                    upper_sample1.append(i)
                elif allPoints_y[i] == 1:  # 样本点2
                    upper_sample2.append(i)
                else:
                    unKnown_sample.append(i)  # 未知
            else:  # 下部分点
                if allPoints_y[i] == 0:  # 样本点1
                    lower_sample1.append(i)
                elif allPoints_y[i] == 1:  # 样本点2
                    lower_sample2.append(i)
                else:
                    unKnown_sample.append(i)  # 未知
        upper_num = len(upper_sample1) if len(upper_sample1) < len(upper_sample2) else len(upper_sample2)  # 取数量少的为标准
        lower_num = len(lower_sample1) if len(lower_sample1) < len(lower_sample2) else len(lower_sample2)
        selected_upper_sample2 = random.sample(upper_sample2, int(upper_num * 0.8))  # 上半图阳性点
        selected_upper_sample1 = random.sample(upper_sample1, len(selected_upper_sample2))  # 阴性点与阳性点数量应该相同
        selected_lower_sample2 = random.sample(lower_sample2, int(lower_num * 0.8))
        selected_lower_sample1 = random.sample(lower_sample1, len(selected_lower_sample2))
        selected_points = [0] * num_of_point  # 全图选择的点
        test_points = [0] * num_of_point  # 测试集的点
        for point in selected_upper_sample1:
            selected_points[point] = 1
        for point in selected_upper_sample2:
            selected_points[point] = 1
        for point in selected_lower_sample1:
            selected_points[point] = 1
        for point in selected_lower_sample2:
            selected_points[point] = 1
        for point in unKnown_sample:
            test_points[point] = 1
        upper_points_mask = selected_points[:num_of_point // 2] + [0] * (num_of_point - num_of_point // 2)  # 使用上部分点
        lower_points_mask = [0] * (num_of_point // 2) + selected_points[num_of_point // 2: num_of_point]
        data.train_mask = BoolTensor(upper_points_mask)  # 转换为张量，上部分为训练集
        data.val_mask = BoolTensor(lower_points_mask)
        data.test_mask = BoolTensor(test_points)
        data.ini_points = ini_points
        self.partitionedDataSets[self.targetData] = data
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
