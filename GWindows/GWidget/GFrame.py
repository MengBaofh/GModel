import pandas as pd
from tkinter import *
from tkinter.ttk import *
from Pmw import Balloon
from GWindows.GWidget.GMenu import ContextMenu
from GWindows.GWidget.GButton import AddDataButton, DivideDataButton1, DivideDataButton2, GCNButton, GATButton
from GWindows.GWidget.GTreeView import MenuTreeView, DataShowTreeView
from GWindows.GWidget.publicMember import PublicMember


class Frame0(Frame, PublicMember):
    pBar = None
    balloon = None

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        PublicMember().__init__()
        self.divideDataButton1 = None
        self.divideDataButton2 = None
        self.gatButton = None
        self.addDataButton = None
        self.gcnButton = None
        self.master = master
        self.balloon = Balloon(self.master)  # 主窗口气泡提示信息
        self.setButton()

    def setButton(self):
        self.addDataButton = AddDataButton(self)
        self.divideDataButton1 = DivideDataButton1(self)
        self.divideDataButton2 = DivideDataButton2(self)
        self.gcnButton = GCNButton(self)
        self.gatButton = GATButton(self)
        self.balloon.bind(self.addDataButton, '添加图数据(Excel)')
        self.balloon.bind(self.divideDataButton1, '划分数据集')
        self.balloon.bind(self.divideDataButton2, '导入训练区域')
        self.balloon.bind(self.gcnButton, 'GCN模型')
        self.balloon.bind(self.gatButton, 'GAT模型')
        Label(self, text='数\n据').place(relx=0, rely=0, relheight=1, relwidth=0.025)
        self.addDataButton.place(relx=0.025, rely=0, relheight=1, relwidth=0.05)
        self.divideDataButton1.place(relx=0.075, rely=0, relheight=1, relwidth=0.05)
        self.divideDataButton2.place(relx=0.125, rely=0, relheight=1, relwidth=0.05)
        Label(self, text='模\n型').place(relx=0.175, rely=0, relheight=1, relwidth=0.025)
        self.gcnButton.place(relx=0.2, rely=0, relheight=1, relwidth=0.05)
        self.gatButton.place(relx=0.25, rely=0, relheight=1, relwidth=0.05)


class ButtonFrame(Frame, PublicMember):
    """
    存放按钮或标签的容器
    """

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        PublicMember().__init__()
        self.master = master
        self.rowconfigure(0, weight=1)  # 组件横向充满
        self.columnconfigure(0, weight=1)


class Frame10(Frame, PublicMember):
    buttonFrame = None
    frame10_leftTreeView = None

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        PublicMember().__init__()
        self.master = master
        self.contextMenu = ContextMenu
        self.rowconfigure(0, weight=1)  # 组件横向充满
        self.columnconfigure(0, weight=1)
        self.setFrame()
        self.setTreeView()

    def setFrame(self):
        self.buttonFrame = ButtonFrame(self)  # 存放标签的容器
        self.buttonFrame.place(relx=0, rely=0, relheight=1, relwidth=0.1)

    def setTreeView(self):
        self.frame10_leftTreeView = MenuTreeView(self, self.leftClick, self.contextMenu, show="tree")
        self.frame10_leftTreeView.place(relx=0.1, rely=0, relheight=1, relwidth=0.9)

    def leftClick(self, item):
        """
        左键点击函数
        :param item: 点击的item项目
        :return:
        """
        if item in self.selectedData:  # 若已加入右侧按钮则将其移除
            self.selectedData.remove(item)
        else:
            self.selectedData.append(item)  # 若未选中则加入右侧
        self.getTreeView().updateText()  # 更新标签状态
        for widget in self.master.master.getFrame()[2].winfo_children():
            widget.destroy()
        self.master.master.getFrame()[2].setFrame()
        self.master.master.getFrame()[2].setTreeView(pd.DataFrame(), 'None')

    def getButtonFrame(self):
        return self.buttonFrame

    def getTreeView(self):
        return self.frame10_leftTreeView


class Frame11(Frame, PublicMember):
    root = None
    treeView = None
    sroll_11_x_b = None
    sroll_11_x = None
    sroll_11_y = None
    totalRow = None
    buttonCanvas = None
    buttonFrame = None

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        PublicMember().__init__()
        self.master = master
        self.rowconfigure(0, weight=1)  # 组件横向充满
        self.columnconfigure(0, weight=1)
        self.setFrame()
        self.setTreeView(pd.DataFrame(), 'None')

    def on_configure(self, event):
        # 配置 Canvas 的视窗大小为内部内容的大小
        self.buttonCanvas.configure(scrollregion=self.buttonCanvas.bbox('all'))

    def setFrame(self):
        # 创建右侧按钮选项
        self.buttonCanvas = Canvas(self)  # 创建画布实现滚动
        self.buttonFrame = ButtonFrame(self.buttonCanvas)  # 存放按钮的容器
        self.buttonFrame.bind('<Configure>', self.on_configure)
        for graph in self.selectedData:
            Button(self.buttonFrame, text=graph.split('/')[-1].split('/')[0],
                   command=lambda t=graph: self.setTreeView(self.loadedData[t], t)).pack(fill=BOTH, side=LEFT)
        self.buttonCanvas.create_window((0, 0), window=self.buttonFrame, anchor='nw')  # 将frame放到画布中
        self.buttonCanvas.place(relx=0, rely=0, relheight=0.04, relwidth=1)

    def setTreeView(self, dataFrame, path):
        # 设置treeview
        PublicMember.targetData = path  # 设置当前选中文件
        # print(PublicMember.targetData)
        self.treeView = DataShowTreeView(self, show='headings', columns=dataFrame.columns.tolist())
        self.treeView.showData(dataFrame, f'当前所在图文件：{path.split("/")[-1]}')
        self.treeView.place(relx=0, rely=0.07, relheight=0.9, relwidth=0.98)
        self.setScrollBar()

    def setScrollBar(self):
        self.sroll_11_x_b = Scrollbar(self, orient=HORIZONTAL, command=self.buttonCanvas.xview)  # 按钮的水平条
        self.sroll_11_x = Scrollbar(self, orient=HORIZONTAL, command=self.treeView.xview)  # TREE的水平条
        self.sroll_11_y = Scrollbar(self, orient=VERTICAL, command=self.treeView.yview)
        self.sroll_11_x_b.place(relx=0, rely=0.04, relheight=0.03, relwidth=0.98)
        self.sroll_11_x.place(relx=0, rely=0.97, relheight=0.03, relwidth=0.98)
        self.sroll_11_y.place(relx=0.98, rely=0.05, relheight=0.95, relwidth=0.02)
        self.buttonCanvas['xscrollcommand'] = self.sroll_11_x_b.set
        self.treeView['xscrollcommand'] = self.sroll_11_x.set
        self.treeView['yscrollcommand'] = self.sroll_11_y.set

    def getButtonFrame(self):
        return self.buttonFrame

    def getTreeView(self):
        return self.treeView


class Frame2(Frame, PublicMember):

    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        PublicMember().__init__()
        self.master = root
