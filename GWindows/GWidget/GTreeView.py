from tkinter import *
from tkinter.ttk import *
from GWindows.GWidget.publicMember import PublicMember


class MenuTreeView(Treeview, PublicMember):
    # 菜单列表，可以实现左键和右键点击
    def __init__(self, master, leftFunc, contextMenu, **kw):
        """
        :param master:
        :param leftFunc: 左键回调函数
        :param contextMenu: 右键菜单类
        :param kw:
        """
        super().__init__(master, **kw)
        PublicMember().__init__()
        self.contextMenu = None
        self.master = master
        self.leftFunc = leftFunc
        self.contextMenu = contextMenu
        self.eventBind()

    def eventBind(self):
        self.bind("<Button-1>", self.onSelect)  # 左键
        self.bind("<Button-3>", self.popupMenu)  # 右键

    def onSelect(self, event):
        """
        treeview左键选中回调函数
        :return:
        """
        item = self.identify('item', event.x, event.y)
        if item:  # 若为Item对象
            self.selection_set(item)  # 选中并高亮
            self.leftFunc(item)

    def popupMenu(self, event):
        item = self.identify('item', event.x, event.y)
        if item:
            self.selection_set(item)  # 选中并高亮
            self.contextMenu(self, tearoff=False).post(event.x_root, event.y_root)  # 弹出右键菜单

    def updateText(self):
        """
        更新选中节点对应的标签状态
        :return:
        """
        for label in self.master.getButtonFrame().winfo_children():
            label.destroy()
        for node in self.get_children():
            Label(self.master.getButtonFrame(), text='√' if node in self.selectedData else '□').pack(fill=X)


class DataShowTreeView(Treeview, PublicMember):
    # 数据展示列表
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.master = master

    def showData(self, dataFrame, lastText):
        """
        显示数据
        :param dataFrame: 待展示的数据框
        :param lastText: 最后一行显示的文字
        :return:
        """
        data_list = dataFrame.values.tolist()
        data_columns = dataFrame.columns.tolist()
        totalRow = dataFrame.shape[0]  # 数据的总行数
        totalCol = dataFrame.shape[1] if data_list else len(data_columns)
        for column in data_columns:
            self.heading(column, text=column, anchor=CENTER)
        for index, rowData in enumerate(data_list):
            self.insert('', index, values=rowData)
        self.insert('', totalRow, values=['|---(None)---'] * totalCol)
        self.insert('', totalRow + 1, values=[f'{lastText}'])
