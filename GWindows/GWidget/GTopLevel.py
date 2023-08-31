import numpy as np
from tkinter import *
from tkinter import ttk
from Pmw import Balloon
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.messagebox import showwarning
from GWindows.GWidget.publicMember import PublicMember
from GWindows.GWidget.GText import EndSeeText


class ParaMulSelectTop(Toplevel, PublicMember):
    """
    参数选则 多下拉框弹窗类，点击按钮可添加下拉框
    """
    width = 535
    height = 350
    frame = None
    canvas = None
    sroll_y = None

    def __init__(self, master, title, headings, availSelections, method, **kw):
        super().__init__(master, kw)
        PublicMember().__init__()
        self.master = master
        self.balloon = Balloon(self.master)
        self.myTitle = title
        self.headings = headings  # 下拉框表头，[第一个下拉框表头, 第二个下拉框表头, 第三个下拉框表头]
        self.availSelections = availSelections  # 下拉框可选项，[[第一个下拉框], [第二个下拉框], [第三个下拉框]]
        self.method = method  # 确定函数
        self.vars = []  # stringVar对象列表
        self.varCount = 0
        self.columnNum = len(self.headings)
        self.scale = self.columnNum / 3  # 因为起始按3算
        self.rescale = 3 / self.columnNum
        self.setParaMulSelectTop()  # 设置弹窗
        self.setInputFrame()  # 设置输入框

    def setParaMulSelectTop(self):
        self.title(self.myTitle)
        self.width = int(self.width * self.scale)
        self.height = int(self.height * self.scale)
        self.iconbitmap('GImage/sys.ico')
        self.geometry(f'{self.width}x{self.height}+100+200')
        self.resizable(False, False)
        # self.grab_set()  # 禁止与其他窗口交互
        self.lift()  # 保持窗口最上

    def on_configure(self, event):
        # 配置 Canvas 的视窗大小为内部内容的大小
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def setInputFrame(self):
        for i in range(self.columnNum):
            Label(self, text=self.headings[i]).place(relx=(0.03 + i * 0.33) * self.rescale,
                                                     rely=0, relheight=0.1,
                                                     relwidth=0.27 * self.rescale)
        self.canvas = Canvas(self)
        self.frame = Frame(self.canvas, relief='groove')
        self.frame.bind('<Configure>', self.on_configure)
        self.canvas.create_window((0, 0), window=self.frame, anchor='nw')  # 将frame放到画布中
        self.canvas.place(relx=0.03, rely=0.1, relheight=0.35, relwidth=0.93)
        for i in range(self.varCount // self.columnNum):
            for j in range(self.columnNum):
                print(self.varCount)
                ttk.Combobox(self.frame, values=self.availSelections[j],
                             textvariable=self.vars[i * self.columnNum + j],
                             justify='center').grid(row=i, column=j)
        self.setScrollBar()
        self.setButton()

    def setScrollBar(self):
        self.sroll_y = Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)
        self.sroll_y.place(relx=0.96, rely=0, relheight=0.5, relwidth=0.04)
        self.canvas['yscrollcommand'] = self.sroll_y.set

    def setButton(self):
        Button(self, text='添加底部图层', command=self.addCondition).place(relx=0.2, rely=0.5, relheight=0.1,
                                                                           relwidth=0.2)
        Button(self, text='删除底部图层', command=self.delCondition).place(relx=0.6, rely=0.5, relheight=0.1,
                                                                           relwidth=0.2)
        Button(self, text='确定', command=self.saveCondition).place(relx=0.3, rely=0.7, relheight=0.1, relwidth=0.4)
        Label(self, text=f'当前图文件：{PublicMember.targetData}').place(relx=0, rely=0.8, relheight=0.2, relwidth=1)

    def addCondition(self):
        # 一次加一行，一行三个下拉框
        if self.varCount > 0:
            for i in range(self.columnNum):
                if self.vars[-(i + 1)].get() == '':
                    showwarning('GModel', '请先配置好上一个图层！')
                    return
        for i in range(self.columnNum):
            myVar = StringVar()
            self.vars.append(myVar)
            ttk.Combobox(self.frame, values=self.availSelections[i], textvariable=myVar,
                         justify='center').grid(row=self.varCount // self.columnNum, column=i)
            self.varCount += 1

    def delCondition(self):
        # 一次删一行，一行self.columnNum个下拉框
        if self.varCount == 0: return
        for i in range(self.columnNum):
            self.frame.winfo_children()[-1].destroy()
            self.vars.pop(self.varCount - 1)
            self.varCount -= 1

    def saveCondition(self):
        # 将vars里的内容保存到字典并跳转到超参数输入
        """
        convs
        {
            conv1: [inputdim, outputdim, dropout],
        }
        """
        if self.varCount > 0:
            for i in range(self.columnNum):
                if self.vars[-(i + 1)].get() == '':
                    showwarning('GModel', '请先配置好上一个图层！')
                    return
        if self.varCount == 0:
            showwarning('GModel', '请至少输入一个图层！')
            return
        for i in range(self.varCount):
            if not i % self.columnNum:
                self.master.convs[f'conv{i // self.columnNum}'] = [self.vars[i + j].get() for j in range(self.columnNum)]
        self.method()
        self.destroy()


class ParaSelectTop(Toplevel, PublicMember):
    """
    参数选则下拉框弹窗类
    """
    width = 350
    height = 350

    def __init__(self, master, myTitle, modelType, parameters, method, isProgressive=False, function=None, **kw):
        super().__init__(master, kw)
        PublicMember().__init__()
        self.balloon = Balloon(self.master)
        self.master = master
        self.myTitle = myTitle
        self.parameters = parameters  # {题目:[选项1, 选项2, ...],}
        self.method = method  # 确定函数
        self.modeType = modelType  # 模型类型
        self.isProgressive = isProgressive  # 是否递进式下拉框
        self.function = function  # 递进规则
        self.vars = {}  # 题目:对应存储变量对象
        self.setVars()
        self.setParaSelectTop()
        self.setLabelDropDown()

    def setVars(self):
        # 设置存储变量
        for parameter, default in self.parameters.items():
            self.vars[parameter] = StringVar()
            self.vars[parameter].set(default[0])

    def setParaSelectTop(self):
        self.title(self.myTitle)
        self.iconbitmap('GImage/sys.ico')
        self.geometry(f'{self.width}x{self.height}+300+200')
        # self.grab_set()  # 禁止与其他窗口交互
        self.lift()  # 保持窗口最上

    def dropdownChanged(self, event):
        # 下拉框选项改变时的回调函数
        formerVar = list(self.vars.items())[0][1]
        if formerVar.get() == '请先选择第一个条件': return
        for parameter, var in list(self.vars.items())[1:]:
            self.parameters[parameter] = self.function(formerVar.get())
            formerVar = var
        self.setLabelDropDown()

    def setLabelDropDown(self):
        n = len(self.vars)
        x, y = 0.2, 0.15
        for parameter, var in self.vars.items():
            Label(self, text=parameter, relief='groove', anchor='center') \
                .place(relx=x, rely=y, relheight=0.07, relwidth=0.25)
            if not self.isProgressive:
                ttk.Combobox(self, values=self.parameters[parameter], textvariable=self.vars[parameter],
                             justify='center').place(relx=x + 0.35, rely=y, relheight=0.07,
                                                     relwidth=0.35)
            else:
                combobox = ttk.Combobox(self, values=self.parameters[parameter], textvariable=self.vars[parameter],
                                        justify='center')
                combobox.bind("<<ComboboxSelected>>", self.dropdownChanged)  # 绑定选择事件
                combobox.place(relx=x + 0.35, rely=y, relheight=0.07, relwidth=0.35)
            y += 0.6 / n
        Button(self, text='确定', command=lambda: self.method(self.modeType, self.vars, self.master.convs)) \
            .place(relx=0.3, rely=0.9, relheight=0.07, relwidth=0.4)


class GraphTextShowTop(Toplevel, PublicMember):
    """
    双轴图文显示弹窗类
    """
    width = 1100
    height = 600

    def __init__(self, master, topTitle, leftGTitle, rightGTitle, gTitle, bottomTitle, buttonFuncDict, **kw):
        super().__init__(master, kw)
        PublicMember().__init__()
        self.ax2 = None
        self.ax1 = None
        self.canvas = None
        self.textbox = None
        self.balloon = Balloon(self.master)
        self.master = master
        self.topTitle = topTitle  # toplevel的标题
        self.leftGTitle = leftGTitle  # 图左轴文本
        self.rightGTitle = rightGTitle  # 图右轴文本
        self.gTitle = gTitle  # 图标题文本
        self.bottomTitle = bottomTitle  # 图底部文本
        self.buttonFuncDict = buttonFuncDict  # 按钮
        self.setGraphTextShowTop()
        self.setText()
        self.setGraph()
        self.setButton()

    def setGraphTextShowTop(self):
        self.title(self.topTitle)
        self.iconbitmap('GImage/sys.ico')
        self.geometry(f'{self.width}x{self.height}+100+100')
        # self.grab_set()  # 禁止与其他窗口交互
        self.lift()  # 保持窗口最上

    def setText(self):
        self.textbox = EndSeeText(self)
        self.textbox.place(relx=0.01, rely=0.1, relheight=0.6, relwidth=0.48)

    def setGraph(self):
        fig = plt.figure()
        # 坐标系ax1画曲线1
        self.ax1 = fig.add_subplot(111)  # 指的是将plot界面分成1行1列，此子图占据从左到右从上到下的1位置
        plt.ylabel(self.leftGTitle)
        plt.title(self.gTitle)
        # 坐标系ax2画曲线2
        self.ax2 = fig.add_subplot(111, sharex=self.ax1, frameon=False)  # 其本质就是添加坐标系，设置共享ax1的x轴，ax2背景透明
        self.ax2.yaxis.tick_right()  # 开启右边的y坐标
        self.ax2.yaxis.set_label_position("right")
        plt.ylabel(self.rightGTitle)
        plt.xlabel(self.bottomTitle)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().place(relx=0.5, rely=0.1, relheight=0.6, relwidth=0.49)

    def setButton(self):
        for i, (key, value) in enumerate(self.buttonFuncDict.items()):
            Button(self, text=key, command=value).place(relx=0.1 + i * 0.8 / len(self.buttonFuncDict), rely=0.8,
                                                        relheight=0.1, relwidth=0.8 / len(self.buttonFuncDict) - 0.05)

    def drawCanvas(self, loss_history, train_acc_history, val_acc_history):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(range(len(loss_history)), loss_history, c=np.array([255, 71, 90]) / 255.,
                      label='Train_Loss')  # c为颜色-红
        self.ax1.legend()
        self.ax2.plot(range(len(train_acc_history)), train_acc_history, c=np.array([79, 179, 255]) / 255.,
                      label='Train_Acc')
        self.ax2.plot(range(len(val_acc_history)), val_acc_history, c=np.array([0, 255, 0]) / 255.0, label='Val_Acc')
        self.ax2.legend()
        self.canvas.draw()


class GraphSliderTop(Toplevel, PublicMember):
    """
    分区显示弹窗类
    """
    width = 1100
    height = 600

    def __init__(self, master, topTitle, gTitle, xTitle, yTitle, x, y, z, buttonText, buttonFunc, **kw):
        super().__init__(master, kw)
        PublicMember().__init__()
        self.colorbar = None
        self.update_canvas_id = None
        self.slider = None
        self.ax = None
        self.canvas = None
        self.balloon = Balloon(self.master)
        self.master = master
        self.topTitle = topTitle  # toplevel的标题
        self.gTitle = gTitle  # 图标题文本
        self.xTitle = xTitle  # x轴文本
        self.yTitle = yTitle  # y轴文本
        self.x = x
        self.y = y
        self.z = z
        self.buttonText = buttonText
        self.buttonFunc = buttonFunc
        self.setGraphSliderTop()
        self.setSlider()
        self.setGraph()
        self.setButton()

    def setGraphSliderTop(self):
        self.title(self.topTitle)
        self.iconbitmap('GImage/sys.ico')
        self.geometry(f'{self.width}x{self.height}+100+100')
        # self.grab_set()  # 禁止与其他窗口交互
        self.lift()  # 保持窗口最上

    def setSlider(self):
        self.slider = Scale(self, from_=2, to=20, orient="vertical", command=self.scheduleAdjustCanvas)
        self.slider.set(10)  # 设置初始默认值
        Label(self, text='调\n\n整\n\n等\n\n值\n\n线\n\n条\n\n数', font=("SimHei", 15)).place(relx=0.88,
                                                                                              rely=0.01,
                                                                                              relheight=0.98,
                                                                                              relwidth=0.05)
        self.slider.place(relx=0.93, rely=0.1, relheight=0.8, relwidth=0.05)

    def setGraph(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        # 填充等值线
        contour = self.ax.tricontourf(self.x, self.y, self.z, levels=10, cmap='hot_r')
        self.colorbar = self.ax.figure.colorbar(contour)
        plt.title(self.gTitle)
        plt.ylabel(self.yTitle)
        plt.xlabel(self.xTitle)
        plt.axis('equal')  # x\y轴间隔相同
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().place(relx=0.03, rely=0.01, relheight=0.84, relwidth=0.85)
        self.canvas.draw()

    def setButton(self):
        Button(self, text=self.buttonText, command=self.buttonFunc).place(relx=0.2, rely=0.86, relheight=0.13,
                                                                          relwidth=0.6)

    def updateCanvas(self, levels):
        # print(levels)
        self.ax.clear()  # 清除之前的图形
        self.colorbar.remove()  # 清除之前的colorbar
        contour = self.ax.tricontourf(self.x, self.y, self.z, levels=levels, cmap='hot_r')
        self.colorbar = self.ax.figure.colorbar(contour)
        plt.title(self.gTitle)
        self.canvas.draw()
        self.update_canvas_id = None  # 重置延时更新的标识符

    def scheduleAdjustCanvas(self, event):
        """
        延时更新画布，防止卡顿
        :param event: 事件对象
        """
        if self.update_canvas_id is None:
            self.update_canvas_id = self.after(10, lambda: self.updateCanvas(int(event)))
