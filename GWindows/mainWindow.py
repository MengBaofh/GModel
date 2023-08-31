from tkinter import *
from GWindows.GWidget.GFrame import Frame0, Frame10, Frame11, Frame2
from GWindows.GWidget.GMenu import MenuBar


class MainWindow(Frame):
    frame0 = None  # frame0
    buttonFrame = None
    panedWindow = None
    frame10 = None
    frame11 = None
    frame2 = None  # 状态栏
    menu_bar = None  # 菜单栏
    root = None

    def __init__(self, root):
        super().__init__(root)
        self.master = root
        self.setFrame()
        self.setMenuBar()

    def setFrame(self):
        self.panedWindow = PanedWindow(self, sashrelief=SUNKEN, relief='groove')
        self.frame11 = Frame11(self.panedWindow)
        self.frame10 = Frame10(self.panedWindow)
        self.frame0 = Frame0(self)
        self.frame2 = Frame2(self)
        self.setPanedWindow()
        self.frame0.place(relx=0, rely=0, relheight=0.07, relwidth=1)
        self.frame2.place(relx=0, rely=0.97, relheight=0.03, relwidth=1)

    def setPanedWindow(self):
        self.panedWindow.add(self.frame10)
        self.panedWindow.add(self.frame11)
        self.panedWindow.paneconfigure(self.frame10, minsize=200)
        self.panedWindow.paneconfigure(self.frame11, minsize=700)
        self.panedWindow.place(relx=0, rely=0.07, relheight=0.9, relwidth=1)

    def setMenuBar(self):
        self.menu_bar = MenuBar(self)
        self.master['menu'] = self.menu_bar

    def getFrame(self):
        """
        获取该窗口下的全部frame对象
        :return:
        """
        return self.frame0, self.frame10, self.frame11, self.frame2
