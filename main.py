import os
import matplotlib
from GWindows.mainWindow import MainWindow
from tkinter import Tk
from tkinter.messagebox import *


def StopAll():
    if askyesno('GModel', '确定要退出吗？'):
        root.quit()
        root.destroy()
        exit()


if __name__ == '__main__':
    if not os.path.exists('Torch_Output'):
        os.mkdir('Torch_Output')
    matplotlib.use('TkAgg')
    main_width = 1100  # 宽
    main_height = 700
    root = Tk()
    root.title("GModel")  # 窗口名称
    root.geometry(f"{main_width}x{main_height}")  # 窗口大小
    root.resizable(width=True, height=True)  # 窗口可变
    root.minsize(width=main_width // 2, height=main_height // 2)
    root.iconbitmap('GImage/sys.ico')
    root.protocol('WM_DELETE_WINDOW', StopAll)
    a = MainWindow(root)
    a.place(relx=0, rely=0, relheight=1, relwidth=1)
    root.mainloop()  # 不断刷新主窗口
