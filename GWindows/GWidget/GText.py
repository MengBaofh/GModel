from tkinter import *
from GWindows.GWidget.publicMember import PublicMember


class EndSeeText(Text, PublicMember):
    """
    大文本显示框，自动清除多余文本并跟踪最后一条文本，防止卡顿
    """

    def __init__(self, master, **kw):
        super().__init__(master, kw)
        PublicMember().__init__()
        self.master = master

    def showProcess(self, text):
        current_text = self.get('1.0', 'end-1c')  # Get all text from the Text widget
        lines = current_text.split("\n")
        if len(lines) > 100:
            self.delete('1.0', '2.0')
        self.insert('end', text)
        self.see('end')
        self.update()
