import os
from tkinter import Tk, Button


def open_application():
    # 替换下面的路径和程序名以适应你的实际需求
    application_path = "C:\Windows\system32\\mspaint.exe"

    # 使用os.system()函数来打开应用程序
    os.system(f'start {application_path}')


# 创建主窗口
root = Tk()
root.title("Button Application Launcher")

# 创建一个按钮
button = Button(root, text="Open Example App", command=open_application)
button.pack()

# 运行主循环
root.mainloop()