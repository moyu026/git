import tkinter as tk

root = tk.Tk()
root.geometry("500x300+100+100")  # 设置窗口大小和屏幕左上角的偏移量
root.title("demo1")

frame1 = tk.Frame(root)
frame1.pack()

tk.Label(frame1, text="hello world", font=24, pady=10, padx=10).pack(side=tk.LEFT, anchor=tk.N)

img = tk.PhotoImage(file="1.gif")
label_img = tk.Label(frame1, image=img, pady=30, padx=30, bd=0)
label_img.pack(side=tk.LEFT, anchor=tk.N)

frame2 = tk.Frame(root)
# frame2.pack()

root.mainloop()
