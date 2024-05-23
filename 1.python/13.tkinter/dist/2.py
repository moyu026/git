import tkinter as tk

def calculate_operation():
    operation = op_menu_var.get()  # 获取当前选中的运算符
    num1 = float(entry_a.get())
    num2 = float(entry_b.get())

    if operation == "+":
        result = num1 + num2
    elif operation == "-":
        result = num1 - num2
    elif operation == "*":
        result = num1 * num2
    elif operation == "/":
        if num2 != 0:  # 避免除数为零的情况
            result = num1 / num2
        else:
            result_label.config(text="Error: Division by zero!")
            return
    else:
        result_label.config(text="Invalid operation selected!")
        return

    result_label.config(text=f"Result: {result}")

# 创建主窗口
window = tk.Tk()
window.title("Simple Calculator")

# 创建输入框
entry_a = tk.Entry(window)
entry_a.grid(row=0, column=0, padx=10, pady=10)
entry_b = tk.Entry(window)
entry_b.grid(row=1, column=0, padx=10, pady=10)

# 创建运算符下拉框
operations = ["+", "-", "*", "/"]
op_menu_var = tk.StringVar(window)
op_menu = tk.OptionMenu(window, op_menu_var, *operations)
op_menu.config(width=5)
op_menu.grid(row=0, column=1, padx=10, pady=10)

# 创建计算按钮，并绑定计算函数
calc_button = tk.Button(window, text="Calculate", command=calculate_operation)
calc_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

# 创建结果标签
result_label = tk.Label(window, text="")
result_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# 运行主循环
window.mainloop()