import os

# 定义源文件夹路径
folder1 = r'F:\PythonWork\dataset\electrical_cabinet_all\detect\cabinet-new1\labels\mie'
folder2 = r'F:\PythonWork\dataset\electrical_cabinet_all\detect\cabinet-new1\labels\train'

# 定义目标文件夹路径，用于存放合并后的文件
output_folder = r'F:\PythonWork\dataset\electrical_cabinet_all\detect\cabinet-new1\labels\train1'

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历第一个文件夹中的.txt文件
for filename in os.listdir(folder1):
    if filename.endswith('.txt'):
        full_path1 = os.path.join(folder1, filename)

        # 检查第二个文件夹中是否存在同名文件
        full_path2 = os.path.join(folder2, filename)
        if os.path.exists(full_path2):
            with open(full_path1, 'r') as f1, open(full_path2, 'r') as f2:
                content1 = f1.read().strip()
                content2 = f2.read().strip()

            # 合并文件内容并添加换行符
            merged_content = content1 + '\n' + content2

            # 写入合并后的内容到目标文件夹
            output_file = os.path.join(output_folder, filename)
            with open(output_file, 'w') as outfile:
                outfile.write(merged_content)

print("合并完成.")