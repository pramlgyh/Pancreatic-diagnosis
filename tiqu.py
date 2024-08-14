# 指定txt文件的路径
file_path = r'test roc.txt'  # 替换为实际的txt文件路径

# 创建一个空列表来存储提取的内容
extracted_data = []

# 打开文件并读取内容
with open(file_path, "r") as file:
    for line in file:
        # 检查行中是否包含"prob:"
        if "prob:" in line:
            # 使用split方法根据"prob:"分割字符串，并取分割后的第二部分（索引为1）
            # 注意：split会返回一个列表，所以我们需要索引来获取具体的元素
            # strip()用于去除可能存在的换行符或其他空白字符
            extracted_part = line.split("prob:")[1].strip()
            # 将提取的部分添加到列表中
            extracted_data.append(extracted_part)

        # 打印提取的内容列表，或进行其他处理
print(extracted_data)