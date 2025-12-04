import matplotlib.pyplot as plt
import math

filename = "peaks_45cm_rad.txt"   # 需要读取数据的文件名
data = []
with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:        # 空行跳过
            continue

        parts = line.split()   # 默认按空白字符(空格/tab)分割

        if len(parts) < 2:
            # 这一行列数不够，可能是标题或注释，跳过
            continue

        try:
            t = float(parts[0])   # 第一列：时间
            y = float(parts[1])   # 第二列：角度(rad)
        except ValueError:
            # 如果这一行不能转成数字（比如是“Time(s)”这种），跳过
            continue

        data.append([t, y])


a = 2
def find(data,a):
    amplitude = data[0][1] * math.exp(-math.pi/a)
    for i in range(len(data)):
        if data[i][1]<=amplitude:
            return i

num = find(data,a)
print("Q /",a,":",num)


def plot(data,num): #描点
    plt.show(block=True)

    px = [p[0] for p in data]  # time
    py = [p[1] for p in data]  # angle (rad)

    plt.figure(figsize=(10, 5))
    plt.scatter(px, py, color='blue', s=30, label="Peaks")
    plt.axvline(x=data[num][0], color='red', linestyle='--', linewidth=2, label='20% ≈ e^(-π/2) amplitude')

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("To find point when reach required amplitude")
    plt.grid(True)
    plt.legend()

    plt.show()

plot(data,num)