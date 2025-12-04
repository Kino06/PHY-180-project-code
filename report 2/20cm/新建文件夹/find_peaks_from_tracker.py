import matplotlib.pyplot as plt
plt.close('all')
data = []   # 二位列表储存所有t，y数据

filename = "20cm_rad.txt"   # 需要读取数据的文件名

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

print("总共读到行数:", len(data))
print("总数据",data)

def find_peak(data): #自动化寻找极值（基于一个正区间内只会有一个最大值的基础寻找）
    peaks = [[data[0][0],data[0][1]]]
    RAM = []
    status = 0
    status_1 = 0
    for i in range(1,len(data)-1):
        if data[i][1]  <= 0 and status >= 0 :
            status_1 = 1
            status = -1
            max = 0
            x_y = None
            if len(RAM) > 1:
                peaks = peaks[:-len(RAM)]
                for a in range(len(RAM)):
                    if RAM[a][1] > max:
                        max = RAM[a][1]
                        x_y = RAM[a]
                peaks.append(x_y)
            RAM = []

        elif status < 0 and data[i][1] > 0:
            status = 0
        if data[i-1][1] <= data[i][1] >= data[i+1][1] and status_1 !=0 and data[i][1] >= 0 :
            RAM.append([data[i][0], data[i][1]])
            peaks.append([data[i][0], data[i][1]])
            status +=1


    return peaks


def plot_peaks(peaks): #描点
    plt.show(block=True)

    px = [p[0] for p in peaks]  # time
    py = [p[1] for p in peaks]  # angle (rad)
    print(px)
    print(py)
    plt.figure(figsize=(10, 5))
    plt.scatter(px, py, color='blue', s=30, label="Peaks")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Detected Peaks from Tracker Data")
    plt.grid(True)
    plt.legend()

    plt.show()

peaks = find_peak(data)[:-1] #去掉最后一次的极值（因为容易出错而且无巨大意义）
print(peaks)
plot_peaks(peaks)

def save_peaks_to_txt(peaks, filename="peaks_output.txt"): #写入新文件
    with open(filename, "w", encoding="utf-8") as f:
        for t, y in peaks:
            f.write(f"{t:.6f} {y:.6f}\n")

    print(f"已成功写入 {len(peaks)} 行到 {filename}")

save_peaks_to_txt(peaks, "peaks_20cm_rad.txt")
