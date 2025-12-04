# -*- coding: utf-8 -*-
# find_peaks_keep_all.py
# 功能：
#   - 读取 rad.txt (t, theta_raw_deg)
#   - 转换成“竖直下为0, 右正左负”的角度 (rad)
#   - 保留全部数据（不删除噪点/异常）
#   - 峰值检测
#   - 保存 & 画图

import statistics
import math
import matplotlib.pyplot as plt

# ========== 读数据 ==========
def read_data(filename):
    t_list, y_list = [], []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                y = float(parts[1])   # Tracker导出的角度 (deg)
                t_list.append(t)
                y_list.append(y)
            except ValueError:
                continue
    return t_list, y_list

# ========== 角度数据预处理 ==========
def process_angles(y_deg_list):
    """度数 -> rad，下为0，右正左负，wrap到(-pi, pi]"""
    y_rad_list = []
    for deg in y_deg_list:
        rad = math.radians(deg)   # 减去180°再转rad
        rad = math.atan2(math.sin(rad), math.cos(rad))  # wrap
        y_rad_list.append(rad)
    return y_rad_list

# ========== 边界自适应平滑 ==========
def centered_moving_average(y, window=5):
    if window <= 1 or window % 2 == 0:
        return y[:]
    half = window // 2
    n = len(y)
    out = []
    for i in range(n):
        L = max(0, i - half)
        R = min(n, i + half + 1)
        out.append(sum(y[L:R]) / (R - L))
    return out

# ========== 峰值检测 ==========
def find_peaks_clean(t, y,
                     min_gap_sec=0.40,
                     min_prom=0.003,
                     smooth_win=5,
                     look_half_win=4):
    ys = centered_moving_average(y, smooth_win)

    if len(t) > 1:
        dt = statistics.median([t[i+1] - t[i] for i in range(len(t)-1)])
    else:
        dt = 0.02

    peaks_t, peaks_y = [], []
    last_peak_time = -1e30
    n = len(ys)

    for i in range(1, n-1):
        # 局部极大值
        if not (ys[i-1] < ys[i] and ys[i] > ys[i+1]):
            continue
        # 间隔过滤
        if t[i] - last_peak_time < min_gap_sec:
            continue
        # 突显度
        L = max(0, i - look_half_win)
        R = min(n, i + look_half_win + 1)
        if i - L == 0 or R - (i+1) == 0:
            continue
        left_min = min(ys[L:i])
        right_min = min(ys[i+1:R])
        prom = ys[i] - min(left_min, right_min)

        if prom >= min_prom:
            peaks_t.append(t[i])
            peaks_y.append(y[i])
            last_peak_time = t[i]

    return peaks_t, peaks_y

# ========== 保存 ==========
def save_peaks(peaks_t, peaks_y, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("t(s)\ttheta(rad)\n")
        for tt, yy in zip(peaks_t, peaks_y):
            f.write(f"{tt:.6f}\t{yy:.6f}\n")

# ========== 画图 ==========
def plot_peaks_only(peaks_t, peaks_y, save_png=None):
    plt.figure(figsize=(8,5))
    plt.scatter(peaks_t, peaks_y, label="Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("theta (rad)")
    plt.title("Detected Peaks Only (all data kept)")
    plt.legend()
    plt.grid(True)
    if save_png:
        plt.savefig(save_png, dpi=150, bbox_inches="tight")
    plt.show()

# ========== 主程序 ==========
if __name__ == "__main__":
    in_file = "15cm.txt"
    out_file = "peaks_clean.txt"

    # Step 1: 读数据
    t, y_deg = read_data(in_file)

    # Step 2: 角度处理
    y = process_angles(y_deg)

    # Step 3: 峰值检测（不删除噪点）
    peaks_t, peaks_y = find_peaks_clean(
        t, y,
        min_gap_sec=0.40,
        min_prom=0.003,
        smooth_win=5,
        look_half_win=4
    )

    # Step 4: 保存 & 绘图
    save_peaks(peaks_t, peaks_y, out_file)
    print(f"Clean peaks: {len(peaks_t)} saved to {out_file}")
    plot_peaks_only(peaks_t, peaks_y, save_png="peaks_only.png")