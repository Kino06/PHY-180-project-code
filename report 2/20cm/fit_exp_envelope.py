# -*- coding: utf-8 -*-
# fit_exp_envelope.py
# 拟合 θ(t) = A * exp(-B t)，忽略过大的异常点，但画图时全部显示

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ========== 读取峰值数据 ==========
def load_peaks(filename):
    t_list, theta_list = [], []
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("t("):  # 跳过表头
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                theta = float(parts[1])
                t_list.append(t)
                theta_list.append(theta)
            except ValueError:
                continue
    return np.array(t_list), np.array(theta_list)

# ========== 指数模型 ==========
def exp_model(t, A, B):
    return A * np.exp(-B * t)

# ========== 拟合函数 ==========
def fit_envelope(t, theta, max_angle=None):
    amp = np.abs(theta)

    # 只拟合 abs(theta) <= max_angle 的点
    mask = (amp > 0)
    if max_angle is not None:
        mask &= (amp <= max_angle)
    t_fit, a_fit = t[mask], amp[mask]

    # 初始估计
    c1, c0 = np.polyfit(t_fit, np.log(a_fit), 1)
    A0 = np.exp(c0)
    B0 = -c1 if -c1 > 0 else 0.01

    popt, pcov = curve_fit(
        exp_model, t_fit, a_fit,
        p0=[A0, B0],
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        maxfev=20000
    )
    A, B = popt
    perr = np.sqrt(np.diag(pcov)) if pcov.size else [np.nan, np.nan]
    A_err, B_err = perr

    # R² 拟合优度
    yhat = exp_model(t_fit, A, B)
    ss_res = np.sum((a_fit - yhat) ** 2)
    ss_tot = np.sum((a_fit - np.mean(a_fit)) ** 2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return (A, A_err, B, B_err, R2), (t_fit, a_fit, yhat)

# ========== 保存并打印拟合结果 ==========
def save_report(params, out_file):
    A, A_err, B, B_err, R2 = params
    tau = 1 / B if B > 0 else np.inf
    t_half = np.log(2) / B if B > 0 else np.inf

    # --- 打印到控制台 ---
    print("\n拟合结果:")
    print(f"|theta(t)| = A * exp(-B t)")
    print(f"A = {A:.6f} ± {A_err:.6f} rad")
    print(f"B = {B:.6f} ± {B_err:.6f} 1/s")
    print(f"τ (1/B) = {tau:.4f} s")
    print(f"t_half = {t_half:.4f} s")
    print(f"R² = {R2:.6f}")

    # --- 保存到文件 ---
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Fit |theta(t)| = A * exp(-B t)\n")
        f.write(f"A = {A:.6f} ± {A_err:.6f} (rad)\n")
        f.write(f"B = {B:.6f} ± {B_err:.6f} (1/s)\n")
        f.write(f"tau = 1/B = {tau:.4f} s\n")
        f.write(f"t_half = ln(2)/B = {t_half:.4f} s\n")
        f.write(f"R^2 = {R2:.6f}\n")

    print(f"\n结果已保存到 {out_file}\n")

# ========== 绘图 ==========
def plot_fit(t, theta, params, max_angle=None, out_png=None):
    A, A_err, B, B_err, R2 = params
    t_all = np.linspace(0, max(t), 400)
    y_fit = exp_model(t_all, A, B)

    amp = np.abs(theta)
    mask_ok = (amp <= max_angle) if max_angle is not None else np.ones_like(amp, dtype=bool)

    plt.figure(figsize=(8, 5))
    plt.scatter(t[mask_ok], amp[mask_ok], label="Peaks (fit used)", c="blue")
    plt.scatter(t[~mask_ok], amp[~mask_ok], label="Ignored (>1.5 rad)", c="red")
    plt.plot(t_all, y_fit, "k-", label=f"Fit: A e^(-B t)\nA={A:.3f}, B={B:.4f}")
    plt.xlabel("Time (s)")
    plt.ylabel("|theta| (rad)")
    plt.title("Exponential Envelope Fit")
    plt.legend()
    plt.grid(True)

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.show()

# ========== 主程序 ==========
if __name__ == "__main__":
    in_file = "peaks_clean.txt"
    out_txt = "fit_AB.txt"
    out_png = "fit_peaks.png"

    # Step 1: 读取峰值
    t, theta = load_peaks(in_file)

    # Step 2: 拟合（忽略 >1.5 rad）
    params, (t_fit, a_fit, yhat) = fit_envelope(t, theta, max_angle=1.5)

    # Step 3: 保存 + 打印结果
    save_report(params, out_txt)

    # Step 4: 画图
    plot_fit(t, theta, params, max_angle=1.5, out_png=out_png)