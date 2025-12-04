import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ------------------------------
# 1. 读取 peaks 文件
# ------------------------------
def load_peaks(filename):
    t_list = []
    y_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            t, y = line.split()
            t_list.append(float(t))
            y_list.append(float(y))
    return np.array(t_list), np.array(y_list)

# ------------------------------
# 2. 指数衰减模型
# θ(t) = A * exp(-t / τ)
# ------------------------------
def envelope_func(t, A, tau):
    return A * np.exp(-t / tau)

# ------------------------------
# 3. 拟合 + 计算不确定度
# ------------------------------
def fit_envelope(t, y):
    # 初始 guess：A≈第一点，tau≈15~20秒(经验值)
    init_guess = (y[0], 20)

    popt, pcov = curve_fit(envelope_func, t, y, p0=init_guess)

    A, tau = popt
    A_err, tau_err = np.sqrt(np.diag(pcov))

    return (A, A_err, tau, tau_err)

# ------------------------------
# 4. 计算 Q 值
# Q = π * τ / T
# ------------------------------
def compute_Q(tau, tau_err, T, T_err):
    Q = np.pi * tau / T

    # 误差传播：
    Q_err = Q * np.sqrt((tau_err / tau)**2 + (T_err / T)**2)

    return Q, Q_err

# ------------------------------
# 5. 画图
# ------------------------------
def plot_envelope(t, y, A, tau, save_name="envelope_fit.png"):

    # 生成光滑曲线用于画拟合线
    t_fit = np.linspace(t.min(), t.max(), 400)
    y_fit = envelope_func(t_fit, A, tau)

    plt.figure(figsize=(10, 6))
    plt.scatter(t, y, s=25, label="Peaks data", color="blue")
    plt.plot(t_fit, y_fit, label=f"Fit: A·exp(-t/τ)", color="red")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Envelope Fit of Damped Pendulum")
    plt.grid(True)
    plt.legend()

    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.show()


def calculation_T(t):
    T_list = []
    sum = 0
    for i in range(len(t)-1):
        T_list.append(t[i+1] - t[i])
        sum += T_list[i]
    T_mean = sum / len(T_list)
    T_list_d_s = []
    T_d_s_sum = 0
    for i in range(len(T_list)):
        T_list_d_s.append((T_list[i] - T_mean)**2)
        T_d_s_sum += T_list_d_s[i]

    sd_s = 1/(len(T_list) - 1) * T_d_s_sum
    uncertainty = np.sqrt(sd_s) / np.sqrt(len(T_list))
    return T_mean, uncertainty




# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":

    filename = "peaks_25cm_rad.txt"     # 你刚刚生成的文件名


    t, y = load_peaks(filename)

    T, T_err = calculation_T(t)  # 平均周期（你给我之后我帮你填） # 周期误差（你给我之后我帮你改）
    print("Average period:",T,"±",T_err)
    # 拟合
    A, A_err, tau, tau_err = fit_envelope(t, y)

    print("\n=== Fit Results ===")
    print(f"A = {A:.4f} ± {A_err:.4f}")
    print(f"tau = {tau:.4f} ± {tau_err:.4f}  (seconds)")

    # 计算 Q
    Q, Q_err = compute_Q(tau, tau_err, T, T_err)

    print("\n=== Q Factor ===")
    print(f"Q = {Q:.4f} ± {Q_err:.4f}")

    # 画图
    plot_envelope(t, y, A, tau)
