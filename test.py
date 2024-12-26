import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# 生成一个示例信号，例如一个正弦波
t = np.linspace(0, 1, 500, endpoint=False)  # 时间轴
x = np.sin(2 * np.pi * 5 * t)  # 频率为5Hz的正弦波

# 计算信号的希尔伯特变换
x_hilbert = hilbert(x)

# 获取实部和虚部
x_real = np.real(x_hilbert)
x_imag = np.imag(x_hilbert)

# 计算信号的瞬时幅度和瞬时相位
inst_amplitude = np.abs(x_hilbert)
inst_phase = np.angle(x_hilbert)

# 绘制原始信号和希尔伯特变换结果
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, x, label='Original Signal')
plt.title('Original Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, x_real, label='Original Signal')
plt.plot(t, x_imag, label='ilbert Transform', linestyle='--')
plt.title('Real and Imaginary Parts')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, inst_amplitude, label='幅度')
plt.plot(t, inst_phase, label='相位', linestyle='--')
# plt.title('Instantaneous Amplitude and Phase')瞬时幅度是复信号的模，而瞬时相位是复信号的相角
plt.legend()

plt.tight_layout()
plt.show()
