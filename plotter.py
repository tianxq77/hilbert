import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题



def plot_results(device_num,time_points ,signal_diff_vref,signal_diff_vac):
        # 绘制每个设备的信号相位过零点差值图
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, signal_diff_vref, label=f'Device {device_num} vref Signal Diff', color='b')
        plt.plot(time_points, signal_diff_vac, label=f'Device {device_num} vac Signal Diff', color='g')
        # 绘制相位过零点
        plt.scatter(time_points, signal_diff_vref, color='b', s=100, zorder=5, label="Zero Crossings (vref)")
        plt.scatter(time_points, signal_diff_vac, color='r', s=100, zorder=5, label="Zero Crossings (vac)")

        plt.xlabel('Time (s)')
        plt.ylabel('Signal Diff')
        plt.title(f'Device {device_num} Signal Differences')
        plt.legend()

        plt.show()

        # # 保存图形为 PNG 文件
        # plt.savefig(f'device_{device_num}_signal_diff.png')
        # plt.close()


def plot_hilbert_transform(signal, time_points, device_name, sample_range=None):
    """
    绘制希尔伯特变换结果和相位过零点
    """
    # 计算希尔伯特变换
    analytic_signal = hilbert(signal)
    x_imag = np.imag(analytic_signal)
    phase = np.angle(analytic_signal)

    # 设置显示范围
    if sample_range is None:
        start, end = 0, len(signal)
    else:
        start, end = sample_range
        time_points = [t for t in time_points if start <= t <= end]

    # 创建时间轴
    t = np.arange(start, end)

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制原始信号和包络
    ax1.plot(t, signal[start:end], label='原始信号')
    ax1.plot(t, x_imag[start:end], label='复信号虚部')
    ax1.scatter(time_points, signal[time_points], color='red', label='相位过零点')
    ax1.set_title(f'{device_name} - 复信号')
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('幅值')
    ax1.legend()
    ax1.grid(True)

    # 绘制相位
    ax2.plot(t, phase[start:end], label='瞬时相位')
    ax2.scatter(time_points, [0] * len(time_points), color='red', label='相位过零点')
    ax2.set_title(f'{device_name} - 瞬时相位')
    ax2.set_xlabel('采样点')
    ax2.set_ylabel('相位 (rad)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    return fig

# device_name = f'Device{device_num}_{type_name}'
# fig = plot_hilbert_transform(signal, time_points, device_name)
# fig.savefig(f'{device_name}_hilbert.png')
# plt.close(fig)



# if __name__ == "__main__":

    # t = np.linspace(0, 1, 20, endpoint=False)  # 时间轴
    #
    # # 示例信号：正弦波
    # signal =np.sin(2 * np.pi * 5 * t)  # 频率为5Hz的正弦波
    # analytic_signal = hilbert(signal)
    # print(analytic_signal
    # )
    # phase = np.angle(analytic_signal)
    # # 使用时间向量作为时间点（这里时间点与采样点一一对应）
    # time_points = find_zero_crossings(phase)
    # print(
    #     f'相位过零点：{time_points}'
    # )
    # time_point1 = find_zero(phase)
    # print(
    #     f'零点：{time_point1}'
    # )
    #
    #
    # # 设备名称
    # device_name = 'Example_Device'
    #
    # # 调用函数并保存图形
    # fig = plot_hilbert_transform(signal, time_point1, device_name)
    # fig.savefig(f'{device_name}_hilbert.png')
    # plt.show()
    # # plt.close(fig)