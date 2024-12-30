import logging
from typing import overload

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
import pandas as pd

from plotter import plot_hilbert_transform
from typing import List
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# 生成一个测试信号：正弦波 + 随机噪声
def generate_test_signal(duration=3, fs=2000, noise_level=0.5):
    # 生成时间序列（3秒，采样频率2000Hz）
    t = np.linspace(0, duration, int(fs * duration))

    # 正弦波信号：50Hz的正弦波
    sine_wave = np.sin(2 * np.pi * 50 * t)

    # 随机噪声
    noise = noise_level * np.random.randn(len(t))

    # 合成信号：正弦波 + 噪声
    signal = sine_wave + noise

    return t, signal

def butter_lowpass(cutoff, fs, order=4):
    """设计一个低通Butterworth滤波器"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=4):
    """对数据应用低通滤波器"""
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def find_zero_crossings(phase):
    zero_crossings = []
    seen_indices = set()  # 用于存储已经添加的索引，确保不重复
    for i in range(len(phase) - 1):
        if (phase[i] <= 0 and phase[i + 1] > 0) or (phase[i] >= 0 and phase[i + 1] < 0):
            # 找出哪个点（索引）更接近0
            if abs(phase[i]) <= abs(phase[i + 1]):
                point = i
            else:
                point = i + 1
            if point not in seen_indices:  # 检查索引是否已存在
                zero_crossings.append(point)
                seen_indices.add(point)  # 添加新索引到已见索引集合
    return zero_crossings
def find_zero(phase):
    threshold = 1e-3  # 设置一个阈值来判断瞬时相位是否接近零
    zero_phase_indices = np.where(np.abs(phase) < threshold)[0]
    return zero_phase_indices

def process_signal(signal, zero_crossings: List[int] = None):
    fs = 2000  # 采样频率 2000 Hz
    cutoff = 100  # 截止频率 100 Hz
    # 降噪：首先对信号进行低通滤波
    signal = lowpass_filter(signal, cutoff, fs)

    # 如果没有提供零交叉点，则计算零交叉点
    if zero_crossings is None:
        analytic_signal = hilbert(signal)
        phase = np.angle(analytic_signal)
        zero_crossings = find_zero_crossings(phase)

    if len(zero_crossings) < 2:
        logging.warning(f"Device signal has insufficient zero-crossings: {len(zero_crossings)} found.")
        return [], []

    time_points = []
    signal_diffs = []

    for i in range(len(zero_crossings) - 1):
        time_points.append(zero_crossings[i])
        # 在零点之间的信号范围内，计算最大值和最小值的差
        signal_segment = signal[zero_crossings[i]:zero_crossings[i + 1]]
        max_value = np.max(signal_segment)
        min_value = np.min(signal_segment)
        signal_diffs.append(max_value - min_value)

    return zero_crossings, signal_diffs

def main(data, types, devices):
    results = {}
    # 处理每个设备的数据
    for device in range(devices):
        col_idx = device * 3 + 1
        signal_vref = data[col_idx]  # vref的信号列
        zero_crossings, signal_diffs_vref = process_signal(signal_vref)
        signal_vac = data[col_idx + 1]  # vac的信号列
        zero_crossings, signal_diffs_vac = process_signal(signal_vac, zero_crossings)
        time_points = zero_crossings[:-1]

        device_num = device + 1
        col_name_time = f'Device{device_num}_vref_Time'
        col_name_diff1 = f'Device{device_num}_vref_Signal_Diff'
        col_name_diff2 = f'Device{device_num}_vac_Signal_Diff'
        results[col_name_time] = time_points
        results[col_name_diff1] = signal_diffs_vref
        results[col_name_diff2] = signal_diffs_vac

    max_length = max(len(v) for v in results.values())
    for key in results:
        results[key] = results[key] + [np.nan] * (max_length - len(results[key]))

    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    logging.info("Results saved successfully.")

if __name__ == "__main__":
    # 生成一个测试信号
    t, signal = generate_test_signal()

    # 模拟数据传入 main 函数的格式
    # 假设我们有24个设备的数据，每个设备有vref和vac两列信号
    data = []
    for i in range(24):  # 假设24个设备
        vref_signal = signal + 0.2 * np.random.randn(len(signal))  # vref信号带有轻微噪声
        vac_signal = signal + 0.3 * np.random.randn(len(signal))  # vac信号带有稍微大的噪声
        data.append(np.column_stack([t, vref_signal, vac_signal]))  # 结合t,vref和vac信号

    data = np.hstack(data)  # 将所有设备的数据合并为一个大数组

    types = ['type1']  # 信号类型
    devices = 24  # 24个设备

    # 调用 main 函数并传入数据
    main(data, types, devices)