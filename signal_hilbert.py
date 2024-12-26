import logging

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
import pandas as pd

from plotter import plot_hilbert_transform

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')




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

def process_signal(signal):
    fs = 2000  # 采样频率 2000 Hz
    cutoff = 100  # 截止频率 100 Hz
    # 降噪：首先对信号进行低通滤波
    signal = lowpass_filter(signal, cutoff, fs)
    # 复值信号
    analytic_signal = hilbert(signal)
    # 瞬时相位
    phase = np.angle(analytic_signal)
    zero_crossings = find_zero_crossings(phase)
    # 复信号的虚部
    # imag = np.imag(analytic_signal)
    # zero_crossings = find_zero_crossings(imag)

    if len(zero_crossings) < 2:
        logging.warning(f"Device signal has insufficient zero-crossings: {len(zero_crossings)} found.")
        return [], []

    time_points = []
    signal_diffs = []

    for i in range(len(zero_crossings) - 1):
        time_points.append(zero_crossings[i])
        signal_diffs.append(signal[zero_crossings[i + 1]] - signal[zero_crossings[i]])

    return time_points, signal_diffs


def main(data,types ,devices):
    results = {}
    # 处理每个设备的数据
    # for device in range(24):
    #     for type_idx, type_name in enumerate(types):
    for type_idx, type_name in enumerate(types):
        for device in range(devices):
            col_idx = device * 3 + type_idx
            signal = data
            time_points, signal_diffs = process_signal(signal)
            plot_hilbert_transform(signal, time_points,'a')

            # 创建列名
            device_num = device + 1
            col_name_time = f'Device{device_num}_{type_name}_Time'
            col_name_diff = f'Device{device_num}_{type_name}_Signal_Diff'

            results[col_name_time] = time_points
            results[col_name_diff] = signal_diffs

    max_length = max(len(v) for v in results.values())
    for key in results:
        results[key] = results[key] + [np.nan] * (max_length - len(results[key]))

    df = pd.DataFrame(results)

    df.to_csv('results.csv', index=False)
    logging.info("Results saved successfully.")


if __name__ == "__main__":
    # data = np.loadtxt('data.csv', delimiter=',')
    # types = ['Iac','Vac',  'rcef']
    # main(data,types ,24)
    # 生成一个测试信号：正弦波 + 高频噪声
    t = np.linspace(0, 6000 * 0.0005, 6000)  # 时间轴（6000个点，每个时间点0.5ms）
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(6000)  # 50Hz的正弦波加噪声
    types = ['type1']
    devices = 1
    main(signal , types, devices)