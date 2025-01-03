import logging
from typing import overload

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
import pandas as pd

from plotter import plot_hilbert_transform, plot_results
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
        first_point = zero_crossings[0]
        last_point = zero_crossings[-1]
        if first_point > 0:
            zero_crossings.insert(0, 0)
        if last_point < len(signal) - 1:
            zero_crossings.append(len(signal) - 1)


    if len(zero_crossings) < 2:
        logging.warning(f"Device signal has insufficient zero-crossings: {len(zero_crossings)} found.")
        return [], [],[]

    # time_points = []
    signal_diffs = []
    phases = []#记录瞬时相位

    for i in range(len(zero_crossings) - 1):
        # ime_points.append(zero_crossings[i])
        # 在零点之间的信号范围内，计算最大值和最小值的差
        signal_segment = signal[zero_crossings[i]:zero_crossings[i + 1]]
        max_value = np.max(signal_segment)
        min_value = np.min(signal_segment)
        signal_diffs.append(max_value - min_value)

        # 计算瞬时相位
        if zero_crossings is not None:
            analytic_signal = hilbert(signal)
            inst_phase = np.angle(analytic_signal)
        phases.append(inst_phase[zero_crossings[i]])

    # phases.append(np.angle(hilbert(signal)[-1]))   # 终点的瞬时相位



    return zero_crossings, signal_diffs, phases


def process_device_group(data, start_device, end_device,start_time=0):
    results = {}
    # # 处理每组设备的数据(根据原有取数据的格式修改)
    ref_device = start_device# 选择每组的第一台设备
    ref_signal = data[ref_device * 3 + 1]# 选择每组的第一台设备的vref信号
    zero_crossings, _, _ = process_signal(ref_signal)

    # 创建 time_point 列，并将 start_time 加到 zero_crossings 中的每个值
    time_point = [start_time + x for x in zero_crossings]
    # print(time_point)
    results['Time'] = time_point[:-1]
    # 处理组内每个设备(根据原有取数据的格式修改)
    for device in range(start_device, end_device):
        col_idx = device * 3 + 1
        # 处理vref信号
        signal_vref = data[col_idx] # vref的信号列(保持原有格式)
        _, signal_diffs_vref, phases_vref = process_signal(signal_vref, zero_crossings)

        # 处理vac信号
        signal_vac = data[col_idx + 1]# vac的信号列(保持原有格式)
        _, signal_diffs_vac, phases_vac = process_signal(signal_vac, zero_crossings)
        if device==start_device:
            # phases_vref[1:] = np.round(phases_vref[1:])  # 对第一台设备的相位取整
            # phases_vref = np.floor(phases_vref)  # 对第一台设备的相位向下取整
            phases_vref[1:] = np.zeros_like(phases_vref[1:])  # 将第一台设备的瞬时相位设置为0(除了起点时刻)


        device_num = device + 1
        results[f'Device{device_num}_vref_Signal_Diff'] = signal_diffs_vref
        results[f'Device{device_num}_vref_Phase'] = phases_vref
        results[f'Device{device_num}_vac_Signal_Diff'] = signal_diffs_vac
        results[f'Device{device_num}_vac_Phase'] = phases_vac
    # 在保存之前确保所有数据长度一致（两种方式）
    # 补nan
    # max_length = max(len(v) for v in results.values())
    # for key in results:
    #     if len(results[key])< max_length:
    #         results[key] = results[key] + [np.nan] * (max_length - len(results[key]))
    # 截取
    min_length = len(results['Time'])
    for key in results:
        if len(results[key]) > min_length:
            logging.warning(f" {key} has something wrong,truncating {key} from {len(results[key])} to {min_length}")
            results[key] = results[key][:min_length]

    df = pd.DataFrame(results)
    df.to_csv(f'results_devices_{start_device + 1}_to_{end_device}.csv', index=False)
    logging.info(f"Results for devices_{start_device + 1}_to_{end_device} saved successfully.")

    return df

# def main(data, types, devices):
#     results = {}
#     # 处理每个设备的数据(保持原有取数据的格式)
#     for device in range(devices):
#         col_idx = device * 3 + 1
#         signal_vref = data[col_idx]  # vref的信号列(保持原有格式)
#         zero_crossings, signal_diffs_vref = process_signal(signal_vref)
#         signal_vac = data[col_idx + 1]  # vac的信号列(保持原有格式)
#         zero_crossings, signal_diffs_vac = process_signal(signal_vac, zero_crossings)
#         time_points = zero_crossings[:-1]
#
#         device_num = device + 1# 设备号（保持原有格式)
#         col_name_time = f'Device{device_num}_vref_Time'
#         col_name_diff1 = f'Device{device_num}_vref_Signal_Diff'
#         col_name_diff2 = f'Device{device_num}_vac_Signal_Diff'
#         results[col_name_time] = time_points
#         results[col_name_diff1] = signal_diffs_vref
#         results[col_name_diff2] = signal_diffs_vac
#
#         plot_results(device_num,time_points ,signal_diffs_vref,signal_diffs_vac)
#         # print(time_points)
#         # print(signal_diffs_vref)
#         # print(signal_diffs_vac)
#
#     max_length = max(len(v) for v in results.values())
#     for key in results:
#         results[key] = results[key] + [np.nan] * (max_length - len(results[key]))
#
#     df = pd.DataFrame(results)
#     df.to_csv('results.csv', index=False)
#     logging.info("Results saved successfully.")

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

    data = np.hstack(data)
    print(data)

    types = ['type1']  # 信号类型
    devices = 24  # 24个设备

    # 调用 main 函数并传入数据
    # main(data, types, devices)
    #将设备分组处理
    device_groups = [(0, 6), (6, 12), (12, 18), (18, 24)]
    for start, end in device_groups:
        process_device_group(data, start, end,200)


