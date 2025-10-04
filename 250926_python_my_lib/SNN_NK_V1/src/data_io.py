import numpy as np
import serial
import queue
import threading
from scipy.signal import butter, filtfilt


def read_emg_from_file(filename, delimiter=None):
    """
    Считает данные из файла, возвращая numpy массив.
    """
    data = np.loadtxt(filename, delimiter=delimiter)
    return data

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Создаёт коэффициенты фильтра полосового пропускания.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Применяет полосовой фильтр к входным данным.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data, axis=0)
    return y