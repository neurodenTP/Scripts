import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
from SNN_my_lib_V0 import Network, Layer, Connection
from SNN_my_lib_V0 import MonitorU, MonitorI, MonitorW
from SNN_my_lib_V0 import poisson_intervals_array

# 1. Создание сети
net = Network()

neuron_in_params = {
    'U_tay': 10,
    'U_th': 1.0,
    'U_rest': 0.0,
    'U_start': 0.0,
    'I_tay': 10,
    'I_out_start': 0.0
    }

neuron_out_params = {  
    'U_tay': 100,
    'U_th': 1.0,
    'U_rest': 0.0,
    'U_start': 0.0,
    'I_tay': 100,
    'I_out_start': 0.0
    }

# 2. Создание слоев
input_layer = Layer('input', 1, neuron_params=neuron_in_params)
output_layer = Layer('output', 2, neuron_params=neuron_out_params)

net.add_layer(input_layer)
net.add_layer(output_layer)

# 3. Создание связей (случайные веса)
# weights = np.random.rand(3, 2) * 1 # Small weights

connection_in_out_params = {
    'learning_coefficient': 0.001,
    'asymmetry_coefficient': 0.8,
    'forgetting_coefficient': 0.001
    }

weights_in_out = np.array([[0.45],[0.1]])
# weights_in_out = np.random.random((2,1))
# weights_in_out = np.sort(weights_in_out, axis=0)[::-1]
conn_in_out = Connection('in_out',input_layer, output_layer, 
                         weights_in_out, connection_in_out_params)
net.add_connection(conn_in_out)

connection_out_out_params = {
    }

weights_out_out = np.array([[0, -1.],[-1., 0]])
conn_out_out = Connection('out_out',output_layer, output_layer, 
                          weights_out_out, connection_out_out_params)
net.add_connection(conn_out_out)


# 4. Настройка монитора
monitorU = MonitorU([input_layer, output_layer])
net.add_monitor(monitorU)

monitorI = MonitorI([input_layer, output_layer])
net.add_monitor(monitorI)

monitorW = MonitorW([conn_in_out])
net.add_monitor(monitorW)

#%%
# Импорт данных

data = np.loadtxt("RawData/20250804_1408_data.txt")
data = data[1:-1:5]
data[:,0] = data[:,0] * 1000
N_data = len(data[:,0])
dt = (data[-1,0] - data[0,0]) / N_data

print("data N = ", N_data, " , data dt = ", dt, " ms")

# Предобработка

def high_pass_filter_scipy(data, cutoff_freq, sampling_rate, order=4):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

data_filtred = high_pass_filter_scipy(data[:,1], 1, 1000/dt)
data_abs = abs(data_filtred)
data_norm = 0.5 * data_abs / np.mean(data_abs)

plt.plot(data[:,0], data[:,1])
plt.plot(data[:,0], data_abs)
plt.show()
#%%
# 5. Запуск моделирования с данными из файла
N_calc = int(N_data)

bias_tay = 200.
bias_input = 1.1*poisson_intervals_array(N_calc, bias_tay/dt)
bias_output = 1.1*np.array([poisson_intervals_array(N_calc, bias_tay/dt),
                        poisson_intervals_array(N_calc, bias_tay/dt)]).T

learning_current = 0*1.1*np.array([data[:N_calc,2], 1 - data[:N_calc,2]]).T

I_external = [{
    'input': np.array([data_norm[i] + bias_input[i]]),
    'output': bias_output[i] + learning_current[i]
    } for i in range(N_calc)]


monitorU.clear()
monitorI.clear()
# monitorW.clear()

start_time = time.time()
net.run(dt, I_external)
print("passed_time = ", time.time() - start_time)

# 6. Построение графиков
plt.plot(data[:N_calc,0], data_norm[:N_calc])
plt.title("external current")
plt.show()

monitorU.plot()
monitorI.plot()
monitorI.plot_spike()
monitorW.plot()

#%%
N_window = int(20000/dt)
N_step = int(5000/dt)

acc = []
acc_plus = []
acc_minus = []
acc_time = []

data_time = data[:N_calc,0]
data_ref = data[:N_calc,2]
data_out = monitorI.data['output'][:, 0]
data_out = np.where(data_out > 0.5, 1, 0)

plt.plot(data_time, -data_ref)
plt.plot(data_time, data_out[1:])
plt.title("out and ref")
plt.show()

for i in range(0, N_calc - N_window, N_step):
    acc.append(sum(abs(data_out[i:i+N_window] + 
                    data_ref[i:i+N_window] - 1))/N_window)
    acc_plus.append(sum(data_out[i:i+N_window]*data_ref[i:i+N_window])/
                    sum(data_ref[i:i+N_window]))
    acc_minus.append(sum((1-data_out[i:i+N_window])*(1-data_ref[i:i+N_window]))/
                     sum(1-data_ref[i:i+N_window]))
    acc_time.append(data[int(i+N_window/2),0])

plt.plot(acc_time, acc_plus, label='plus')
plt.plot(acc_time, acc_minus, label='minus')
plt.plot(acc_time, acc, label='all')
plt.title("accuracy")
plt.legend()
plt.show()