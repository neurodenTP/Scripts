import numpy as np
import matplotlib.pyplot as plt
import time
from SNN_my_lib_V0 import Network, Layer, Connection
from SNN_my_lib_V0 import MonitorU, MonitorI, MonitorW
from SNN_my_lib_V0 import poisson_intervals_array
from SNN_my_lib_V0 import high_pass_filter_scipy, low_pass_filter_scipy

#%%
def data_import(name):
    data = np.loadtxt(name)
    data = data[1:-1:5]
    data[:,0] = data[:,0] * 1000
    
    N_data = len(data)
    dt = (data[-1, 0] - data[0, 0]) / N_data
    print("data N = ", N_data, " , data dt = ", dt, " ms")
    
    data_filtred = high_pass_filter_scipy(data[:,1], 1, 1000/dt)
    data_abs = abs(data_filtred)
    data_norm = 0.15 * data_abs / np.mean(data_abs)
    data[:,1] = data_norm
    
    plt.plot(data[:,0], data[:,1])
    plt.show()
    
    return data, N_data, dt


def calc_data_out(data_time, _data_out, dt):
    data_out = 1.0*_data_out
    for i in range(len(data_out)-1):
        n_filter = 20
        data_out[i+1] = ((n_filter - 1)*data_out[i] + data_out[i+1]) / n_filter

    data_out = np.where(data_out > 0.3, 1, 0)
    return data_out[1:]

def plot_ref_out(data_time, data_ref, data_out, dt):
    plt.scatter(data_time, data_ref + 0.02, s=1, label='ref')
    plt.scatter(data_time, data_out, s=1, label='out')
    plt.title("button")
    plt.legend()
    plt.show()
    

def plot_accuracy(data_time, data_ref, data_out, dt):
    N_window = int(10000/dt)
    N_step = int(5000/dt)
    
    acc = []
    acc_plus = []
    acc_minus = []
    acc_time = []
    
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

def calc_total_accuracy(data_ref, data_out):
    acc_plus = sum(data_out*data_ref) / sum(data_ref)
    acc_minus = sum((1-data_out)*(1-data_ref)) / sum(1-data_ref)
    acc = sum(abs(data_out + data_ref - 1))/len(data_ref)
    return acc_plus, acc_minus, acc
#%%
# 1. Создание сети
net = Network()

neuron_in_params = {
    'U_tay': 100,
    'U_th': 1.0,
    'U_rest': 0.0,
    'U_start': 0.0,
    'I_tay': 20,
    'I_out_start': 0.0,
    'refraction_time': 20
    }

neuron_out_params = {  
    'U_tay': 1000,
    'U_th': 1.0,
    'U_rest': 0.0,
    'U_start': 0.0,
    'I_tay': 100,
    'I_out_start': 0.0,
    'refraction_time': 1000
    }

# 2. Создание слоев
input_layer = Layer('input', 2, neuron_params=neuron_in_params)
output_layer = Layer('output', 2, neuron_params=neuron_out_params)

net.add_layer(input_layer)
net.add_layer(output_layer)

# 3. Создание связей (случайные веса)
# weights = np.random.rand(3, 2) * 1 # Small weights

connection_in_out_params = {
    'learning_coefficient': 0.0002,
    'asymmetry_coefficient': 0.8,
    'forgetting_coefficient': 0.0002
    }

# weights_in_out = np.array([[0.45],[0.1]])
weights_in_out = np.random.random((2,2))
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

weights_in_in = np.array([[0, -1.],[-1., 0]])
conn_in_in = Connection('in_in',input_layer, input_layer, 
                          weights_in_in, connection_out_out_params)
net.add_connection(conn_in_in)


# 4. Настройка монитора
# monitorU = MonitorU([input_layer, output_layer], 1)
# net.add_monitor(monitorU)

monitorI = MonitorI([input_layer, output_layer], 1)
net.add_monitor(monitorI)

monitorW = MonitorW([conn_in_out], 100)
net.add_monitor(monitorW)

total_acc = np.empty((0, 3), dtype=int)
#%%
# Импорт данных
# data = np.loadtxt("RawData/20250804_1408_data.txt") # good 2
# data = np.loadtxt("RawData/20250804_1358_data.txt") # top 1
# data = np.loadtxt("RawData/20250804_1350_data.txt") # good 3 (long learning)
# data, N_data, dt = data_import("RawData/20250804_1342_data.txt") # top 1
# data = np.loadtxt("RawData/20250804_1335_data.txt") # bad low mio reaction 4
# data = np.loadtxt("RawData/20250804_1326_data.txt") # bad low mio reaction 4
# data = np.loadtxt("RawData/20250804_1258_data.txt") # bad low mio reaction 4
# data = np.loadtxt("RawData/20251002_1329_data.txt") # no activity
# data, N_data, dt = data_import("RawData/20251002_1348_data.txt") # noise no ground
# data, N_data, dt = data_import("RawData/20251002_1353_data.txt") # fast every 5s
data, N_data, dt = data_import("RawData/20251002_1358_data.txt") # 5s-10s
# data, N_data, dt = data_import("RawData/20251002_1343_data.txt") # 5s-5s


#%%
# 5. Подготовка входного тока
N_calc = int(N_data/10)

bias_tay = 400.
bias_input = 0.4*poisson_intervals_array(N_calc, bias_tay/dt/30)
bias_output = 2.5*np.array([poisson_intervals_array(N_calc, 5*bias_tay/dt),
                        poisson_intervals_array(N_calc, 5*bias_tay/dt)]).T

learning_current = 0*1.1*np.array([data[:N_calc,2], 1 - data[:N_calc,2]]).T

I_external = [{
    'input': np.array([data[i, 1], bias_input[i]]),
    'output': bias_output[i] + learning_current[i]
    } for i in range(N_calc)]
#%%
for i in range(1):
    # monitorU.clear()
    monitorI.clear()
    # monitorW.clear()
    
    start_time = time.time()
    net.run(dt, I_external)
    print("passed_time = ", time.time() - start_time)
    
    # 6. Построение графиков
    plt.plot(data[:N_calc,0], data[:N_calc,1])
    plt.plot(data[:N_calc,0], monitorI.data['output'][:N_calc, 1])
    plt.title("external current")
    plt.show()
    
    # monitorU.plot()
    monitorI.plot()
    # monitorI.plot_spike()
    
    
    
    data_out = calc_data_out(data[:N_calc,0], monitorI.data['output'][:, 1], dt)
    plot_ref_out(data[:N_calc,0], data[:N_calc,2], data_out, dt)
    # plot_accuracy(data[:N_calc,0], data[:N_calc,2], data_out, dt)
    total_acc = np.append(total_acc, np.array([calc_total_accuracy(data[:N_calc,2], data_out)]), axis=0)
    print(total_acc[-1])

monitorW.plot()
plt.plot(total_acc[:, 0], label="acc_plus")
plt.plot(total_acc[:, 1], label="acc_minus")
plt.plot(total_acc[:, 2], label="acc_all")
plt.legend()
plt.show()
