import numpy as np
import matplotlib.pyplot as plt
# import time
from scipy import signal

#TODO
# Мониторы спайков продумать, чтобы сохранялись только времена спайков
# Алгоритм обучения
# Выходные сигналы
# Импорт в режиме реального времени с ком порта
# Монитор - шаг
# Монитор - сохранение в файл
# Выходное значение?
# Режим реального времени


class Neuron:
    def __init__(self, params):
        self.params = params
        self.U = params['U_start']
        self.I_out = params['I_out_start']
        self.t_r = 0
    
    def step(self, dt, I_in):

        params = self.params       
        U_tay = params['U_tay']
        U_th = params['U_th']
        U_rest = params['U_rest']
        I_tay = params['I_tay']

        #Расчитываем ток 
        I_out = self.I_out
        I_out *= np.exp(-dt/I_tay)
        
        t_r = self.t_r
        if (t_r > 0):
            t_r -= dt
            self.I_out = I_out
            return

        #Если не в рефракции Расчитываем напряжение
        U = self.U
        U *= np.exp(-dt/U_tay)
        U += I_in
                
        if U > U_th:
            I_out = 1
            U = U_rest
            t_r = params['refraction_time']
        if U < 0:
            U = U_rest
        
        self.U = U
        self.I_out = I_out

        
    def get_I_out(self):
        return self.I_out 
    
    def get_U(self):
        return self.U
    
    
class Layer:
    def __init__(self, name, neurons_num, neuron_params):
        self.name = name
        self.neurons_num = neurons_num
        self.neurons = [Neuron(neuron_params) for i in range(neurons_num)]
    
    def step(self, dt, I_in):
        for i, neuron in enumerate(self.neurons):
            neuron.step(dt, I_in[i])
    
    def get_I_out(self):
        I_out = np.zeros(self.neurons_num)
        for i, neuron in enumerate(self.neurons):
            I_out[i] = neuron.get_I_out()
        return I_out
    
    def get_U(self):
        U = np.zeros(self.neurons_num)
        for i, neuron in enumerate(self.neurons):
            U[i] = neuron.get_U()
        return U
        
            
class Connection:
    def __init__(self, name, source, target, weights, params):
        self.name = name
        self.params = params
        self.source = source
        self.target = target
        self.weights = weights
        
    def propagate(self, I_out_source):
        I_in_tagret = np.dot(self.weights, I_out_source)
        return I_in_tagret
    
    def learning(self, dt):
        # TODO
        params = self.params
        weights = self.weights
        
        if 'learning_coefficient' not in params:
            return
        
        I_out_source = self.source.get_I_out()
        I_out_target = self.target.get_I_out()

        # STDP
        # l_c = params['learning_coefficient']
        # a_c = params['asymmetry_coefficient']
        # for i, I_sourсe in enumerate(I_out_source):
        #     if int(I_sourсe):
        #        weights[:,i] -= (l_c * a_c * weights[:,i] * I_out_target) * dt
        # for j, I_target in enumerate(I_out_target):
        #     if int(I_target):
        #        weights[j] += l_c * (1 - weights[j]) * I_out_source * dt

        # STDP + forgetting        
        l_c = params['learning_coefficient']
        f_c = params['forgetting_coefficient']
        a_c = params['asymmetry_coefficient']
        # for i, I_sourсe in enumerate(I_out_source):
        #     if int(I_sourсe):
        #         weights[:,i] -= (l_c * a_c * weights[:,i] * I_out_target) * dt
        for j, I_target in enumerate(I_out_target):
            if int(I_target):
               weights[j] += (l_c * (1 - weights[j]) * I_out_source - f_c * weights[j]) * dt
        
        self.weights = weights
        
        
    def get_weights(self):
        return self.weights


class Network:
    def __init__(self):
        self.layers = []
        self.connections = []
        self.monitors = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def add_connection(self, connection):
        self.connections.append(connection)
        
    def add_monitor(self, monitor):
        self.monitors.append(monitor)
    
    def step(self, dt, I_external):
        I_in = I_external
            
        for connection in self.connections:
            I_out_source = connection.source.get_I_out()
            I_in_target = connection.propagate(I_out_source)
            I_in[connection.target.name] += I_in_target

        for layer in self.layers:
            layer.step(dt, I_in[layer.name])
            
        for connection in self.connections:
            connection.learning(dt)
        
        for monitor in self.monitors:
            monitor.get_data()
        
            
    def run(self, dt, I_external_all):
        for I_external in I_external_all:
            self.step(dt, I_external)


class MonitorU:
    def __init__(self, layers, step):
        self.step = step
        self.step_from_last = 0
        self.layers = layers
        self.data = {}
        for layer in layers:
            self.data[layer.name] = np.array([layer.get_U()])
    
    def get_data(self):
        step = self.step
        step_from_last = self.step_from_last + 1
        
        if step_from_last < step:
            self.step_from_last = step_from_last
            return
        else:
            step_from_last = 0
            self.step_from_last = step_from_last
        
        data = self.data
        for layer in self.layers:
            data[layer.name] = np.vstack((data[layer.name], np.array([layer.get_U()])))
        self.data = data

    def plot(self):
        data = self.data
        for layer in self.layers:
            for i in range(layer.neurons_num):
                plt.plot(data[layer.name][:, i])
                plt.title(layer.name + ' U')
            plt.show()      
            
    def clear(self):
        self.step_from_last = 0
        for layer in self.layers:
            self.data[layer.name] = np.array([layer.get_U()])
        
            
class MonitorI:
    def __init__(self, layers, step):
        self.step = step
        self.step_from_last = 0
        self.layers = layers
        self.data = {}
        for layer in layers:
            self.data[layer.name] = np.array([layer.get_I_out()])
    
    def get_data(self):
        step = self.step
        step_from_last = self.step_from_last + 1
        
        if step_from_last < step:
            self.step_from_last = step_from_last
            return
        else:
            step_from_last = 0
            self.step_from_last = step_from_last
            
        data = self.data
        for layer in self.layers:
            data[layer.name] = np.vstack((data[layer.name], np.array([layer.get_I_out()])))
        self.data = data
    
    def plot(self):
        data = self.data
        for layer in self.layers:
            for i in range(layer.neurons_num):
                plt.plot(data[layer.name][:, i])
                plt.title(layer.name + ' I_out')
            plt.show()
    
    def plot_spike(self):
        data = self.data
        for layer in self.layers:
            for i in range(layer.neurons_num):
                spikes = np.where(data[layer.name][:, i] > 0.99)[0]
                
                plt.scatter(spikes, np.full(shape=len(spikes), fill_value=i), s=1)
                plt.title(layer.name + ' spikes')
                plt.xlim(0, len(data[layer.name][:, i]))
            plt.show()
            
    def clear(self):
        self.step_from_last = 0
        for layer in self.layers:
            self.data[layer.name] = np.array([layer.get_U()])


class MonitorS:
    pass
    # TODO

class MonitorW:
    def __init__(self, connections, step):
        self.step = step
        self.step_from_last = 0
        self.connections = connections
        self.data = {}
        for connection in connections:
            self.data[connection.name] = np.array([connection.get_weights()])
    
    def get_data(self):
        step = self.step
        step_from_last = self.step_from_last + 1
        
        if step_from_last < step:
            self.step_from_last = step_from_last
            return
        else:
            step_from_last = 0
            self.step_from_last = step_from_last
            
        data = self.data
        for connection in self.connections:
            data[connection.name] = np.vstack((data[connection.name],
                                                np.array([connection.get_weights()])))
        self.data = data
        
    def plot(self):
        data = self.data
        for connection in self.connections:
            target_size = len(data[connection.name][0])
            sourse_size = len(data[connection.name][0][0])
            for i in range(sourse_size):
                for j in range(target_size):
                    plt.plot(data[connection.name][:, j, i], label=f's{i}t{j}')
                    plt.title(connection.name + 'weigth')
                    plt.legend()
            plt.show()
            
    def clear(self):
        self.step_from_last = 0
        for connection in self.connections:
            self.data[connection.name] = np.array([connection.get_weights()])


def poisson_intervals_array(N, lambda_param, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    arr = np.zeros(N)
    positions = [0]
    
    current_pos = 0
    while current_pos < N:
        # Генерируем интервал из пуассоновского распределения
        interval = np.random.poisson(lambda_param)
        
        next_pos = current_pos + interval + 1
        
        if next_pos < N:
            positions.append(next_pos)
            current_pos = next_pos
        else:
            break
    
    for pos in positions:
        if pos < N:
            arr[pos] = 1.
    
    arr[0] = 0.
    
    return arr


def high_pass_filter_scipy(data, cutoff_freq, sampling_rate, order=4):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def low_pass_filter_scipy(data, cutoff_freq, sampling_rate, order=4):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data