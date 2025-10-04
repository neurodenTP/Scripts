import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MonitorLayer(ABC):
    def __init__(self, name, layers, save_step=1, max_points=None):
        """
        name: имя монитора
        layers: Layer или список слоев для мониторинга
        save_step: сохранять данные каждый n-й вызов collect
        max_points: максимальное количество последних точек для хранения (None - без ограничения)
        """
        self.name = name
        if not isinstance(layers, list):
            layers = [layers]
        self.layers = layers
        self.save_step = max(1, save_step)
        self.max_points = max_points
        self.counter = 0

        self.data = {layer.name: [] for layer in self.layers}

    @abstractmethod
    def _get_layer_data(self, layer):
        """Определяет что именно сохранять (потенциалы, токи, спайки)."""
        pass

    def collect(self):
        self.counter += 1
        if self.counter % self.save_step != 0:
            return
        for layer in self.layers:
            datum = self._get_layer_data(layer)
            points = self.data[layer.name]
            points.append(datum)
            if self.max_points is not None and len(points) > self.max_points:
                points.pop(0)
                
    def get_data(self, layer_name):
        if layer_name not in self.data:
            raise ValueError(f"Данные для слоя {layer_name} не собраны")
        return self.data[layer_name]

    def clear(self):
        self.data = {layer.name: [] for layer in self.layers}

    @abstractmethod
    def plot(self, layer_name, dt):
        pass


class PotentialMonitor(MonitorLayer):
    def _get_layer_data(self, layer):
        return layer.get_states()

    def get_data(self, layer_name):
        return np.array(super().get_data(layer_name))

    def plot(self, layer_name, dt):
        data = self.get_data(layer_name)
        times = (self.counter - len(data) + np.arange(len(data))) * dt
        plt.imshow(data.T, extent=[times[0], times[-1], 0, data.shape[1]], 
                   aspect='auto', origin='lower', interpolation='none')
        plt.colorbar()
        plt.xlabel('Время (мс)')
        plt.ylabel('Нейроны')
        plt.title(f"Потенциалы слоя {layer_name}")
        plt.show()

class CurrentMonitor(MonitorLayer):
    def _get_layer_data(self, layer):
        return layer.get_outputs()

    def get_data(self, layer_name):
        return np.array(super().get_data(layer_name))

    def plot(self, layer_name, dt):
        data = self.get_data(layer_name)
        times = (self.counter - len(data) + np.arange(len(data))) * dt
        plt.imshow(data.T, extent=[times[0], times[-1], 0, data.shape[1]], 
                   aspect='auto', origin='lower', interpolation='none')
        plt.colorbar()
        plt.xlabel('Время (мс)')
        plt.ylabel('Нейроны')
        plt.title(f"Токи слоя {layer_name}")
        plt.show()


class SpikeMonitor(MonitorLayer):
    def _get_layer_data(self, layer):
        outputs = layer.get_spikes()
        return [i for i, val in enumerate(outputs) if val]

    def plot(self, layer_name, dt):
        data = self.get_data(layer_name)
        plt.figure()
        for t, spikes in enumerate(data):
            y = np.ones(len(spikes)) * (self.counter - len(data) + t) * dt
            x = spikes
            plt.scatter(x, y, marker='|')
        plt.xlabel('Нейроны')
        plt.ylabel('Время (мс)')
        plt.title(f"Спайки слоя {layer_name}")
        plt.show()
        

class MonitorConnection:
    def __init__(self, name, connections, save_step=1, max_points=None):
        """
        name: имя монитора
        connections: Connection или список соединений для мониторинга
        save_step: сохранять данные каждый n-й вызов collect
        max_points: максимальное количество последних точек для хранения (None - без ограничения)
        """
        self.name = name
        if not isinstance(connections, list):
            connections = [connections]
        self.connections = connections
        self.save_step = max(1, save_step)
        self.max_points = max_points
        self.counter = 0
        self.data = {conn.name: [] for conn in self.connections}

    def collect(self):
        self.counter += 1
        if self.counter % self.save_step != 0:
            return
        for conn in self.connections:
            weights_copy = conn.weights.copy()
            points = self.data[conn.name]
            points.append(weights_copy)
            if self.max_points is not None and len(points) > self.max_points:
                points.pop(0)

    def get_data(self, connection_name):
        if connection_name not in self.data:
            raise ValueError(f"Данные для соединения {connection_name} не собраны")
        return self.data[connection_name]

    def plot(self, connection_name, dt):
        data = self.get_data(connection_name)
        arr = np.array(data)
        if arr.ndim == 3:
            # Среднее по времени
            weights = np.mean(arr, axis=0)
        else:
            weights = arr

        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Нейроны исходного слоя')
        plt.ylabel('Нейроны целевого слоя')
        timespan = (self.counter - len(data), self.counter) if self.max_points else (0, len(data))
        plt.title(f"Весы соединения {connection_name}\nПоследние {len(data)} шагов (dt={dt})")
        plt.show()