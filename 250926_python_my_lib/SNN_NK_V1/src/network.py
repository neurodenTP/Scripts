import numpy as np


class Network:
    def __init__(self):
        self.layers = {}        # словарь: имя слоя -> Layer
        self.connections = {}   # словарь: имя соединения -> Connection
        self.monitors = {}

    def add_layer(self, layer):
        if layer.name in self.layers:
            raise ValueError(f"Слой с именем {layer.name} уже существует")
        self.layers[layer.name] = layer

    def add_connection(self, connection):
        if connection.name in self.connections:
            raise ValueError(f"Соединение с именем {connection.name} уже существует")
        if (connection.target_layer.name not in self.layers or 
            connection.source_layer.name not in self.layers):
            raise ValueError("Указанные слои должны существовать в сети")
        self.connections[connection.name] = connection
    
    def add_monitor(self, monitor):
        if monitor.name in self.monitors:
            raise ValueError(f"Монитор с именем {monitor.name} уже существует")
        self.monitors[monitor.name] = monitor
        
    def remove_monitor(self, name):
        if name in self.monitors:
            del self.monitors[name]
    

    def reset(self):
        for layer in self.layers.values():
            layer.reset()

    def step(self, dt, I_external):
        """
        Шаг времени dt.
        I_external - словарь {имя_слоя: входные внешние токи (numpy массив)}
        """
        # Инициализируем входы нейронов каждого слоя внешними токами
        I_in = {}
        for layer_name, layer in self.layers.items():
            # Создаем копии внешних токов, чтобы их модифицировать дальше
            I_in[layer_name] = np.array(I_external.get(layer_name, np.zeros(layer.neurons_num)))
    
        # Добавляем входы от соединений
        for connection in self.connections.values():
            # Получаем выходные токи исходного слоя
            I_out_source = np.array(connection.source_layer.get_outputs())
            # Прогоняем их через веса соединения
            I_in_target = connection.propagate(I_out_source)
            # Складываем с текущими входными токами целевого слоя
            I_in[connection.target_layer.name] += I_in_target
    
        # Делаем шаг для каждого слоя с суммарным входом
        for layer_name, layer in self.layers.items():
            layer.step(dt, I_in[layer_name])
    
        # Производим обучение (если реализовано) для всех соединений
        for connection in self.connections.values():
            if hasattr(connection, 'learning'):
                connection.update_weights(dt)
                
        # Коллекция данных мониторами
        for monitor in self.monitors.values():
            monitor.collect()

    def run(self, dt, inputs):
        """
        Прогон всей сети по временным шагам.
    
        Параметры:
        dt - шаг времени
        inputs - словарь {имя_слоя: np.array формы (num_steps, num_neurons)}
    
        Возвращает:
        outputs - словарь {имя_слоя: np.array выхода формы (num_steps, num_neurons)}
        """
        num_steps = None
        for inp in inputs.values():
            if num_steps is None:
                num_steps = inp.shape[0]
            elif inp.shape[0] != num_steps:
                raise ValueError("Все входы должны иметь одинаковое число временных шагов")
    
        for t in range(num_steps):
            inputs_t = {layer: inputs[layer][t] for layer in inputs}
            self.step(dt, inputs_t)
   