class Layer:
    def __init__(self, name, neurons_num, neuron_class, neuron_params):
        self.name = name
        self.neurons_num = neurons_num
        # Создаем список нейронов указанного класса с параметрами
        self.neurons = [neuron_class(neuron_params) for _ in range(neurons_num)]

    def reset(self):
        # Сброс состояния всех нейронов слоя
        for neuron in self.neurons:
            neuron.reset()

    def step(self, dt, Iin):
        """
        Выполнить один временной шаг для всего слоя.
        Iin - входной сигнал слоя.
        """
        if not hasattr(Iin, '__len__'):
            raise TypeError(f"Iin должен быть списком или массивом длины {self.neurons_num}, получено скалярное значение")
        if len(Iin) != self.neurons_num:
            raise ValueError(f"Длина входного сигнала Iin ({len(Iin)}) не соответствует количеству нейронов ({self.neurons_num})")

        for neuron in self.neurons:
            neuron.step(dt, Iin)

    def get_outputs(self):
        # Получить текущие выходные токи всех нейронов слоя
        return [neuron.get_Iout() for neuron in self.neurons]

    def get_states(self):
        # Получить текущие мембранные потенциалы всех нейронов слоя
        return [neuron.get_U() for neuron in self.neurons]
    
    def get_spikes(self):
        return [neuron.get_is_spike() for neuron in self.neurons]