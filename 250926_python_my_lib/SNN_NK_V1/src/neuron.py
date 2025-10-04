import numpy as np

class Neuron:
    def __init__(self, params):
        self.params = params
        self.U = params.get('Ustart', 0.0)  # Мембранный потенциал
        self.Iout = params.get('Ioutstart', 0.0)  # Выходной ток

    def reset(self):
        self.U = self.params.get('Ustart', 0.0)
        self.Iout = self.params.get('Ioutstart', 0.0)

    def step(self, dt, Iin):
        raise NotImplementedError

    def get_Iout(self):
        return self.Iout

    def get_U(self):
        return self.U
    
    def get__is_spike(self):
        return self.is_spike


class LIFNeuron(Neuron):
    def __init__(self, params):
        super().__init__(params)
        self.itay = params.get('Itay', 10.0)
        self.utay = params.get('Utay', 10.0)
        self.uth = params.get('Uth', 1.0)
        self.urest = params.get('Urest', 0.0)
        self.iout_max = params.get('Iout_max', 1.0)
        self.refraction_time = params.get('refractiontime', 5.0)
        self.reset()

    def reset(self):
        self.U = self.params.get('Ustart', 0.0)
        self.Iout = self.params.get('Ioutstart', 0.0)
        self.is_spike = False
        self.tr = 0

    def step(self, dt, Iin):
        self.Iout *= np.exp(-dt / self.itay)
        self.is_spike = False
        
        if self.tr > 0:
            self.tr -= dt
            return

        self.U += (-self.U / self.utay) * dt + Iin

        if self.U >= self.uth:
            self.Iout = self.iout_max
            self.U = self.urest
            self.tr = self.refraction_time
            self.is_spike = True
            
class IzhikevichNeuron(Neuron):
    def __init__(self, params):
        super().__init__(params)
        self.a = params.get('a', 0.02)
        self.b = params.get('b', 0.2)
        self.c = params.get('c', -65.0)
        self.d = params.get('d', 8.0)
        self.uth = params.get('Uth', 30.0)
        self.iout_max = params.get('Iout_max', 1.0)
        self.reset()

    def reset(self):
        self.v = self.params.get('v_init', -65.0)
        self.u = self.b * self.v
        self.Iout = 0
        self.is_spike = False

    def step(self, dt, Iin):
        dv = (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + Iin) * dt
        du = self.a * (self.b * self.v - self.u) * dt

        self.v += dv
        self.u += du

        if self.v >= self.uth:
            self.v = self.c
            self.u += self.d
            self.Iout = self.iout_max
            self.is_spike = True
        else:
            self.Iout *= np.exp(-dt)
            self.is_spike = False