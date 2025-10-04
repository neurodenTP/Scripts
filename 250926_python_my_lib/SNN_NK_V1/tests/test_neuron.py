import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import unittest
from neuron import LIFNeuron

class TestLIFNeuron(unittest.TestCase):
    def setUp(self):
        params = {
            'Ustart': 0.0,
            'Ioutstart': 0.0,
            'Utay': 10.0,
            'Uth': 1.0,
            'Urest': 0.0,
            'Itay': 10.0,
            'refractiontime': 5.0,
            'Iout_max': 1.0
        }
        self.neuron = LIFNeuron(params)

    def test_initial_values(self):
        self.assertEqual(self.neuron.U, 0.0)
        self.assertEqual(self.neuron.Iout, 0.0)

    def test_reset(self):
        self.neuron.U = 5
        self.neuron.Iout = 0.5
        self.neuron.reset()
        self.assertEqual(self.neuron.U, 0.0)
        self.assertEqual(self.neuron.Iout, 0.0)

    def test_step_no_spike(self):
        self.neuron.reset()
        self.neuron.step(1, 0.1)
        self.assertLess(self.neuron.U, self.neuron.params['Uth'])
        self.assertEqual(self.neuron.Iout, 0)

    def test_step_spike(self):
        self.neuron.reset()
        self.neuron.step(1, 5.0)
        self.assertEqual(self.neuron.Iout, self.neuron.params.get('Iout_max', 1.0))
        self.assertEqual(self.neuron.U, self.neuron.params.get('Urest', 0.0))

    def test_refractory(self):
        self.neuron.reset()
        self.neuron.step(1, 5.0)  # spike
        self.assertTrue(self.neuron.tr > 0)
        self.neuron.step(1, 5.0)
        self.assertEqual(self.neuron.Iout, 0)

if __name__ == '__main__':
    unittest.main()
