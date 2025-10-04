import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from neuron import LIFNeuron

neuron_params = {}
LIFNeuron(neuron_params)
