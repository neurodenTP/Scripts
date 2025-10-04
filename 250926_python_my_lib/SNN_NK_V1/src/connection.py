import numpy as np

class Connection:
    def __init__(self, name, source_layer, target_layer,
                 learning_class, learning_params, weight_matrix=None):
        self.name = name
        self.source_layer = source_layer
        self.target_layer = target_layer
        
        if learning_class == None:
            self.learning = None
        else:
            self.learning = learning_class(learning_params)

        if weight_matrix is None:
            self.weights = self._generate_random_weights()
        else:
            if weight_matrix.shape != (target_layer.neurons_num, source_layer.neurons_num):
                raise ValueError("Размер weight_matrix не соответствует количествам нейронов в слоях")
            self.weights = weight_matrix

    def _generate_random_weights(self):
        return np.random.normal(0, 1, (self.target_layer.neurons_num, 
                                       self.source_layer.neurons_num))

    def propagate(self):
        source_output = np.array(self.source_layer.get_outputs())
        return np.dot(self.weights, source_output)

    def update_weights(self, dt):
        self.weights += self.learning.rule(self.weights, 
                                           self.source_layer.get_outputs,
                                           self.target_layer.get_outputs)

    def reset_weights(self, new_weights=None):
        if new_weights is None:
            self.weights = self._generate_random_weights()
        else:
            if new_weights.shape != self.weights.shape:
                raise ValueError("Форма new_weights должна совпадать с формой weights")
            self.weights = new_weights

        