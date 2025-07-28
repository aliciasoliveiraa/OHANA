import tensorflow as tf

class TFFedProxLoss(tf.keras.losses.Loss):
    def __init__(self, mu=0.01, name="fedprox_loss"):
        super().__init__(name=name)
        if mu < 0.0:
            raise ValueError("mu should be no less than 0.0")
        self.mu = mu

    def call(self, input_model, target_model):
        prox_loss = 0.0        
        for input_param, target_param in zip(input_model.variables, target_model.variables):
            prox_loss += (self.mu / 2) * tf.reduce_sum(tf.square(input_param - target_param))
        
        return prox_loss
