import tensorflow as tf
from keras import backend, initializers
from keras.layers import Layer


class GEV(Layer):

    def __init__(self, **kwargs):
        super(GEV, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(1,),
                                  initializer=initializers.constant(0.),
                                  trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                     shape=(1,),
                                     initializer=initializers.constant(1.),
                                     trainable=True)
        self.xi = self.add_weight(name='xi',
                                  shape=(1,),
                                  initializer=initializers.constant(0.),
                                  trainable=True)
        super(GEV, self).build(input_shape)

    def call(self, x):
        sigma = backend.maximum(backend.epsilon(), self.sigma)  # Assert sigma>0

        # Type 1: For xi = 0 (Gumbel)
        def t1(x=x, mu=self.mu, sigma=sigma):
            
            y = -(x-mu)/sigma
            return backend.exp(-backend.exp(y))

        # Type 2: For xi>0 (Frechet) or xi<0 (Reversed Weibull) 
        def t23(x=x, mu=self.mu, sigma=sigma, xi=self.xi):
            y = (x - self.mu) / sigma
            cdf = backend.exp(
                -tf.pow(
                    backend.maximum(1 + self.xi * y, backend.epsilon()),  # Assert xi*y>-1
                    -1 / self.xi)
            )
            return cdf

        return tf.cond(backend.equal(tf.constant(0.), self.xi), t1, t23)  # This chooses the type based on xi

    def compute_output_shape(self, input_shape):
        return input_shape