from keras.engine import Layer
from keras import initializers
from keras import backend as K
# Attention GRU network       



class MulDimLayer(Layer):
    def __init__(self, **kwargs):
        super(MulDimLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.dim = input_shape[2]
        super(MulDimLayer, self).build(input_shape)


    def call(self, inputs):
        outputs = inputs * (self.dim**0.5)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]



