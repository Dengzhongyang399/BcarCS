from keras.engine import Layer
from keras import initializers
from keras import backend as K
# Attention GRU network       
class LayerNormLayer(Layer):
    def __init__(self,
                 center=True,
                 scale=False,
                 epsilon=None,
                 **kwargs):
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.gamma, self.beta = 0., 0.
        super(LayerNormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_1 = self.add_weight(name='w_1',
                                   shape=(input_shape[2], input_shape[2]),
                                   initializer='uniform',
                                   trainable=True)
        self.W_2 = self.add_weight(name='w_2',
                                   shape=(input_shape[2], input_shape[2]),
                                   initializer='uniform',
                                   trainable=True)
        super(LayerNormLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs_1 = (inputs - mean) / std
        outputs_2 = K.relu(K.dot(outputs_1, self.W_1))
        outputs_3 = K.dot(outputs_2, self.W_2)
        outputs_4 = K.dropout(outputs_3,level=0.25)
        outputs = outputs_1 + outputs_4

        print(outputs)
        return outputs
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]
