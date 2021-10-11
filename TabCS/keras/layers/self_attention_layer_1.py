from keras.engine import Layer
from keras import initializers
from keras import backend as K
# Attention GRU network       
class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):

        assert len(input_shape)==3
        
        # W.shape = (time_steps, time_steps)

        # self.W_q = self.add_weight(name='W_q',
        #                          shape=(input_shape[2], input_shape[2]),
        #                          initializer='uniform',
        #                          trainable=True)
        # self.W_k = self.add_weight(name='w_k',
        #                          shape=(input_shape[2], input_shape[2]),
        #                          initializer='uniform',
        #                          trainable=True)
        self.W_v = self.add_weight(name='w_v',
                                 shape=(input_shape[2], input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)


        super(SelfAttentionLayer, self).build(input_shape)


    def call(self, inputs):
        V_n = K.dot(inputs, self.W_v)
        return V_n

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]



class SubLayer(Layer):
    def __init__(self, **kwargs):
        super(SubLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.dim = input_shape[2]
        super(SubLayer, self).build(input_shape)


    def call(self, inputs):
        outputs = inputs /(self.dim**0.5)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]



