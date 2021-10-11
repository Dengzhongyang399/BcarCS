from keras.engine import Layer
from keras import initializers
from keras import backend as K
# Attention GRU network       
class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):

        assert len(input_shape)==3
        self.dim = input_shape[2]
        self.n_words = input_shape[1]
        self.num_heads = 10
        # W.shape = (time_steps, time_steps)

        self.W_q = self.add_weight(name='W_q',
                                 shape=(input_shape[2], input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)
        self.W_k = self.add_weight(name='w_k',
                                 shape=(input_shape[2], input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)
        self.W_v = self.add_weight(name='w_v',
                                 shape=(input_shape[2], input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)
        self.W_1 = self.add_weight(name='w_1',
                                   shape=(input_shape[2], input_shape[2]),
                                   initializer='uniform',
                                   trainable=True)

        super(SelfAttentionLayer, self).build(input_shape)


    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        Q_n = K.dot(inputs, self.W_q)
        V_n = K.dot(inputs, self.W_v)
        K_n = K.dot(inputs, self.W_k)

        Q_n = K.permute_dimensions(K.reshape(Q_n, (-1, self.n_words, self.num_heads, int(self.dim/self.num_heads))),
                                   (0, 2, 1, 3))
        V_n = K.permute_dimensions(K.reshape(V_n, (-1, self.n_words, self.num_heads, int(self.dim / self.num_heads))),
                                   (0, 2, 1, 3))
        K_n = K.permute_dimensions(K.reshape(K_n, (-1, self.n_words, self.num_heads, int(self.dim / self.num_heads))),
                                   (0, 2, 1, 3))

        A = K.softmax(K.batch_dot(Q_n, K.permute_dimensions(K_n, (0, 1, 3, 2))) / (self.dim ** 0.5))
        context_n = K.batch_dot(A, V_n)
        context_n = K.permute_dimensions(context_n, (0, 2, 1, 3))
        context_n = K.reshape(context_n, (-1, self.n_words, self.dim))
        outputs = K.dot(context_n, self.W_1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]



