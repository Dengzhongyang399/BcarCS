from keras import backend as K
from keras.engine.topology import Layer

class JointSelfAttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(JointSelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('An attention layer should be called '
                             'on a list of 2 inputs.')
        if not input_shape[0][2] == input_shape[1][2]:
            raise ValueError('Embedding sizes should be of the same size')
        self.dim=input_shape[0][2]
        self.W_qc = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),         #(dim,dim)
                                      initializer='glorot_uniform',
                                      name='W_qc',
                                      trainable=True)
        self.W_vc = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),  # (dim,dim)
                                    initializer='glorot_uniform',
                                    name='W_vc',
                                    trainable=True)
        self.W_kd = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),  # (dim,dim)
                                    initializer='glorot_uniform',
                                    name='W_kd',
                                    trainable=True)
        self.W_vd = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),  # (dim,dim)
                                    initializer='glorot_uniform',
                                    name='W_vd',
                                    trainable=True)

        super(JointSelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        Q_c = K.dot(inputs[0], self.W_qc)
        V_c = K.dot(inputs[0], self.W_vc)
        K_d = K.dot(inputs[1], self.W_kd)
        V_d = K.dot(inputs[1], self.W_vd)
        A = K.softmax(K.batch_dot(Q_c, K.permute_dimensions(K_d, (0, 2, 1))))/(self.dim**0.5)
        c = K.batch_dot(A, V_d)
        d = K.batch_dot(K.permute_dimensions(A, (0, 2, 1)), V_c)


        C = K.mean(c, axis=1, keepdims=False)
        D = K.mean(d, axis=1, keepdims=False)

        return [C, D]


    
    def compute_output_shape(self, input_shape):

        return [(None, input_shape[0][2]), (None, input_shape[1][2])]
