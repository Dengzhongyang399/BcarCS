from keras.engine import Layer
from keras import initializers
from keras import backend as K
import tensorflow as tf
# Attention GRU network
import numpy as np
class PositionEncodingLayer(Layer):
    def __init__(self, **kwargs):
        super(PositionEncodingLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        self.dim = input_shape[2]
        self.n_words = input_shape[1]
        super(PositionEncodingLayer, self).build(input_shape)

    def call(self, inputs):
        angle_rates = positional_encoding(self.n_words, self.dim)

        output = inputs+K.cast_to_floatx(angle_rates)
        return output
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]

def positional_encoding(pos, d_model):
    def get_angles(position, i):
        # shape=[position_num, d_model]
        return position / np.power(10000., 2. * (i // 2.) / np.float32(d_model))

    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])
    # 2i:sin，2i+1:cos
    angle_rates[:, 0::2] = np.sin(angle_rates[:, 0::2])
    angle_rates[:, 1::2] = np.cos(angle_rates[:, 1::2])
    return angle_rates[np.newaxis, ...]

class PositionEmbedding(Layer):
    """定义可训练的位置Embedding
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            merge_mode='add',
            hierarchical=None,
            embeddings_initializer='zeros',
            custom_position_ids=False,
            **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim  # 输入维度max_position
        self.output_dim = output_dim  # 输出维度embedding_size，bert中用的是768
        self.merge_mode = merge_mode  # add模式或者mul模式
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )  # 初始化待训练的位置编码权重

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]
        # 自己输入位置编码及其位置id
        if self.custom_position_ids:  
            inputs, position_ids = inputs
            if K.dtype(position_ids) != 'int32':
                position_ids = K.cast(position_ids, 'int32')
        else:
            # 得到位置编码id 加了[None]变成两维的 [[0,1,2,...,seq_len]]
            position_ids = K.arange(0, seq_len, dtype='int32')[None]  

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = K.gather(embeddings, position_ids // self.input_dim)
            embeddings_y = K.gather(embeddings, position_ids % self.input_dim)
            pos_embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            # 如果是自己输入位置编码，就用位置id读取相应的位置编码
            if self.custom_position_ids:  
                pos_embeddings = K.gather(self.embeddings, position_ids)
            else:
                # 直接拿初始化的位置编码权重
                pos_embeddings = self.embeddings[None, :seq_len]
        # add模式直接把原有特征和位置编码相加即可
        if self.merge_mode == 'add':  
            return inputs + pos_embeddings
        # mul模式是把原有特征和位置编码对应相乘
        elif self.merge_mode == 'mul':  
            return inputs * pos_embeddings
        else:
            if not self.custom_position_ids:
                pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])
            # 如果不属于上述两种模式，则用concat的形式
            return K.concatenate([inputs, pos_embeddings])  

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul']:
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SinusoidalPositionEmbedding(Layer):
    """定义Sin-Cos位置Embedding
    """

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False, **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        input_shape = K.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]

        if self.custom_position_ids:
            inputs, position_ids = inputs
        else:
            # 得到位置编码id 加了[None]变成两维的 [[0,1,2,...,seq_len]]
            position_ids = K.arange(0, seq_len, dtype=K.floatx())[None]
        # 根据公式开始计算
        # 取一半的，方便2i的计算
        indices = K.arange(0, self.output_dim // 2, dtype=K.floatx())
        # 对前一个参数x，取后一个参数y的平方，x^y，即10000^(2i/dim)
        indices = K.pow(10000.0, -2 * indices / self.output_dim)

        # shape=(btz, seq_len, dim)
        pos_embeddings = tf.einsum('bn,d->bnd', position_ids, indices)
        pos_embeddings = K.concatenate([
            K.sin(pos_embeddings)[..., None],
            K.cos(pos_embeddings)[..., None]
        ])
        # [...,None]会在最后一维增加一维，把每个值用[]包起来
        # 比如a = K.arange(0, 10) 本来输出的是:[0 1 2 3 4 5 6 7 8 9];a = K.arange(0, 10)[..., None]变成了[[0] [1] [2] [3] [4] [5] [6] [7] [8] [9]]
        # 同K.expand_dim(pos_embeddings, -1)的效果

        # 重新reshape成shape=(btz, seq_len, dim)
        pos_embeddings = K.reshape(
            pos_embeddings, (-1, seq_len, self.output_dim)
        )

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        elif self.merge_mode == 'mul':
            return inputs * pos_embeddings
        else:
            if not self.custom_position_ids:
                pos_embeddings = K.tile(pos_embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul']:
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(SinusoidalPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))