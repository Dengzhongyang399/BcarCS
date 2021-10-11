from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.engine import Input
from keras.layers import Concatenate, Dot, Embedding, Dropout, Lambda, Activation, LSTM, Dense,Reshape,Conv1D,Conv2D,MaxPooling1D,Flatten,GlobalMaxPooling1D,dot,Bidirectional,SimpleRNN,GlobalAveragePooling1D,Reshape,Multiply,GlobalAveragePooling2D
from keras import backend as K
#from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras.utils import plot_model
import pickle
import numpy as np
import logging
from layers.coattention_layer import COAttentionLayer
from layers.joint_self_attention_layer import JointSelfAttentionLayer
from layers.attention_layer import AttentionLayer
from layers.position_encoding_layer import PositionEncodingLayer,PositionEmbedding,SinusoidalPositionEmbedding
from layers.mul_dim_layer import MulDimLayer
from layers.layer_norm_layer import LayerNormLayer
logger = logging.getLogger(__name__)


class JointEmbeddingModel:
    def __init__(self, config):
        self.config = config
        self.model_params = config.get('model_params', dict())
        self.data_params = config.get('data_params',dict())
        self.methname = Input(shape=(self.data_params['methname_len'],), dtype='int32', name='i_methname')
        self.apiseq= Input(shape=(self.data_params['apiseq_len'],),dtype='int32',name='i_apiseq')
        self.sbt= Input(shape=(self.data_params['sbt_len'],),dtype='int32',name='i_sbt')
        self.tokens=Input(shape=(self.data_params['tokens_len'],),dtype='int32',name='i_tokens')
        self.desc_good = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_good')
        self.desc_bad = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_bad')
        
        # initialize a bunch of variables that will be set later
        self._sim_model = None        
        self._training_model = None
        self._shared_model=None
        #self.prediction_model = None
        
        #create a model path to store model info
        if not os.path.exists(self.config['workdir']+'models/'+self.model_params['model_name']+'/'):
            os.makedirs(self.config['workdir']+'models/'+self.model_params['model_name']+'/')
    
    def build(self):
        '''
        1. Build Code Representation Model
        '''
        logger.debug('Building Code Representation Model')
        methname = Input(shape=(self.data_params['methname_len'],), dtype='int32', name='methname')
        apiseq= Input(shape=(self.data_params['apiseq_len'],),dtype='int32',name='apiseq')
        tokens=Input(shape=(self.data_params['tokens_len'],),dtype='int32',name='tokens')
        sbt=Input(shape=(self.data_params['sbt_len'],),dtype='int32',name='sbt')

        ## method name representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_methname']) if self.model_params['init_embed_weights_methname'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_methname'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_methodname_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers in the model must support masking, otherwise an exception will be raised.
                              name='embedding_methname')
        methname_embedding = embedding(methname)
        dropout = Dropout(0.25,name='dropout_methname_embed')
        methname_dropout= dropout(methname_embedding)
        methname_out = AttentionLayer(name = 'methname_attention_layer')(methname_dropout)

        ## API Sequence Representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_api']) if self.model_params['init_embed_weights_api'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_api'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_api_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                                         #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_apiseq')
        apiseq_embedding = embedding(apiseq)
        dropout = Dropout(0.25,name='dropout_apiseq_embed')
        apiseq_dropout= dropout(apiseq_embedding)
        api_out = AttentionLayer(name = 'apiseq_attention_layer')(apiseq_dropout)
        #api_position_embedding=SinusoidalPositionEmbedding(output_dim=self.model_params.get('n_embed_dims'),
                                              #merge_mode="mul",
                                              #custom_position_ids=False,
                                              #name='position_embedding_api')
        #api_out=api_position_embedding(api_out)


        ## Tokens Representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_tokens']) if self.model_params['init_embed_weights_tokens'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_tokens'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_tokens_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_tokens')
        tokens_embedding = embedding(tokens)
        dropout = Dropout(0.25,name='dropout_tokens_embed')
        tokens_dropout= dropout(tokens_embedding)
        tokens_out = AttentionLayer(name = 'tokens_attention_layer')(tokens_dropout)



        ## Sbt Representation ##
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_sbt']) if self.model_params['init_embed_weights_sbt'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_sbt'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_sbt_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_sbt')
        sbt_embedding = embedding(sbt)
        dropout = Dropout(0.25,name='dropout_sbt_embed')
        sbt_dropout= dropout(sbt_embedding)
        sbt_out = AttentionLayer(name = 'sbt_attention_layer')(sbt_dropout)
        #sbt_position_embedding=PositionEmbedding(input_dim=self.data_params['n_sbt_words'],
                                               #output_dim=self.model_params.get('n_embed_dims'),
                                               #merge_mode="mul",
                                               #hierarchical=None,
                                               #embeddings_initializer='zeros',
                                               #custom_position_ids=False,
                                               #name='position_embedding_sbt')
        #sbt_out = sbt_position_embedding(sbt_out)

        #sbt_position_embedding=SinusoidalPositionEmbedding(output_dim=self.model_params.get('n_embed_dims'),
                                              #merge_mode="mul",
                                              #custom_position_ids=False,
                                              #name='position_embedding_sbt')
        #sbt_out = sbt_position_embedding(sbt_out)
        '''
        2. Build Desc Representation Model
        '''
        ## Desc Representation ##
        logger.debug('Building Desc Representation Model')
        desc = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='desc')
        #1.embedding
        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_desc']) if self.model_params['init_embed_weights_desc'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_desc'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_desc_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                                      #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_desc')
        desc_embedding = embedding(desc)
        dropout = Dropout(0.25,name='dropout_desc_embed')
        desc_dropout = dropout(desc_embedding)
        merged_desc = AttentionLayer(name = 'desc_attention_layer')(desc_dropout)

        #AP networks#
        attention = COAttentionLayer(name='coattention_layer') #  (122,60)
        attention_mq_out=attention([methname_out,merged_desc])
        attention_pq_out=attention([api_out,merged_desc])
        attention_tq_out=attention([tokens_out,merged_desc])
        attention_aq_out=attention([sbt_out,merged_desc])
        
        normalOp=Lambda(lambda x: tf.matrix_diag(x),name='normalOp')
        # out_1 colum wise
        gap_cnn=GlobalAveragePooling2D(name='globalaveragepool_cnn')
        #gmp_mq_1=GlobalAveragePooling1D(name='mq_blobalmaxpool_colum')
        #att_mq_1=gmp_mq_1(attention_mq_out)
        activ_mq_1=Activation('softmax',name='mq_AP_active_colum')
        #att_mq_1_next=activ_mq_1(att_mq_1)
        #att_mq_1_next=tf.matrix_diag(att_mq_1_next)
        #att_mq_1_next=normalOp(att_mq_1_next)
        dot_mq_1=Dot(axes=1,normalize=False,name='mq_column_dot')
        #mq_desc_out = dot_mq_1([att_mq_1_next, merged_desc]
        attention_mq_matrix = Lambda(lambda x: K.expand_dims(x, 1))(attention_mq_out)
        mq_conv1 = Conv2D(100,2,data_format='channels_first',padding='same', activation='relu',strides=1,name='mq_conv1')
        #mq_conv2 = Conv2D(100,4,data_format='channels_first',padding='same', activation='relu',strides=1,name='mq_conv2')
        #mq_conv3 = Conv1D(100,4,padding='valid', activation='relu',strides=1,name='mq_conv3')
        mq_desc_conv = mq_conv1(attention_mq_matrix)
        #mq_conv2_out = mq_conv2(attention_mq_matrix)
        #mq_conv3_out = mq_conv3(mq_desc_out)
        #mq_desc_conv = Concatenate(name='mq_desc_merge',axis=1)([mq_conv1_out,mq_conv1_out])
        dense_mq_desc = Dense(30,use_bias=False,name='dense_mq_desc')
        mq_desc_conv= dense_mq_desc(mq_desc_conv)
        mq_desc_conv = gap_cnn(mq_desc_conv)
        mq_desc_att = activ_mq_1(mq_desc_conv)
        mq_desc_out = dot_mq_1([mq_desc_att, merged_desc])
        # out_2 row wise
        attention_trans_layer = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)),name='trans_coattention')
        
        attention_transposed = attention_trans_layer(attention_mq_out)
        #gmp_mq_2=GlobalMaxPooling1D(name='mq_blobalmaxpool_row')
        #att_mq_2=gmp_mq_2(attention_transposed)
        activ_mq_2=Activation('softmax',name='mq_AP_active_row')
        #att_mq_2_next=activ_mq_2(att_mq_2)
        #att_mq_2_next=tf.matrix_diag(att_mq_2_next)
        #att_mq_2_next=normalOp(att_mq_2_next)
        dot_mq_2=Dot(axes=1,normalize=False,name='mq_row_dot')
        #mq_out = dot_mq_2([att_mq_2_next, methname_out])
        attention_mq_matrix = Lambda(lambda x: K.expand_dims(x, 1))(attention_transposed)
        mq_conv4 = Conv2D(100,2,data_format='channels_first',padding='same', activation='relu',strides=1,name='mq_conv4')
        #mq_conv5 = Conv2D(100,4,data_format='channels_first',padding='same', activation='relu',strides=1,name='mq_conv5')
        #mq_conv6 = Conv1D(100,4,padding='valid', activation='relu',strides=1,name='mq_conv6')
        mq_out_conv = mq_conv4(attention_mq_matrix)
        #mq_conv5_out = mq_conv5(attention_mq_matrix)
        #mq_conv6_out = mq_conv6(mq_out)
        #mq_out_conv=Concatenate(name='mq_merge',axis=1)([mq_conv4_out,mq_conv4_out])
        dense_mq = Dense(6,use_bias=False,name='dense_mq')
        mq_out_conv=dense_mq(mq_out_conv)
        mq_out_conv=gap_cnn(mq_out_conv)
        mq_att = activ_mq_2(mq_out_conv)
        mq_out = dot_mq_2([mq_att, methname_out])
        # out_1 colum wise
        #gmp_pq_1=GlobalMaxPooling1D(name='pq_blobalmaxpool_colum')
        #att_pq_1=gmp_pq_1(attention_pq_out)
        activ_pq_1=Activation('softmax',name='pq_AP_active_colum')
        #att_pq_1_next=activ_pq_1(att_pq_1)
        #att_pq_1_next=tf.matrix_diag(att_pq_1_next)
        #att_pq_1_next=normalOp(att_pq_1_next)
        dot_pq_1=Dot(axes=1,normalize=False,name='pq_column_dot')
        #pq_desc_out = dot_pq_1([att_pq_1_next, merged_desc])
        attention_pq_matrix = Lambda(lambda x: K.expand_dims(x, 1))(attention_pq_out)
        pq_conv1 = Conv2D(100,2,data_format='channels_first',padding='same', activation='relu',strides=1,name='pq_conv1')
        #pq_conv2 = Conv2D(100,4,data_format='channels_first',padding='same', activation='relu',strides=1,name='pq_conv2')
        #pq_conv3 = Conv1D(100,4,padding='valid', activation='relu',strides=1,name='pq_conv3')
        pq_desc_conv = pq_conv1(attention_pq_matrix)
        #pq_conv2_out = pq_conv2(attention_pq_matrix)
        #pq_conv3_out = pq_conv3(pq_desc_out)
        #pq_desc_conv=Concatenate(name='pq_desc_merge',axis=1)([pq_conv1_out,pq_conv1_out])
        dense_pq_desc = Dense(30,use_bias=False,name='dense_pq_desc')
        pq_desc_conv=dense_pq_desc(pq_desc_conv)

        pq_desc_conv=gap_cnn(pq_desc_conv)
        pq_desc_att=activ_pq_1(pq_desc_conv)
        pq_desc_out=dot_pq_1([pq_desc_att, merged_desc])
        # out_2 row wise
        attention_transposed = attention_trans_layer(attention_pq_out)
        #gmp_pq_2=GlobalMaxPooling1D(name='pq_blobalmaxpool_row')
        #att_pq_2=gmp_pq_2(attention_transposed)
        activ_pq_2=Activation('softmax',name='pq_AP_active_row')
        #att_pq_2_next=activ_pq_2(att_pq_2)
        #att_pq_2_next=tf.matrix_diag(att_pq_2_next)
        #att_pq_2_next=normalOp(att_pq_2_next)
        dot_pq_2=Dot(axes=1,normalize=False,name='pq_row_dot')
        #pq_out = dot_pq_2([att_pq_2_next, api_out])
        attention_pq_matrix = Lambda(lambda x: K.expand_dims(x, 1))(attention_transposed)
        pq_conv4 = Conv2D(100,2,data_format='channels_first',padding='same', activation='relu',strides=1,name='pq_conv4')
        #pq_conv5 = Conv2D(100,4,data_format='channels_first',padding='same', activation='relu',strides=1,name='pq_conv5')
        #pq_conv6 = Conv1D(100,4,padding='valid', activation='relu',strides=1,name='pq_conv6')
        pq_out_conv = pq_conv4(attention_pq_matrix)
        #pq_conv5_out = pq_conv5(attention_pq_matrix)
        #pq_conv6_out = pq_conv6(pq_out)
        #pq_out_conv=Concatenate(name='pq_merge',axis=1)([pq_conv4_out,pq_conv4_out])
        dense_pq = Dense(30,use_bias=False,name='dense_pq')
        pq_out_conv=dense_pq(pq_out_conv)

        pq_out_conv=gap_cnn(pq_out_conv)
        pq_out_att=activ_pq_2(pq_out_conv)
        pq_out=dot_pq_2([pq_out_att, api_out])
        
        # out_1 colum wise
        #gmp_tq_1=GlobalMaxPooling1D(name='tq_blobalmaxpool_colum')
        #att_tq_1=gmp_tq_1(attention_tq_out)
        activ_tq_1=Activation('softmax',name='tq_AP_active_colum')
        #att_tq_1_next=activ_tq_1(att_tq_1)
        #att_tq_1_next=tf.matrix_diag(att_tq_1_next)
        #att_tq_1_next=normalOp(att_tq_1_next)
        dot_tq_1=Dot(axes=1,normalize=False,name='tq_column_dot')
        #tq_desc_out = dot_tq_1([att_tq_1_next, merged_desc])
        attention_tq_matrix = Lambda(lambda x: K.expand_dims(x, 1))(attention_tq_out)
        tq_conv1 = Conv2D(100,2,data_format='channels_first',padding='same', activation='relu',strides=1,name='tq_conv1')
        #tq_conv2 = Conv2D(100,4,data_format='channels_first',padding='same', activation='relu',strides=1,name='tq_conv2')
        #tq_conv3 = Conv1D(100,4,padding='valid', activation='relu',strides=1,name='tq_conv3')
        tq_desc_conv = tq_conv1(attention_tq_matrix)
        #tq_conv2_out = tq_conv2(attention_tq_matrix)
        #tq_conv3_out = pq_conv3(tq_desc_out)
        #tq_desc_conv=Concatenate(name='tq_desc_merge',axis=1)([tq_conv1_out,tq_conv1_out])
        dense_tq_desc = Dense(30,use_bias=False,name='dense_tq_desc')
        tq_desc_conv=dense_tq_desc(tq_desc_conv)
        tq_desc_conv=gap_cnn(tq_desc_conv)
        tq_desc_att=activ_tq_1(tq_desc_conv)
        tq_desc_out=dot_tq_1([tq_desc_att, merged_desc])
        # out_2 row wise
        attention_transposed = attention_trans_layer(attention_tq_out)
        #gmp_tq_2=GlobalMaxPooling1D(name='tq_blobalmaxpool_row')
        #att_tq_2=gmp_tq_2(attention_transposed)
        activ_tq_2=Activation('softmax',name='tq_AP_active_row')
        #att_tq_2_next=activ_tq_2(att_tq_2)
        #att_tq_2_next=tf.matrix_diag(att_tq_2_next)
        #att_tq_2_next=normalOp(att_tq_2_next)
        dot_tq_2=Dot(axes=1,normalize=False,name='tq_row_dot')
        #tq_out = dot_tq_2([att_tq_2_next, tokens_out])
        attention_tq_matrix = Lambda(lambda x: K.expand_dims(x, 1))(attention_transposed)
        tq_conv4 = Conv2D(100,2,data_format='channels_first',padding='same', activation='relu',strides=1,name='tq_conv4')
        #tq_conv5 = Conv2D(100,4,data_format='channels_first',padding='same', activation='relu',strides=1,name='tq_conv5')
        #tq_conv6 = Conv1D(100,4,padding='valid', activation='relu',strides=1,name='tq_conv6')
        tq_out_conv = tq_conv4(attention_tq_matrix)
        #tq_conv5_out = tq_conv5(attention_tq_matrix)
        #tq_conv6_out = pq_conv6(tq_out)
        #tq_out_conv=Concatenate(name='tq_merge',axis=1)([tq_conv4_out,tq_conv4_out])
        dense_tq = Dense(50,use_bias=False,name='dense_tq')
        tq_out_conv=dense_tq(tq_out_conv)
        tq_out_conv=gap_cnn(tq_out_conv)
        tq_out_att=activ_tq_2(tq_out_conv)
        tq_out=dot_tq_2([tq_out_att, tokens_out])
        
        # out_1 colum wise
        #gmp_aq_1=GlobalMaxPooling1D(name='aq_blobalmaxpool_colum')
        #att_aq_1=gmp_aq_1(attention_aq_out)
        activ_aq_1=Activation('softmax',name='aq_AP_active_colum')
        #att_aq_1_next=activ_aq_1(att_aq_1)
        #att_aq_1_next=tf.matrix_diag(att_aq_1_next)
        #att_aq_1_next=normalOp(att_aq_1_next)
        dot_aq_1=Dot(axes=1,normalize=False,name='aq_column_dot')
        #aq_desc_out = dot_aq_1([att_aq_1_next, merged_desc])
        attention_aq_matrix = Lambda(lambda x: K.expand_dims(x, 1))(attention_aq_out)
        aq_conv1 = Conv2D(100,2,data_format='channels_first',padding='same', activation='relu',strides=1,name='aq_conv1')
        #aq_conv2 = Conv2D(100,4,data_format='channels_first',padding='same', activation='relu',strides=1,name='aq_conv2')
        #aq_conv3 = Conv1D(100,4,padding='valid', activation='relu',strides=1,name='aq_conv3')
        aq_desc_conv = aq_conv1(attention_aq_matrix)
        #aq_conv2_out = aq_conv2(attention_aq_matrix)
        #aq_conv3_out = aq_conv3(aq_desc_out)
        #aq_desc_conv=Concatenate(name='aq_desc_merge',axis=1)([aq_conv1_out,aq_conv1_out])
        dense_aq_desc = Dense(30,use_bias=False,name='dense_aq_desc')
        aq_desc_conv=dense_aq_desc(aq_desc_conv)
        aq_desc_conv=gap_cnn(aq_desc_conv)
        aq_desc_att=activ_aq_1(aq_desc_conv)
        aq_desc_out=dot_aq_1([aq_desc_att, merged_desc])
        
        # out_2 row wise
        attention_transposed = attention_trans_layer(attention_aq_out)
        #gmp_aq_2=GlobalMaxPooling1D(name='aq_blobalmaxpool_row')
        #att_aq_2=gmp_aq_2(attention_transposed)
        activ_aq_2=Activation('softmax',name='aq_AP_active_row')
        #att_aq_2_next=activ_aq_2(att_aq_2)
        #att_aq_2_next=tf.matrix_diag(att_aq_2_next)
        #att_aq_2_next=normalOp(att_aq_2_next)
        dot_aq_2=Dot(axes=1,normalize=False,name='aq_row_dot')
        #aq_out = dot_aq_2([att_aq_2_next, sbt_out])
        attention_aq_matrix = Lambda(lambda x: K.expand_dims(x, 1))(attention_transposed)
        aq_conv4 = Conv2D(100,2,data_format='channels_first',padding='same', activation='relu',strides=1,name='aq_conv4')
        #aq_conv5 = Conv2D(100,4,data_format='channels_first',padding='same', activation='relu',strides=1,name='aq_conv5')
        #aq_conv6 = Conv1D(100,4,padding='valid', activation='relu',strides=1,name='aq_conv6')
        aq_out_conv = aq_conv4(attention_aq_matrix)
        #aq_conv5_out = aq_conv5(attention_aq_matrix)
        #aq_conv6_out = aq_conv6(aq_out)
        #aq_out_conv=Concatenate(name='aq_merge',axis=1)([aq_conv4_out,aq_conv4_out])
        dense_aq = Dense(150,use_bias=False,name='dense_aq')
        aq_out_conv=dense_aq(aq_out_conv)
        aq_out_conv=gap_cnn(aq_out_conv)
        aq_out_att=activ_aq_2(aq_out_conv)
        aq_out=dot_aq_2([aq_out_att, sbt_out])
        #mq_sim_score=Dot(axes=1, normalize=True)([mq_out, mq_desc_out])
        #pq_sim_score=Dot(axes=1, normalize=True)([pq_out, pq_desc_out])
        #tq_sim_score=Dot(axes=1, normalize=True)([tq_out, tq_desc_out])
        #aq_sim_score=Dot(axes=1, normalize=True)([aq_out, aq_desc_out])

        merged_desc_out=Concatenate(name='desc_orig_merge',axis=1)([mq_desc_out,pq_desc_out,tq_desc_out,aq_desc_out])
        merged_code_out=Concatenate(name='code_orig_merge',axis=1)([mq_out,pq_out,tq_out,aq_out])
        reshape_desc=Reshape((4,100))(merged_desc_out)
        reshape_code=Reshape((4,100))(merged_code_out)
      
        att_desc_out=AttentionLayer(name = 'desc_merged_attention_layer')(reshape_desc)
        att_code_out=AttentionLayer(name = 'code_merged_attention_layer')(reshape_code)
        gap=GlobalAveragePooling1D(name='blobalaveragepool')
        mulop=Lambda(lambda x: x*4.0,name='mulop')
        desc_out=mulop(gap(att_desc_out))
        code_out=mulop(gap(att_code_out))
        
        
        #sumop=Lambda(lambda x: x[0]+x[1]+x[2]+x[3], name='sumop')
        #mulop=Lambda(lambda x: x[0]*x[1], name='mulop')
        #weightop=Lambda(lambda x: x[0]*0.4+x[1]*0.2+x[2]*0.2+x[3]*0.2, name='mulop')
        #desc_out=sumop([mulop([mq_desc_out,mq_sim_score]),mulop([pq_desc_out,pq_sim_score]),mulop([tq_desc_out,tq_sim_score]),mulop([aq_desc_out,aq_sim_score])])
        #desc_out=sumop([mq_desc_out,pq_desc_out,tq_desc_out,aq_desc_out])
        #desc_out=weightop([mq_desc_out,pq_desc_out,tq_desc_out,aq_desc_out])
        #code_out=sumop([mq_out,pq_out,tq_out,aq_out])
        #code_out=sumop([mulop([mq_out,mq_sim_score]),mulop([pq_out,pq_sim_score]),mulop([tq_out,tq_sim_score]),mulop([aq_out,aq_sim_score])])
        #code_out=weightop([mq_out,pq_out,tq_out,aq_out])
        
        #dense_desc = Dense(400,use_bias=False,name='dense_desc')
        #dense_code = Dense(400,use_bias=False,name='dense_code')
        #gap=GlobalAveragePooling1D(name='blobalaveragepool')
        #desc_out=Concatenate(name='desc_orig_merge',axis=1)([mq_desc_out,pq_desc_out,tq_desc_out,aq_desc_out])
        #code_out=Concatenate(name='code_orig_merge',axis=1)([mq_out,pq_out,tq_out,aq_out])
        #desc_out=dense_desc(desc_out)
        #code_out=dense_code(code_out)
        """
        3: calculate the cosine similarity between code and desc
        """     
        logger.debug('Building similarity model')
        cos_sim=Dot(axes=1, normalize=True, name='cos_sim')([code_out, desc_out])
        
        sim_model = Model(inputs=[methname,apiseq,tokens,sbt,desc], outputs=[cos_sim],name='sim_model')   
        self._sim_model=sim_model  #for model evaluation  
        print ("\nsummary of similarity model")
        self._sim_model.summary() 
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_sim_model.png'
        #plot_model(self._sim_model, show_shapes=True, to_file=fname)
        
        
        '''
        4:Build training model
        '''
        good_sim = sim_model([self.methname,self.apiseq,self.tokens,self.sbt, self.desc_good])# similarity of good output
        bad_sim = sim_model([self.methname,self.apiseq,self.tokens,self.sbt, self.desc_bad])#similarity of bad output
        loss = Lambda(lambda x: K.maximum(1e-6, self.model_params['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0], name='loss')([good_sim, bad_sim])

        logger.debug('Building training model')
        self._training_model=Model(inputs=[self.methname,self.apiseq,self.tokens,self.sbt, self.desc_good,self.desc_bad],
                                   outputs=[loss],name='training_model')
        print ('\nsummary of training model')
        self._training_model.summary()      
        fname=self.config['workdir']+'models/'+self.model_params['model_name']+'/_training_model.png'
        #plot_model(self._training_model, show_shapes=True, to_file=fname)     

    def compile(self, optimizer, **kwargs):
        logger.info('compiling models')
        self._training_model.compile(loss=lambda y_true, y_pred: y_pred+y_true-y_true, optimizer=optimizer, **kwargs)
        #+y_true-y_true is for avoiding an unused input warning, it can be simply +y_true since y_true is always 0 in the training set.
        self._sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self._training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1],dtype=np.float32)
        return self._training_model.fit(x, y, **kwargs)


    def predict(self, x, **kwargs):
        return self._sim_model.predict(x, **kwargs)

    def save(self, sim_model_file, **kwargs):
        assert self._sim_model is not None, 'Must compile the model before saving weights'
        self._sim_model.save_weights(sim_model_file, **kwargs)


    def load(self, sim_model_file,  **kwargs):
        assert self._sim_model is not None, 'Must compile the model loading weights'
        self._sim_model.load_weights(sim_model_file, **kwargs)

 
 
 
 