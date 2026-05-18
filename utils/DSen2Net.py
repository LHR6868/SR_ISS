from __future__ import division
from keras.layers import  GlobalAveragePooling2D, Conv1D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Input, Activation, Lambda
from keras.models import Model
from keras import backend as K
from keras import regularizers
import tensorflow as tf
from keras.activations import sigmoid
import math

K.set_image_data_format('channels_first')

channel_axis = 1 if K.image_data_format() == "channels_first" else 3

#  kernel_initializer='he_normal','glorot_normal'
def regularized_padded_conv(*args, **kwargs):
    return Conv2D(*args, **kwargs, padding='same', use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(5e-4))


def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = int(input_feature.shape[channel_axis])
	
	shared_layer_one = Dense(units=int(channel//ratio),
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(units=int(channel),
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		# cbam_feature = Permute((2,3,1))(input_feature)
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(cbam_feature)  
	#assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(cbam_feature)
	#assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(3)([avg_pool, max_pool])  
	#assert concat._keras_shape[-1] == 2
	concat = Permute((3, 1, 2))(concat)
	cbam_feature = Conv2D(1, (1, 1))(concat)
		
	return multiply([input_feature, cbam_feature])

def eca_block(input_feature, b=1, gamma=2, name=""):
	channel = input_feature.shape[1]
	kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
	kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
	
	avg_pool = GlobalAveragePooling2D()(input_feature)
	
	x = Reshape((-1,1))(avg_pool)
	x = Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False,)(x)
	x = Activation('sigmoid')(x)
	x = Reshape((1, 1, -1))(x)
	x = Permute((3, 1, 2))(x)
	output = multiply([input_feature,x])
	return output

def resBlock(x, channels, kernel_size=[3, 3], scale=0.1):
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(channels, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    # tmp = eca_block(tmp)
    tmp = cbam_block(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)
    return Add()([x, tmp])


def s2model(input_shape, num_layers=32, feature_size=256):

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=1)([input10, input20, input60])
    else:
        x = Concatenate(axis=1)([input10, input20])

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same')(x)

    for i in range(num_layers):
        x = resBlock(x, feature_size)

    # One more convolution, and then we add the output of our first conv layer
    x = Conv2D(input_shape[-1][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    # x = Dropout(0.3)(x)
    if len(input_shape) == 3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)
    return model

