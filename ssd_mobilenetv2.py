""" 
Try to use MobileNet v2 as feature extractor of SSD

A Keras implementaiton

# Reference:
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   https://arxiv.org/abs/1801.04381
"""

import keras.backend as K
from keras.layers import Activation
#from keras.layers import AtrousConv2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import merge
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers import Dropout
from keras.models import Model
from keras.layers import add
from keras.layers import Activation
from keras.applications.mobilenet import relu6, DepthwiseConv2D
#from keras.utils import plot_model

#from ssd_layers import Normalize
#from ssd_layers import PriorBox


def MobileNetSSD(input_shape, num_classes=21):
    """MobileSSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    net = {}

    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    net['input'] = input_tensor

    ### Conv2D 1 strides=2 ###
    net['conv1'] = Conv2D(32, (3,3), strides=(2,2), 
                            padding='same', name='conv1')(net['input'])
    net['conv1_bn'] = BatchNormalization(axis=bn_axis, name='conv1_bn')(net['conv1'])
    net['conv1_relu'] = Activation('relu', name='conv1_relu')(net['conv1_bn'])

    
    ### Invertable Residual Block 1_1 ###

    # Bottleneck 1: t=1, c=16, s=1
    t, c, s = 1, 16, 1

    tchannel = int(net['conv1_relu'].shape[bn_axis]*t)
    net['conv_pw_in_1'] = Conv2D(int(net['conv1_relu'].shape[bn_axis]*t), (1,1), strides=(1,1), 
                                     padding='same', name='conv_pw_in_1')(net['conv1_relu'])
    net['conv_pw_in_1_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_1_bn')(net['conv_pw_in_1'])
    net['conv_pw_in_1_relu'] = Activation('relu', name='conv_pw_in_1_relu')(net['conv_pw_in_1_bn'])

    
    net['conv_dw_1'] = DepthwiseConv2D((3,3), strides=s, padding='same', name='conv_dw_1')(net['conv_pw_in_1_relu'])
    net['conv_dw_1_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_1_bn')(net['conv_dw_1'])
    net['conv_dw_1_relu'] = Activation('relu', name='conv_dw_1_relu')(net['conv_dw_1_bn'])


    net['conv_pw_out_1'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_1')(net['conv_dw_1_relu'])
    net['conv_pw_out_1_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_1_bn')(net['conv_pw_out_1'])


    ### Inverted Residual Block 2_2 ###

    # Bottleneck 2 : t=6, c=24, s=2
    t, c, s = 6, 24, 2
    
    tchannel = int(net['conv_pw_out_1_bn'].shape[bn_axis]*t)
    net['conv_pw_in_2'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_2')(net['conv_pw_out_1_bn'])
    net['conv_pw_in_2_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_2_bn')(net['conv_pw_in_2'])
    net['conv_pw_in_2_relu'] = Activation('relu', name='conv_pw_in_2_relu')(net['conv_pw_in_2_bn'])

    
    net['conv_dw_2'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_2')(net['conv_pw_in_2_relu'])
    net['conv_dw_2_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_2_bn')(net['conv_dw_2'])
    net['conv_dw_2_relu'] = Activation('relu', name='conv_dw_2_relu')(net['conv_dw_2_bn'])


    net['conv_pw_out_2'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_2')(net['conv_dw_2_relu'])
    net['conv_pw_out_2_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_2_bn')(net['conv_pw_out_2'])


    # Bottleneck 3 : t=6, c=24, s=1, residual = true
    s = 1
    
    tchannel = int(net['conv_pw_out_2_bn'].shape[bn_axis]*t)
    net['conv_pw_in_3'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_3')(net['conv_pw_out_2_bn'])
    net['conv_pw_in_3_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_3_bn')(net['conv_pw_in_3'])
    net['conv_pw_in_3_relu'] = Activation('relu', name='conv_pw_in_3_relu')(net['conv_pw_in_3_bn'])

    
    net['conv_dw_3'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_3')(net['conv_pw_in_3_relu'])
    net['conv_dw_3_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_3_bn')(net['conv_dw_3'])
    net['conv_dw_3_relu'] = Activation('relu', name='conv_dw_3_relu')(net['conv_dw_3_bn'])


    net['conv_pw_out_3'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_3')(net['conv_dw_3_relu'])
    net['conv_pw_out_3_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_3_bn')(net['conv_pw_out_3'])
    net['add_1'] = add([net['conv_pw_out_3_bn'], net['conv_pw_out_2_bn']])

    
    ### Inverted Residual Block 3_3 ###

    # Bottleneck 4 : t=6, c=32, s=2
    t, c, s = 6, 32, 2

    tchannel = int(net['add_1'].shape[bn_axis]*t)
    net['conv_pw_in_4'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_4')(net['add_1'])
    net['conv_pw_in_4_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_4_bn')(net['conv_pw_in_4'])
    net['conv_pw_in_4_relu'] = Activation('relu', name='conv_pw_in_4_relu')(net['conv_pw_in_4_bn'])

    
    net['conv_dw_4'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_4')(net['conv_pw_in_4_relu'])
    net['conv_dw_4_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_4_bn')(net['conv_dw_4'])
    net['conv_dw_4_relu'] = Activation('relu', name='conv_dw_4_relu')(net['conv_dw_4_bn'])


    net['conv_pw_out_4'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_4')(net['conv_dw_4_relu'])
    net['conv_pw_out_4_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_4_bn')(net['conv_pw_out_4'])

    
    # Bottleneck 5 : t=6, c=32, s=1, residual = true
    s = 1

    tchannel = int(net['conv_pw_out_4_bn'].shape[bn_axis]*t)
    net['conv_pw_in_5'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_5')(net['conv_pw_out_4_bn'])
    net['conv_pw_in_5_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_5_bn')(net['conv_pw_in_5'])
    net['conv_pw_in_5_relu'] = Activation('relu', name='conv_pw_in_5_relu')(net['conv_pw_in_5_bn'])

    
    net['conv_dw_5'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_5')(net['conv_pw_in_5_relu'])
    net['conv_dw_5_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_5_bn')(net['conv_dw_5'])
    net['conv_dw_5_relu'] = Activation('relu', name='conv_dw_5_relu')(net['conv_dw_5_bn'])


    net['conv_pw_out_5'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_5')(net['conv_dw_5_relu'])
    net['conv_pw_out_5_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_5_bn')(net['conv_pw_out_5'])
    net['add_2'] = add([net['conv_pw_out_5_bn'], net['conv_pw_out_4_bn']])

    
    
    # Bottleneck 6 : t=6, c=32, s=1， residual = true

    tchannel = int(net['add_2'].shape[bn_axis]*t)
    net['conv_pw_in_6'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_6')(net['add_2'])
    net['conv_pw_in_6_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_6_bn')(net['conv_pw_in_6'])
    net['conv_pw_in_6_relu'] = Activation('relu', name='conv_pw_in_6_relu')(net['conv_pw_in_6_bn'])

    
    net['conv_dw_6'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_6')(net['conv_pw_in_6_relu'])
    net['conv_dw_6_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_6_bn')(net['conv_dw_6'])
    net['conv_dw_6_relu'] = Activation('relu', name='conv_dw_6_relu')(net['conv_dw_6_bn'])


    net['conv_pw_out_6'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_6')(net['conv_dw_6_relu'])
    net['conv_pw_out_6_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_6_bn')(net['conv_pw_out_6'])
    net['add_3'] = add([net['add_2'], net['conv_pw_out_5_bn']])
    
    
    ### Inverted Residual Block 4_4 ###

    # Bottleneck 7 : t=6, c=64, s=2, residual = false
    t, c, s = 6, 64, 2

    tchannel = int(net['add_3'].shape[bn_axis]*t)
    net['conv_pw_in_7'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_7')(net['add_3'])
    net['conv_pw_in_7_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_7_bn')(net['conv_pw_in_7'])
    net['conv_pw_in_7_relu'] = Activation('relu', name='conv_pw_in_7_relu')(net['conv_pw_in_7_bn'])

    
    net['conv_dw_7'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_7')(net['conv_pw_in_7_relu'])
    net['conv_dw_7_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_7_bn')(net['conv_dw_7'])
    net['conv_dw_7_relu'] = Activation('relu', name='conv_dw_7_relu')(net['conv_dw_7_bn'])


    net['conv_pw_out_7'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_7')(net['conv_dw_7_relu'])
    net['conv_pw_out_7_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_7_bn')(net['conv_pw_out_7'])


    # Bottleneck 8 : t=6, c=64, s=1, residual = true
    s = 1

    tchannel = int(net['conv_pw_out_7_bn'].shape[bn_axis]*t)
    net['conv_pw_in_8'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_8')(net['conv_pw_out_7_bn'])
    net['conv_pw_in_8_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_8_bn')(net['conv_pw_in_8'])
    net['conv_pw_in_8_relu'] = Activation('relu', name='conv_pw_in_8_relu')(net['conv_pw_in_8_bn'])

    
    net['conv_dw_8'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_8')(net['conv_pw_in_8_relu'])
    net['conv_dw_8_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_8_bn')(net['conv_dw_8'])
    net['conv_dw_8_relu'] = Activation('relu', name='conv_dw_8_relu')(net['conv_dw_8_bn'])


    net['conv_pw_out_8'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_8')(net['conv_dw_8_relu'])
    net['conv_pw_out_8_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_8_bn')(net['conv_pw_out_8'])
    net['add_4'] = add([net['conv_pw_out_7_bn'],  net['conv_pw_out_8_bn']])


    # Bottleneck 9 : t=6, c=64, s=1, residual = true

    tchannel = int(net['add_4'].shape[bn_axis]*t)
    net['conv_pw_in_9'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_9')(net['add_4'])
    net['conv_pw_in_9_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_9_bn')(net['conv_pw_in_9'])
    net['conv_pw_in_9_relu'] = Activation('relu', name='conv_pw_in_9_relu')(net['conv_pw_in_9_bn'])

    
    net['conv_dw_9'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_9')(net['conv_pw_in_9_relu'])
    net['conv_dw_9_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_9_bn')(net['conv_dw_9'])
    net['conv_dw_9_relu'] = Activation('relu', name='conv_dw_9_relu')(net['conv_dw_9_bn'])


    net['conv_pw_out_9'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_9')(net['conv_dw_9_relu'])
    net['conv_pw_out_9_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_9_bn')(net['conv_pw_out_9'])
    net['add_5'] = add([net['add_4'],  net['conv_pw_out_9_bn']])



    # Bottleneck 10 : t=6, c=64, s=1, residual = true

    tchannel = int(net['add_5'].shape[bn_axis]*t)
    net['conv_pw_in_10'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_10')(net['add_5'])
    net['conv_pw_in_10_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_10_bn')(net['conv_pw_in_10'])
    net['conv_pw_in_10_relu'] = Activation('relu', name='conv_pw_in_10_relu')(net['conv_pw_in_10_bn'])

    
    net['conv_dw_10'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_10')(net['conv_pw_in_10_relu'])
    net['conv_dw_10_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_10_bn')(net['conv_dw_10'])
    net['conv_dw_10_relu'] = Activation('relu', name='conv_dw_10_relu')(net['conv_dw_10_bn'])


    net['conv_pw_out_10'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_10')(net['conv_dw_10_relu'])
    net['conv_pw_out_10_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_10_bn')(net['conv_pw_out_10'])
    net['add_6'] = add([net['add_5'],  net['conv_pw_out_10_bn']])


    ### Inverted Reisual Block 5_3 ###

    # Bottleneck 11 : t=6, c=96, s=1, residual = false
    t, c, s = 6, 96, 1

    tchannel = int(net['add_6'].shape[bn_axis]*t)
    net['conv_pw_in_11'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_11')(net['add_6'])
    net['conv_pw_in_11_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_11_bn')(net['conv_pw_in_11'])
    net['conv_pw_in_11_relu'] = Activation('relu', name='conv_pw_in_11_relu')(net['conv_pw_in_11_bn'])

    
    net['conv_dw_11'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_11')(net['conv_pw_in_11_relu'])
    net['conv_dw_11_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_11_bn')(net['conv_dw_11'])
    net['conv_dw_11_relu'] = Activation('relu', name='conv_dw_11_relu')(net['conv_dw_11_bn'])


    net['conv_pw_out_11'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_11')(net['conv_dw_11_relu'])
    net['conv_pw_out_11_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_11_bn')(net['conv_pw_out_11'])


    # Bottleneck 12 : t=6, c=96, s=1, residual = true
    s = 1

    tchannel = int(net['conv_pw_out_11_bn'].shape[bn_axis]*t)
    net['conv_pw_in_12'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_12')(net['conv_pw_out_11_bn'])
    net['conv_pw_in_12_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_12_bn')(net['conv_pw_in_12'])
    net['conv_pw_in_12_relu'] = Activation('relu', name='conv_pw_in_12_relu')(net['conv_pw_in_12_bn'])

    
    net['conv_dw_12'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_12')(net['conv_pw_in_12_relu'])
    net['conv_dw_12_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_12_bn')(net['conv_dw_12'])
    net['conv_dw_12_relu'] = Activation('relu', name='conv_dw_12_relu')(net['conv_dw_12_bn'])


    net['conv_pw_out_12'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_12')(net['conv_dw_12_relu'])
    net['conv_pw_out_12_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_12_bn')(net['conv_pw_out_12'])
    net['add_7'] = add([net['conv_pw_out_11_bn'],  net['conv_pw_out_12_bn']])

    # Bottleneck 13 : t=6, c=96, s=1, residual = true

    tchannel = int(net['add_7'].shape[bn_axis]*t)
    net['conv_pw_in_13'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_13')(net['add_7'])
    net['conv_pw_in_13_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_13_bn')(net['conv_pw_in_13'])
    net['conv_pw_in_13_relu'] = Activation('relu', name='conv_pw_in_13_relu')(net['conv_pw_in_13_bn'])

    
    net['conv_dw_13'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_13')(net['conv_pw_in_13_relu'])
    net['conv_dw_13_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_13_bn')(net['conv_dw_13'])
    net['conv_dw_13_relu'] = Activation('relu', name='conv_dw_13_relu')(net['conv_dw_13_bn'])


    net['conv_pw_out_13'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_13')(net['conv_dw_13_relu'])
    net['conv_pw_out_13_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_13_bn')(net['conv_pw_out_13'])
    net['add_8'] = add([net['add_7'],  net['conv_pw_out_13_bn']])


    ### Inverted Reisual Block 6_3 ###

    # Bottleneck 14 : t=6, c=160, s=2, residual = false
    t, c, s = 6, 160, 2

    tchannel = int(net['add_8'].shape[bn_axis]*t)
    net['conv_pw_in_14'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_14')(net['add_8'])
    net['conv_pw_in_14_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_14_bn')(net['conv_pw_in_14'])
    net['conv_pw_in_14_relu'] = Activation('relu', name='conv_pw_in_14_relu')(net['conv_pw_in_14_bn'])

    
    net['conv_dw_14'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_14')(net['conv_pw_in_14_relu'])
    net['conv_dw_14_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_14_bn')(net['conv_dw_14'])
    net['conv_dw_14_relu'] = Activation('relu', name='conv_dw_14_relu')(net['conv_dw_14_bn'])


    net['conv_pw_out_14'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_14')(net['conv_dw_14_relu'])
    net['conv_pw_out_14_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_14_bn')(net['conv_pw_out_14'])

    # Bottleneck 15 : t=6, c=160, s=1, residual = true
    s = 1

    tchannel = int(net['conv_pw_out_14_bn'].shape[bn_axis]*t)
    net['conv_pw_in_15'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_15')(net['conv_pw_out_14_bn'])
    net['conv_pw_in_15_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_15_bn')(net['conv_pw_in_15'])
    net['conv_pw_in_15_relu'] = Activation('relu', name='conv_pw_in_15_relu')(net['conv_pw_in_15_bn'])

    
    net['conv_dw_15'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_15')(net['conv_pw_in_15_relu'])
    net['conv_dw_15_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_15_bn')(net['conv_dw_15'])
    net['conv_dw_15_relu'] = Activation('relu', name='conv_dw_15_relu')(net['conv_dw_15_bn'])


    net['conv_pw_out_15'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_15')(net['conv_dw_15_relu'])
    net['conv_pw_out_15_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_15_bn')(net['conv_pw_out_15'])
    net['add_9'] = add([net['conv_pw_out_14_bn'],  net['conv_pw_out_15_bn']])

    
    # Bottleneck 16 : t=6, c=160, s=1, residual = true

    tchannel = int(net['add_9'].shape[bn_axis]*t)
    net['conv_pw_in_16'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_16')(net['add_9'])
    net['conv_pw_in_16_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_16_bn')(net['conv_pw_in_16'])
    net['conv_pw_in_16_relu'] = Activation('relu', name='conv_pw_in_16_relu')(net['conv_pw_in_16_bn'])

    
    net['conv_dw_16'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_16')(net['conv_pw_in_16_relu'])
    net['conv_dw_16_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_16_bn')(net['conv_dw_16'])
    net['conv_dw_16_relu'] = Activation('relu', name='conv_dw_16_relu')(net['conv_dw_16_bn'])


    net['conv_pw_out_16'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_16')(net['conv_dw_16_relu'])
    net['conv_pw_out_16_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_16_bn')(net['conv_pw_out_16'])
    net['add_10'] = add([net['add_9'],  net['conv_pw_out_16_bn']])



    ### Inverted Reisual Block 7_1 ###

    # Bottleneck 17 : t=6, c=320, s=1, residual = false
    t, c, s = 6, 320, 2

    tchannel = int(net['add_10'].shape[bn_axis]*t)
    net['conv_pw_in_17'] = Conv2D(tchannel, (1,1), strides=(1,1), 
                                padding='same', name='conv_pw_in_17')(net['add_10'])
    net['conv_pw_in_17_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_17_bn')(net['conv_pw_in_17'])
    net['conv_pw_in_17_relu'] = Activation('relu', name='conv_pw_in_17_relu')(net['conv_pw_in_17_bn'])

    
    net['conv_dw_17'] = DepthwiseConv2D((3,3), strides=(s,s), padding='same', name='conv_dw_17')(net['conv_pw_in_17_relu'])
    net['conv_dw_17_bn'] = BatchNormalization(axis=bn_axis, name='conv_dw_17_bn')(net['conv_dw_17'])
    net['conv_dw_17_relu'] = Activation('relu', name='conv_dw_17_relu')(net['conv_dw_17_bn'])


    net['conv_pw_out_17'] = Conv2D(c, (1,1), strides=(1,1), 
                                    padding='same', name='conv_pw_out_17')(net['conv_dw_17_relu'])
    net['conv_pw_out_17_bn'] = BatchNormalization(axis=bn_axis, name='conv_pw_out_17_bn')(net['conv_pw_out_17'])


    ### Conv2D 2 strides=1 ###
    net['conv2'] = Conv2D(1280, (3,3), strides=(2,2), 
                            padding='same', name='conv2')(net['conv_pw_out_17_bn'])
    net['conv2_bn'] = BatchNormalization(axis=bn_axis, name='conv2_bn')(net['conv2'])
    net['conv2_relu'] = Activation('relu', name='conv2_relu')(net['conv2_bn'])

    ### AvgPool 7*7 ###
    net['avgpool'] = GlobalAveragePooling2D(name='avgpool')(net['conv2_relu'])
    net['avgpool_reshape'] = Reshape((1, 1, 1280), name='avgpool_reshape')(net['avgpool'])
    net['dropout'] = Dropout(0.5, name='dropout')(net['avgpool_reshape'])

    ### Conv for prediction ###
    net['conv_preds'] = Conv2D(1000, (1,1), padding='same', name='conv_preds')(net['dropout'])


    model = Model(net['input'], net['conv_preds'])

    return model


if __name__ == '__main__':
    model = MobileNetSSD((224, 224, 3))
    model.summary()
    # print(model.get_weights())