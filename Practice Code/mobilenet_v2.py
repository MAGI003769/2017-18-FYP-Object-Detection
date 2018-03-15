""" 
MobileNet v2, a Keras implementaiton

# Reference:
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   https://arxiv.org/abs/1801.04381
"""
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, ZeroPadding2D, Activation
from keras.layers import GlobalAveragePolling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense, Flatten, merger, Reshape
from keras.layers import Dropout
from keras.layers import add
from keras.applications.mobilenet import relu6, DepthwiseConv2D

def conv_block(inputs, channels, kernel_size, strides):
    """
    Convolution Block in MobileNet V2
    The conv block in this architecture has BN and relu6 activation.

    # Arguments
    -----------
        inputs: Tensors feed to conv block (w,h,c) / (c, w, h)
        channels: Integer specify the number of output channel of conv layer
        kernel_size: An integer or tuple to specify size of kernel
        strides: An intrger or tuple
    
    # Returns
    ---------
        Output tensor of conv block
    """

    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
    else:
        channel_axis = 1

    x = Conv2D(channels, kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization(axis = channel_axis)
    x = Activation(relu6)(x)

    return x

def bottleneck(inputs, channels, dwise_kernel, t, dwise_stri):
    """
    Bottleneck
    This function defines a basic bottleneck structure. 
    
    (1,1)conv_relu6 --> (3,3)dwise_conv_relu6 --> (1,1)conv_linear

    # Arguments
    -----------
        inputs: Tensor feed to conv layer
        channels: Integer, the number of output channels of bottleneck block
        dwise_kernel: Integer or tuple specify size of window of depthwise conv
        t: Integer, expension factor
        dwise_stri:  Integer or tuple, specifying strides of depthwise conv 
     
    # Returns
    ---------
        Output tensor
    """
    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
    else:
        channel_axis = 1
    
    t_channel = K.int_shape(inputs)[channel_axis] * t

    '''conv (1,1) relu6'''
    x = conv_block(inputs, t_channel, kernel_size=(1,1), strides=(1,1))

    '''depthwise_conv (3,3) relu6'''
    x = DepthwiseConv2D(dwise_kernel, strides=(dwise_stri,dwise_stri), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    '''conv (1,1) linear'''
    x = Conv2D(channels, kernel_size=(1,1), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    '''residual short connection'''
    if dwise_stri == 1:
        x = add([x, inputs])

    return x

def inverted_residual_block(inputs, channels, dwise_kernel, t, dwise_stri, n):
    """
    Inverted residual block
    This function defines an inverted residual block. 

    # Arguments
    -----------
        inputs: Tensor feed to conv layer
        channels: Integer, the number of output channels of inverted residual block
        dwise_kernel: Integer or tuple specify size of window of depthwise conv
        t: Integer, expension factor
        dwise_stri:  Integer or tuple, specifying strides of depthwise conv 
        n: repeatation times of operator
     
    # Returns
    ---------
        Output tensor
    """
    
    x = bottleneck(inputs, channels, dwise_kernel, t, dwise_stri)

    while n>1:
        x = bottleneck(x, channels, dwise_kernel, t, dwise_stri)
        n = n - 1

    return x


def MobileNet_v2(input_shape, num_classes=21):
    """
    MobileNet V2 architecture

    # Arguments
    -----------
        input_shape: Shape of the input image, expected to be (300, 300, 3)
        num_classes: NUmber of classes including background.

    # Returns
    ---------
        MobileNet V2 model
    """

    inputs = Input(shape=input_shape)
    
    x = conv_block(inputs, 32, kernel_size=(3, 3), strides=(2,2))

    x = inverted_residual_block(x, 16, dwise_kernel=(3,3), t=1, dwise_stri=1, n=1)
    x = inverted_residual_block(x, 24, dwise_kernel=(3,3), t=6, dwise_stri=2, n=2)
    x = inverted_residual_block(x, 32, dwise_kernel=(3,3), t=6, dwise_stri=2, n=3)
    x = inverted_residual_block(x, 64, dwise_kernel=(3,3), t=6, dwise_stri=2, n=4)
    x = inverted_residual_block(x, 96, dwise_kernel=(3,3), t=6, dwise_stri=1, n=3)
    x = inverted_residual_block(x, 160, dwise_kernel=(3,3), t=6, dwise_stri=2, n=3)
    x = inverted_residual_block(x, 320, dwise_kernel=(3,3), t=6, dwise_stri=1, n=3)

    x = conv_block(x, 1280, kernel_size=(1,1), strides=1)
    x = GlobalAveragePolling2D()(x)
    x = Conv2D(num_classes, kernel_size=(1,1), padding='same')(x)

    return x

    
    



