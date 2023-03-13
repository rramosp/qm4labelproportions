import tensorflow as tf

from ..models import classicregr
from ..models import classicsegm


downsampl01 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(
                    conv_layers=[
                        dict(kernel_size=8, filters=96, activation='relu', padding='valid', strides=4, dropout=0.1)
                    ]
                )
            )

downsampl02 = dict(
                model_class=classicsegm.Custom_DownsamplingSegmentation,
                model_init_args=dict(
                    conv_layers=[
                        dict(kernel_size=8, filters=96, activation='relu', padding='valid', strides=4, dropout=0.1)
                    ],
                    use_alexnet_weights = True
                )
            )

smvgg16 = dict(
                model_class = classicsegm.SM_UnetSegmentation,
                model_init_args = dict(sm_keywords = dict(backbone_name = 'vgg16'))
            )

vgg16regr = dict (
                model_class = classicregr.KerasBackbone_ConvolutionsRegression,
                model_init_args = dict(number_of_classes = 5, backbone = tf.keras.applications.VGG16)
            )


convreg01 = {
           'model_class': classicregr.Custom_ConvolutionsRegression,
           'model_init_args': 
                dict(conv_layers = [
                                    dict(kernel_size=11, filters=16, activation='elu', padding='same', strides=2, dropout=0.2, maxpool=2),
                                    dict(kernel_size=11, filters=16, activation='elu', padding='same', strides=2, dropout=0.2, maxpool=2),
                                    dict(kernel_size=11, filters=16, activation='elu', padding='same', strides=4, dropout=0.2, maxpool=2),
                                    ], 
                     dense_layers = [
                                    dict(units=16, activation='relu', dropout=0.2),
                                    dict(units=16, activation='relu', dropout=0.2)
                                    ]
                    )
          }