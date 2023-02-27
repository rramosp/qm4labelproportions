from ..models_multiclass import *
from .classic import *
from collections import OrderedDict

nlbe_class_weights = OrderedDict()
nlbe_class_weights[0] = 1
nlbe_class_weights[1] = 1
nlbe_class_weights[(2,3,6)] = 1
nlbe_class_weights[4] = 1
nlbe_class_weights[5] = 1

# -------------------------------------
# UNet segmentation models
# -------------------------------------
cunet_01 = {'model_class': CustomUnetSegmentation,
          'model_init_args':dict(nlayers=1, dropout=0.1),
          'learning_rate': 0.001}

cunet_02 = {'model_class': CustomUnetSegmentation,
          'model_init_args':dict(nlayers=2, dropout=0.1),
          'learning_rate': 0.001}

sunet_01 = {'model_class': SMUnetSegmentation,
          'model_init_args': dict(backbone_name = 'vgg16', encoder_weights='imagenet'),
          'learning_rate': 0.001}
          

# -------------------------------------
# QM patch segmentation models
# -------------------------------------
qmpatchsegm_01 = {'model_class': QMPatchSegmentation,
          'model_init_args': dict(
                        input_shape=(96, 96, 3),
                        patch_size=6,
                        pred_strides=2,
                        n_comp=64, 
                        sigma_ini=0.5,
                        deep=False),                            
          'learning_rate': 0.001
         }

qmpatchsegm_02 = {'model_class': QMPatchSegmentation,
          'model_init_args': dict(
                        input_shape=(96, 96, 3),
                        patch_size=6,
                        pred_strides=2,
                        n_comp=64, 
                        sigma_ini=0.5,
                        deep=True                            
                       ),
          'learning_rate': 0.001
         }


# -------------------------------------
# QM regression models
# -------------------------------------
qmreg_01 = {'model_class': QMRegression,
          'model_init_args': dict(backbone=tf.keras.applications.VGG16,
                                 n_comp=64, sigma_ini=0.1)
         }


# -------------------------------------
# patch segmentation models
# -------------------------------------
patchsegm_00 = { 'model_class':  PatchClassifierSegmentation,
           'model_init_args': dict(input_shape=(96,96,3), 
                                   patch_size=6, pred_strides=2, num_units=5, dropout_rate=0.1, activation='relu'),
           'loss': 'kldiv',
           'learning_rate': 0.001}


# -------------------------------------
# Downsampling Segmentation models
# -------------------------------------

dconvsegm_00 = {
            'model_class': Custom_DownsamplingSegmentation,
            'model_init_args': 
                    dict(conv_layers=[
                                    dict(kernel_size=6, filters=5, activation='relu', padding='valid', strides=2, dropout=0.1)
                                    ], 
                                    use_alexnet_weights=False
                                    ),
            'loss': 'kldiv',
            'learning_rate': 0.001
          }

dconvsegm_01 = {
            'model_class': Custom_DownsamplingSegmentation,
            'model_init_args': 
                    dict(conv_layers=[
                                    dict(kernel_size=4, filters=16, activation='relu', padding='valid', strides=4, dropout=0.1)
                                    ], 
                                    use_alexnet_weights=False
                                    ),
            'loss': 'kldiv',
            'learning_rate': 0.001
          }

dconvsegm_02 = {
            'model_class': Custom_DownsamplingSegmentation,
            'model_init_args': 
                    dict(conv_layers=[
                                    dict(kernel_size=4, filters=16, activation='relu', padding='valid', strides=4, dropout=0.1)
                                    ], 
                                    use_alexnet_weights=False
                                    ),
            'loss': 'kldiv',
            'learning_rate': 0.001
          }

dconvsegm_03 = {
            'model_class': Custom_DownsamplingSegmentation,
            'model_init_args': 
                    dict(conv_layers=[
                                    dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=4, dropout=0.1)
                                    ], 
                                    use_alexnet_weights=False
                                    ),
            'loss': 'kldiv',
            'learning_rate': 0.001
          }

dconvsegm_04 = {
            'model_class': Custom_DownsamplingSegmentation,
            'model_init_args': 
                    dict(conv_layers=[
                                    dict(kernel_size=8, filters=96, activation='relu', padding='valid', strides=4, dropout=0.1)
                                    ], 
                                    use_alexnet_weights=False
                                    ),
            'loss': 'kldiv',
            'learning_rate': 0.001
          }

# -------------------------------------
# Convolutions Regression models
# -------------------------------------
kconvreg_01 = {'model_class': KerasBackbone_ConvolutionsRegression,
               'model_init_args': dict(backbone=tf.keras.applications.VGG16)
              }


cconvreg_01 = {
           'model_class': Custom_ConvolutionsRegression,
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


cconvreg_02 = {
            'model_class': Custom_ConvolutionsRegression,
            'model_init_args': 
                    dict(conv_layers=[
                                    dict(kernel_size=4, filters=16, activation='relu', padding='valid', strides=4, dropout=0.1)
                                    ], 
                                    dense_layers=None, use_alexnet_weights=False
                                    ),
            'loss': 'kldiv'
          }

# ------------------------------------------------------
# other models (used at sometime but not yet reviewed)
# ------------------------------------------------------
conv_layers = [
    dict(kernel_size=6, filters=32, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=6,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=6,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=6,  filters=8, activation='relu', padding='valid', strides=1, dropout=0.1),
]

dense_layers = [
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=32, activation='relu', dropout=0.1),
]

model11 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=dense_layers, use_alexnet_weights=False)
          }

conv_layers = [
    dict(kernel_size=2, filters=32, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=2,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=2,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=2,  filters=8, activation='relu', padding='valid', strides=1, dropout=0.1),
]

dense_layers = [
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=32, activation='relu', dropout=0.1),
]

model12 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=dense_layers, use_alexnet_weights=True)
          }

conv_layers = [
    dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=4,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=4,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4,  filters=8, activation='relu', padding='valid', strides=1, dropout=0.1),
]

dense_layers = [
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=32, activation='relu', dropout=0.1),
]

model13 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=dense_layers, use_alexnet_weights=False)
          }

model13b = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=dense_layers, use_alexnet_weights=True)
          }

conv_layers = [
    dict(kernel_size=6, filters=96, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=6,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=6,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=6,  filters=8, activation='relu', padding='valid', strides=1, dropout=0.1),
]

dense_layers = [
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=32, activation='relu', dropout=0.1),
]

model14 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=dense_layers, use_alexnet_weights=False)
          }

model14b = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=dense_layers, use_alexnet_weights=True)
          }


conv_layers = [
    dict(kernel_size=10, filters=96, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=10,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=10,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
]

dense_layers = [
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=64, activation='relu', dropout=0.1),
    dict(units=32, activation='relu', dropout=0.1),
]

model15 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=dense_layers, use_alexnet_weights=True)
          }

conv_layers = [
    dict(kernel_size=2, filters=32, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=2,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=2,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=2,  filters=8, activation='relu', padding='valid', strides=1, dropout=0.1),
]

model16 = {'model_class': Custom_SeparatedConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=4,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=4,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4,  filters=8, activation='relu', padding='valid', strides=1, dropout=0.1),
]

model17 = {'model_class': Custom_SeparatedConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=6, filters=96, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=6,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=6,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=6,  filters=8, activation='relu', padding='valid', strides=1, dropout=0.1),
]

model18 = {'model_class': Custom_SeparatedConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=10, filters=96, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=10,  filters=16, activation='relu', padding='valid', strides=1, dropout=0.1),
    dict(kernel_size=10,  filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
]

model19 = {'model_class': Custom_SeparatedConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }


conv_layers = [
    dict(kernel_size=2, filters=16, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=2, filters=16, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
]

model20 = {'model_class': Custom_SeparatedConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }


conv_layers = [
    dict(kernel_size=6, filters=16, activation='relu', padding='valid', strides=6, dropout=0.1)
]

model22 = {'model_class': Custom_SeparatedConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=12, filters=16, activation='relu', padding='valid', strides=12, dropout=0.1)
]

model23 = {'model_class': Custom_SeparatedConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=2, filters=16, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=2, filters=16, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
]

model24 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=4, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1)
]

model25 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

model25b = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'kldiv'
          }

model25c = {'model_class': Custom_ConvolutionsRegression,
            'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
            'loss': 'mse'
          }

conv_layers = [
    dict(kernel_size=4, filters=48, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4, filters=48, activation='relu', padding='valid', strides=2, dropout=0.1)
]


model25d = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
           }

conv_layers = [
    dict(kernel_size=4, filters=48, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4, filters=48, activation='relu', padding='valid', strides=2, dropout=0.1)
]


model25db = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'kldiv'
           }


conv_layers = [
    dict(kernel_size=4, filters=8, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4, filters=8, activation='relu', padding='valid', strides=2, dropout=0.1)
]

model25e = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
           }


conv_layers = [
    dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4, filters=96, activation='relu', padding='valid', strides=2, dropout=0.1)
]
model25f = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
           }

conv_layers = [
    dict(kernel_size=4, filters=128, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4, filters=129, activation='relu', padding='valid', strides=2, dropout=0.1)
]
model25g = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
           }


conv_layers = [
    dict(kernel_size=6, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=6, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1)
]

model26 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=12, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=12, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1)
]

model27 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }


conv_layers = [
    dict(kernel_size=6, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=2, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1)
]

model28 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=4, filters=16, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1),
    dict(kernel_size=4, filters=16, activation='relu', padding='valid', strides=2, maxpool=2, dropout=0.1)
]

model29 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }

conv_layers = [
    dict(kernel_size=6, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1),
    dict(kernel_size=4, filters=16, activation='relu', padding='valid', strides=2, dropout=0.1)
]

model30 = {'model_class': Custom_ConvolutionsRegression,
           'model_init_args': dict(conv_layers=conv_layers, dense_layers=None, use_alexnet_weights=False),
           'loss': 'ilogkldiv'
          }