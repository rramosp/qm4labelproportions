import wget
import os
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from rlxutils import subplots

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from ..models_multiclass import *



def get_alexnet_weights(kernel_size=None):
    """
    Downloads (if needed) alexnet weights of first layers and resizes them
    to (kernel_size, kernel_size)
    """ 
    
    home = os.environ['HOME']
    dest_dir = f"{home}/.alexnet"
    weights_file = f"{dest_dir}/bvlc_alexnet.npy"
    url = 'https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy'
    os.makedirs(dest_dir, exist_ok=True)

    do_download = True
    if os.path.isfile(weights_file):
        if os.path.getsize(weights_file)!=243861814:
            print ("alexnet weights are corrupt! removing them")
            os.remove(weights_file)
        else:
            do_download = False

    if do_download:
        print ("downloading alexnet weights", flush=True)
        wget.download(url, weights_file)
        
    w = np.load(open(weights_file, "rb"), allow_pickle=True, encoding="latin1").item()
    conv1_weights, conv1_bias =  w['conv1']
    # normalize to -1, 1
    conv1_weights = 2 * (conv1_weights-np.min(conv1_weights))/(np.max(conv1_weights)-np.min(conv1_weights)) - 1
    conv1_weights = np.transpose(conv1_weights, [3,0,1,2])
    
    if kernel_size is not None:
        conv1_weights = np.r_[[resize(i, (kernel_size, kernel_size)) for i in conv1_weights]]
    
    conv1_weights = np.transpose(conv1_weights, [1,2,3,0])
    return conv1_weights

def plot_alexnet_weights(w):
    for ax, i in subplots(w.shape[-1], usizex=0.5, usizey=0.5, n_cols=10):
        plt.imshow(w[:,:,:,i])
        plt.axis("off")

def get_conv_model(input_shape, conv_layers, dense_layers, number_of_classes=None):
    
    """
    example execution:
    
        input_shape = (96,96,3)

        conv_layers = [x
            dict(kernel_size=6, filters=3, activation='relu', padding='same', dropout=0.1, maxpool=2),
            dict(kernel_size=6, filters=3, activation='relu', dropout=0.1, maxpool=2)
        ]

        dense_layers = [
            dict(units=100, activation='relu', dropout=0.2),
            dict(units=50, activation='relu')
        ]    
        
        m = get_conv_model(input_shape, conv_layers, dense_layers)
        
    notes:
    - conv_layers is a list of kwargs which will be passed to Conv2D or Dense
    - 'dropout' or 'maxpool' keywords will be extracted before calling Conv2D or Dense
      and, if present, Dropout and MaxPooling2D layers will be added with the value specified
    - dense layers do not use MaxPooling2D.
    - if dense_layers is None adds a conv layer with softmax with n_classes 1x1 filters so 
      that each output pixel outputs a probability distribution. then the probability 
      distributions of all output pixels are averaged to obtain a single output probability 
      distribution per input image.
    
    """    
    if dense_layers is None:
        if number_of_classes is None:
            raise ValueError("must specify 'number_of_classes' when 'dense_layers' is None")
        
        conv_layers.append(dict(kernel_size=1, filters=number_of_classes, activation='softmax', strides=1, name='probabilities'))    
    
    inputs = Input(shape=input_shape)
    x = inputs
    n = 1
    
    x, n = get_conv_layers(x, conv_layers, n)

    if dense_layers is not None:
        x = Flatten(name=f'{n:02d}_flatten')(x)  
        n += 1

        x, n = get_dense_layers(x, dense_layers, n)
    else:
        x = tf.reduce_mean(x, axis=[1,2])
                
    m = Model([inputs],[x])
    
    return m

def get_conv_layers(input_functional_layer, conv_layers, start_n=1, name_prefix=""):
    # see get_conv_model
    n = start_n
    x = input_functional_layer
    
    for kwargs in conv_layers:
        kwargs = kwargs.copy()
        dropout = maxpool = None

        if 'dropout' in kwargs.keys():
            dropout = kwargs['dropout']
            del(kwargs['dropout'])

        if 'maxpool' in kwargs.keys():
            maxpool = kwargs['maxpool']
            del(kwargs['maxpool'])

        if not 'name' in kwargs.keys():
            kwargs['name'] = f"{name_prefix}{n:02d}_conv2d"

        print ("convlayer", kwargs, end=" ", flush=True)
        x = Conv2D(**kwargs)(x)
        print ("output shape", x.shape)

        if dropout is not None:
            x = Dropout(dropout, name=f'{name_prefix}{n:02d}_dropout')(x)

        if maxpool is not None:
            x = MaxPooling2D(pool_size=maxpool, name=f'{name_prefix}{n:02d}_maxpool')(x)

        n += 1
        
    return x, n

def get_dense_layers(input_functional_layer, dense_layers, start_n=1):
    # see get_conv_model

    n = start_n
    x = input_functional_layer
    
    for kwargs in dense_layers:
        kwargs = kwargs.copy()
        dropout = None
        if 'dropout' in kwargs.keys():
            dropout = kwargs['dropout']
            del(kwargs['dropout'])

        if not 'name' in kwargs.keys():
            kwargs['name'] = f"{n:02d}_dense"

        x = Dense(**kwargs)(x)

        if dropout is not None:
            x = Dropout(dropout, name=f'{n:02d}_dropout')(x)    

        n += 1
        
    return x, n


class KerasBackbone_ConvolutionsRegression(GenericExperimentModel):
    
    def __init__(self, 
                 backbone, 
                 backbone_kwargs={'weights': None},
                 input_shape=(96,96,3),
                 dense_layers = [   
                    dict(units=1024, activation='relu'),
                    dict(units=1024, activation='relu')
                 ]
                 ):
        """
        backbone: a class under tensorflow.keras.applications
        """
        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.input_shape = input_shape
        self.dense_layers = dense_layers
    
    def get_wandb_config(self):
        w = super().get_wandb_config()
        w.update({'input_shape':self.input_shape,
                    'backbone':self.backbone, 
                    'backbone_kwargs': self.backbone_kwargs,
                    'dense_layers': self.dense_layers})
        return w

    def __get_name__(self):
        r = f"convregr_{self.backbone.__name__}"

        if 'weights' in self.backbone_kwargs.keys() and self.backbone_kwargs['weights'] is not None:
            r += f"_{self.backbone_kwargs['weights']}"

        return r

    def produces_pixel_predictions(self):
        return False    
    
    def get_model(self):

        inputs = Input(self.input_shape)
        backcone_output  = self.backbone(include_top=False, input_tensor=inputs, **self.backbone_kwargs)(inputs)
        flat   = Flatten()(backcone_output)

        dense_output, _ = get_dense_layers(flat, self.dense_layers)

        outputs = Dense(self.number_of_classes, activation='softmax')(dense_output)
        model = Model([inputs], [outputs])
        return model

class Custom_ConvolutionsRegression(GenericExperimentModel):
    
    def __init__(self, 
                 input_shape=(96,96,3),
                 conv_layers = [
                     dict(kernel_size=6, filters=3, activation='relu', padding='same', dropout=0.1, maxpool=2),
                     dict(kernel_size=6, filters=3, activation='relu', dropout=0.1, maxpool=2)
                 ],
                 dense_layers = [   
                    dict(units=1024, activation='relu'),
                    dict(units=1024, activation='relu')
                 ],
                 use_alexnet_weights = False
                 ):
        """
        see get_conv_model for example args
        """
        self.input_shape = input_shape
        self.conv_layers = conv_layers.copy()
        self.dense_layers = dense_layers.copy() if dense_layers is not None else None
        self.use_alexnet_weights = use_alexnet_weights
    
    def get_wandb_config(self):
        w = super().get_wandb_config()
        w.update({'input_shape':self.input_shape,
                  'conv_layers': self.conv_layers,
                  'dense_layers': self.dense_layers,
                  'use_alexnet_weights': self.use_alexnet_weights})
        return w

    def __get_name__(self):
        if self.dense_layers is None:
            r = f"convregr_nofc_{len(self.conv_layers)}conv"
        else:
            r = f"convregr_{len(self.conv_layers)}conv_{len(self.dense_layers)}dense"
        return r

    def produces_pixel_predictions(self):
        return False    
    
    def get_model(self):
        m = get_conv_model(self.input_shape, self.conv_layers, self.dense_layers, number_of_classes=self.number_of_classes)

        if self.dense_layers is not None:
            outputs = Dense(self.number_of_classes, activation='softmax', name="probabilities")(m.outputs[0])
            model = Model(m.inputs, [outputs])
        else:
            model = m

        if self.use_alexnet_weights:
            print ("setting alexnet weights", flush=True)
            w = model.get_weights()
            walex = get_alexnet_weights(kernel_size=self.conv_layers[0]['kernel_size'])
            w[0] = walex[:,:,:, :w[0].shape[-1]]
            model.set_weights(w)         

        return model
    
class Custom_SeparatedConvolutionsRegression(GenericExperimentModel):
    """
    this class creates a separated convolutional block for each class starting from
    the same input, and the concatenates their outputs.
    if there are no dense layers an additional convolutions is used to reduce the 
    output of each convolutional block to 1x1 so that each one produces its own probability.
    """
    def __init__(self, 
                 input_shape=(96,96,3),
                 conv_layers = [
                     dict(kernel_size=6, filters=3, activation='relu', padding='same', dropout=0.1, maxpool=2),
                     dict(kernel_size=6, filters=3, activation='relu', dropout=0.1, maxpool=2)
                 ],
                 dense_layers = [   
                    dict(units=1024, activation='relu'),
                    dict(units=1024, activation='relu')
                 ],
                 use_alexnet_weights = False
                 ):
        """
        see get_conv_model for example args
        """
        self.input_shape = input_shape
        self.conv_layers = conv_layers.copy()
        self.dense_layers = dense_layers.copy() if dense_layers is not None else None
        self.use_alexnet_weights = use_alexnet_weights
    
        if use_alexnet_weights:
            raise ValueError("'use alexnet_weights' not yet supported")

    def get_wandb_config(self):
        w = super().get_wandb_config()
        w.update({'input_shape':self.input_shape,
                  'conv_layers': self.conv_layers,
                  'dense_layers': self.dense_layers,
                  'use_alexnet_weights': self.use_alexnet_weights})
        return w

    def __get_name__(self):
        if self.dense_layers is None:
            r = f"sepconvregr_nofc_{len(self.conv_layers)}conv"
        else:
            r = f"sepconvregr_{len(self.conv_layers)}conv_{len(self.dense_layers)}dense"
        return r

    def produces_pixel_predictions(self):
        return False    
    
    def get_model(self):
        inputs = Input(self.input_shape)
        outs = []
        # create a convolutional block of each class
        for i in range(self.number_of_classes):
            out,_ = get_conv_layers(inputs, self.conv_layers, name_prefix=f'{i+1:02d}_')
            if self.dense_layers is None:
                # if there are no dense layers then output a single number per convolution
                out = Conv2D(kernel_size=out.shape[1:3], filters=1, activation='elu', name=f'{i+1:02d}_conv2d_to_one')(out)
            out = Flatten(name=f'{i+1:02d}_flatten')(out)
            outs.append(out)
            
        # concatenate all flattened outputs from separated convolutions
        outputs = tf.concat(outs, axis=-1)

        if self.dense_layers is None:
            outputs = tf.keras.layers.Softmax(name='softmax_output')(outputs)
        else:
            outputs, _ = get_dense_layers(outputs, self.dense_layers, start_n=self.number_of_classes+1)
            outputs = Dense(self.number_of_classes, activation='softmax', name="probabilities")(outputs)
                    
        m = Model([inputs], [outputs])      

        return m