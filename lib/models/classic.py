import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from ..models_multiclass import *

def get_conv_model(input_shape, conv_layers, dense_layers):
    
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
    
    """    
    inputs = Input(shape=input_shape)
    x = inputs
    n = 1
    
    x, n = get_conv_layers(x, conv_layers, n)

    x = Flatten(name=f'{n:02d}_flatten')(x)  
    n += 1
    
    x, n = get_dense_layers(x, dense_layers, n)
                
    m = Model([inputs],[x])
    
    return m

def get_conv_layers(input_functional_layer, conv_layers, start_n=1):
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
            kwargs['name'] = f"{n:02d}_conv2d"

        x = Conv2D(**kwargs)(x)

        if dropout is not None:
            x = Dropout(dropout, name=f'{n:02d}_dropout')(x)

        if maxpool is not None:
            x = MaxPooling2D(pool_size=maxpool, name=f'{n:02d}_maxpool')(x)

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
                 ]
                 ):
        """
        see get_conv_model for example args
        """
        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
    
    def get_wandb_config(self):
        w = super().get_wandb_config()
        w.update({'input_shape':self.input_shape,
                  'conv_layers': self.conv_layers,
                  'dense_layers': self.dense_layers})
        return w

    def __get_name__(self):
        r = f"convregr_{len(self.conv_layers)}conv_{len(self.dense_layers)}dense"
        return r

    def produces_pixel_predictions(self):
        return False    
    
    def get_model(self):
        m = get_conv_model(self.input_shape, self.conv_layers, self.dense_layers)

        outputs = Dense(self.number_of_classes, activation='softmax', name="probabilities")(m.outputs[0])
        model = Model(m.inputs, [outputs])
        return model