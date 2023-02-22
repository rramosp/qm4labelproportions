
from .classic import *
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
tfd = tfp.distributions


class GaussianMixtureLayer(tf.keras.layers.Layer):

    def __init__(self, nb_gaussians, name=None ):
        super().__init__(name=name)
        self.nb_gaussians  = nb_gaussians        
        #self._mu    = tf.Variable(np.random.random((nb_gaussians, dim)), dtype=tf.float32, name="_mu")
        #self._sigma = tf.Variable(np.random.random((nb_gaussians,1)), dtype=tf.float32, name="_sigma")
        
        
    def build(self, input_shape):
        
        self.dim = input_shape[-1]
        self._mu = self.add_weight(
            shape=(self.nb_gaussians, self.dim),
            initializer="random_normal",
            trainable=True,
            name="gmmu"
        )

        self._sigma = self.add_weight(
            shape=(self.nb_gaussians,1),
            initializer="random_normal",
            trainable=True,
            name="gmsigma"
        )        

    def call(self, inputs):

        pi = (np.ones(self.nb_gaussians)/self.nb_gaussians).astype(np.float32)
        self._gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pi),
            components_distribution = tfd.MultivariateNormalDiag(
                    loc = self._mu,
                    scale_diag = tf.repeat(tf.math.abs(self._sigma) + 1e-5,
                                           2, axis=1)
                )
        )
        
        outputs =  self._gmm.log_prob(inputs)
        
        repeated_inputs = tf.reshape(tf.repeat(inputs, [self.nb_gaussians], axis=0), 
                                     [-1,self.nb_gaussians,self.dim])
        output_per_distribution = self._gmm.components_distribution.log_prob(repeated_inputs)
        
        return outputs, output_per_distribution

class Custom_GaussianMixture_DensityEstimation(Custom_ConvolutionsRegression):
    
    def __init__(self, 
                 number_of_gaussians, 
                 gaussian_mixture_dim, 
                 gaussian_mixture_sigma,
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        self.gaussian_mixture_dim = gaussian_mixture_dim
        self.gaussian_mixture_sigma = gaussian_mixture_sigma
        self.number_of_gaussians = number_of_gaussians
    
        self.conv_block = Conv2DBlock(self.conv_layers, start_n=1)
        self.gaussian_mixture_block = GaussianMixtureLayer(self.number_of_gaussians, 
                                                           self.gaussian_mixture_dim, 
                                                           self.gaussian_mixture_sigma,
                                                           name='gm')

        if self.dense_layers is not None:
            self.dense_block = DenseBlock(self.dense_layers, start_n=self.conv_block.end_n+1)
            self.output_features_layer = Dense(self.gaussian_mixture_dim, activation='elu', name="features") 
            self.flatten_to_dense_layer = Flatten(name=f'{self.dense_block.end_n+1:02d}_flatten')
        else:
            self.output_conv_layer = Conv2D(kernel_size=1, filters=self.gaussian_mixture_dim, activation='elu', strides=1, name='features')                    
    
    def produces_pixel_predictions(self):
        return False    

    def produces_label_proportions(self):
        return False    

    def losses_supported(self):
        return ['custom']
    
    def get_wandb_config(self):
        w = super().get_wandb_config()
        w.update({'gaussian_mixture_dim':self.gaussian_mixture_dim,
                  'gaussian_mixture_sigma': self.gaussian_mixture_sigma,
                  'number_of_gaussians': self.number_of_gaussians})
        return w

    def __get_name__(self):
        name = super().__get_name__()
        return f'gmde_{name}'
    
    def custom_loss(self, p, out):
        """
        self supervised MLE and sparsity promoting loss
        p is ignored
        """                
        _, output_logprobs, output_logprobs_per_distribution = out
        loss = tf.reduce_mean( -output_logprobs + tf.reduce_min(-output_logprobs_per_distribution, axis=1))
        return loss
    
    def get_loss_components(self, p, out):
        _, output_logprobs, output_logprobs_per_distribution = out
        r = {
                'gm_logprob': tf.reduce_mean( -output_logprobs ),
                'min_logprob': tf.reduce_min( -output_logprobs_per_distribution, axis=1 )
        }
        return r
    
    def get_model(self):
                
        inputs = Input(shape=self.input_shape)
                
        x = self.conv_block(inputs)
            
        if self.dense_layers is not None:
            x = self.flatten_to_dense_layer(x)  
            x = self.dense_block(x)
            output_features = self.output_features_layer(x)            
        else:
            # if there are no dense layers, adds a conv layer with gaussian_mixture_dim 1x1 filters so 
            # that each output pixel outputs a data point in the space of the mixture of gaussians, 
            # then, the average is taken over all pixels to give a single point in that space
            x = self.output_conv_layer(x)            
            output_features = tf.reduce_mean(x, axis=[1,2])
            
        output_logprobs, output_logprobs_per_distribution = self.gaussian_mixture_block(output_features)
                
        model = Model([inputs], [output_features, output_logprobs, output_logprobs_per_distribution])
        
        if self.use_alexnet_weights:
            # if use alexnet, set weights to first conv layer
            print ("setting alexnet weights", flush=True)
            w = model.get_weights()
            walex = get_alexnet_weights(kernel_size=self.conv_layers[0]['kernel_size'])
            w[0] = walex[:,:,:, :w[0].shape[-1]]
            model.set_weights(w)         

        return model