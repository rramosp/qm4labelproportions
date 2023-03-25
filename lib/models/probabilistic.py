import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

from ..components.classic import *
from ..components.probabilistic import *
from ..utils.autoinit import *



def plot_model(x, y=None, m=None, contours='Pz'):

    assert contours in ['Pz', 'Pc_given_z', 'Pc_given_z/boundary']
    
    if m is not None and 'encoder' in dir(m) and m.encoder is not None:
        x = m.encoder(x).numpy()
    
    sx = np.random.random(size=(10000,2))
    sx = sx*(x.max(axis=0) - x.min(axis=0) ) + x.min(axis=0)
    if m is not None:
        logPz, Pc_given_z = m(sx)
        levels = 14
        vmin=vmax=None
        if contours=='Pz':
            contour_value = -logPz.numpy()
        elif contours == 'Pc_given_z':
            contour_value = Pc_given_z.numpy()
        else:
            contour_value = (Pc_given_z.numpy()>0.5).astype(float)
            vmin=0
            vmax=1
            levels=2
            
        plt.tricontour (sx[:,0], sx[:,1], contour_value, levels=levels, vmin=vmin, vmax=vmax, linewidths=0.5, colors='k')
        plt.tricontourf(sx[:,0], sx[:,1], contour_value, levels=levels, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar()
        
    if y is None or len(x[y==0])==0 or len(x[y==1])==0:
        plt.scatter(x[:,0], x[:,1], color="red", s=10, alpha=.5)
    else:
        plt.scatter(x[y==0][:,0], x[y==0][:,1], color="red", s=10, alpha=.5)
        plt.scatter(x[y==1][:,0], x[y==1][:,1], color="blue", s=10, alpha=.5)

    if m is not None: 
        c = m.gaussian_mixture_block.get_weights()[0]
        plt.scatter(c[:,0], c[:,1], color='white')   
        
    plt.xlim(x[:,0].min(), x[:,0].max())
    plt.ylim(x[:,1].min(), x[:,1].max())
    plt.title(contours)
        

class GMMPrototypes(tf.keras.Model):

    def __init__(self, 
                 number_of_gaussians, 
                 encoder_layers,
                 gm_sigma_value = None,
                 gm_sigma_trainable = True,
                 gm_categorical_weights = 1.,
                 gm_categorical_trainable = False,
                 Pc_given_rk_trainable = True,
                 input_dim = 2):
        super().__init__()
        autoinit(self)

        if encoder_layers is not None:
            self.gaussian_mixture_dim = encoder_layers[-1]['units']
            self.encoder = DenseBlock(self.encoder_layers)
        else:
            self.gaussian_mixture_dim = input_dim
            self.encoder = None
        
        self.gaussian_mixture_block = GaussianMixtureLayer(self.number_of_gaussians, 
                                                           name = 'gm', 
                                                           sigma_value = gm_sigma_value,
                                                           sigma_trainable = gm_sigma_trainable,
                                                           categorical_weights = gm_categorical_weights,
                                                           categorical_trainable = gm_categorical_trainable)
        # Prior on rk
        Prk = tf.ones(number_of_gaussians) / number_of_gaussians
        self.logPrk = tf.math.log(Prk)

        self.Pc_given_rk = self.add_weight(shape=(self.number_of_gaussians,), 
                                           initializer="random_normal", 
                                           name='pc_given_rk', 
                                           trainable=Pc_given_rk_trainable)


    @tf.function
    def call(self, x):
        log = tf.math.log
        exp = tf.math.exp

        # transform data and get probability from prototypes
        logPz, logPz_given_rk = self.Pz_given_rk_model(x)

        # apply Bayes
        logSum_Pz_given_rk = log(1e-7+tf.reduce_sum(exp(logPz_given_rk + self.logPrk), axis=1))
        logPr_given_z = logPz_given_rk + self.logPrk - tf.reshape(logSum_Pz_given_rk, (-1,1))
        
        # compute class probability given each data point
        Pr_given_z = exp(logPr_given_z)
        Pc_given_z = tf.reduce_sum(Pr_given_z * tf.math.sigmoid(5*self.Pc_given_rk), axis=1)        
        
        return logPz, Pc_given_z
    
    def build(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape[1:])
        if self.encoder_layers is not None:
            x = self.encoder(inputs)
        else:
            x = inputs
        x = self.gaussian_mixture_block(x)
        self.Pz_given_rk_model = Model(inputs=[inputs], outputs=x)
        
    def plot(self, train_x, train_y):
        for ax,i in subplots(3, usizex=6, usizey=4):
            if i==0: plot_model( train_x, train_y, m=self, contours='Pz')
            if i==1: plot_model( train_x, train_y, m=self, contours='Pc_given_z')
            if i==2: 
                _, Pc_given_z = self(train_x)
                Pc_given_z = Pc_given_z.numpy()
                acc0 = ((Pc_given_z>0.5)==(train_y==1)).mean()
                acc1 = ((Pc_given_z<0.5)==(train_y==0)).mean()
                plot_model( train_x, train_y, m=self, contours='Pc_given_z/boundary')
                plt.title(f"Pc_given_z/boundary\naccuracy 0 = {acc0:.3f}   accuracy 1 = {acc1:.3f}")
