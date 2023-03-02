import tensorflow as tf
tfkl = tf.keras.layers
import numpy as np

from ..components import kqm as kqmcomps
from ..components.classic import Conv2DBlock, DenseBlock
from .base import *
from ..utils.autoinit import *
'''
A class that implements a KQM Segmentation model as a Keras model.
'''
class QMPatchSegmentation(BaseModel):
    def __init__(self,
                number_of_classes,
                patch_size, 
                pred_strides,
                n_comp,
                sigma_ini,
                deep):
        super().__init__()
        autoinit(self)

        self.dim_x = patch_size ** 2 * 3

        self.sigma = tf.Variable(self.sigma_ini,
                                 dtype=tf.float32,
                                 name="sigma", 
                                 trainable=True)       
        if self.deep:
            # Lenet Model
            self.deep_model = tf.keras.Sequential()
            self.deep_model.add(tfkl.Reshape((self.patch_size, self.patch_size, 3)))
            self.deep_model.add(tfkl.Conv2D(filters=6, kernel_size=(3, 3), 
                                            activation='relu', padding='same'))
            self.deep_model.add(tfkl.AveragePooling2D())
            self.deep_model.add(tfkl.Conv2D(filters=16, kernel_size=(3, 3), 
                                            activation='relu', padding='same'))
            self.deep_model.add(tfkl.AveragePooling2D())
            self.deep_model.add(tfkl.Flatten())
            self.deep_model.add(tfkl.Dense(units=120, activation='relu'))
            self.deep_model.add(tfkl.Dense(units=84, activation='relu'))
            kernel_x = kqmcomps.create_comp_trans_kernel(self.deep_model, 
                                                    kqmcomps.create_rbf_kernel(self.sigma))
        else:
            kernel_x = kqmcomps.create_rbf_kernel(self.sigma)

        self.kqmu = kqmcomps.KQMUnit(kernel_x,
                            dim_x=self.dim_x,
                            dim_y=self.number_of_classes,
                            n_comp=self.n_comp
                            )
        self.patch_extr = kqmcomps.Patches(self.patch_size, 96, self.pred_strides)

    def produces_segmentation_probabilities(self):
        return True    

    def produces_label_proportions(self):
        return True

    def init_model_params(self, run):
        dataloader = run.tr
        batch_size_backup = dataloader.batch_size
        dataloader.batch_size = self.n_comp
        dataloader.on_epoch_end()
        gen_tr = iter(dataloader) 
        tr_x, (tr_p, tr_l) = gen_tr.__next__()
        tr_x, tr_l = run.normitem(tr_x, tr_l)
        self.predict(tr_x)
        patch_extr = kqmcomps.Patches(self.patch_size, 96, self.patch_size)
        patches = patch_extr(tr_x)
        idx = np.random.randint(low=0, high=patch_extr.num_patches ** 2, size=(self.n_comp,))
        patches = tf.gather(patches, idx, axis=1, batch_dims=1)
        self.kqmu.c_x.assign(patches)
        #y = tf.concat([tr_p[:,2:3], 1. - tr_p[:,2:3]], axis=1)
        #y = tf.gather(tr_p, self.metrics.class_ids, axis=1)
        #self.kqmu.c_y.assign(y)
        # restore val dataset config
        dataloader.batch_size = batch_size_backup
        dataloader.on_epoch_end()
        return 


    def call(self, inputs):
        patches = self.patch_extr(inputs)
        w = tf.ones_like(patches[:, :, 0]) / (self.patch_extr.num_patches ** 2)
        rho_x = kqmcomps.comp2dm(w, patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = kqmcomps.dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        # probs = probs[:, tf.newaxis, tf.newaxis, :]
        return probs

    def predict_segmentation(self, inputs):
        '''
        Predicts an output image in contrast to the `call` method 
        that outputs label proportions.
        '''
        patches = self.patch_extr(inputs)
        batch_size = tf.shape(patches)[0]
        indiv_patches = tf.reshape(patches, [batch_size * (self.patch_extr.num_patches ** 2), 
                                             self.dim_x])
        rho_x = kqmcomps.pure2dm(indiv_patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = kqmcomps.dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        # Construct image from label proportions
        conv2dt = tf.keras.layers.Conv2DTranspose(filters=1,
                        kernel_size=self.patch_size,
                        strides=self.pred_strides,
                        kernel_initializer=tf.keras.initializers.Ones(),
                        bias_initializer=tf.keras.initializers.Zeros(),
                        trainable=False)
        probs = tf.reshape(probs, [-1, 
                                   self.patch_extr.num_patches, 
                                   self.patch_extr.num_patches, self.number_of_classes])
        ones = tf.ones_like(probs[..., 0:1])
        outs = []
        for i in range(self.number_of_classes):
            out_i = conv2dt(probs[..., i:i + 1]) / conv2dt(ones)
            outs.append(out_i)
        out = tf.concat(outs, axis=3)
        return out
    


    '''
A class that implements an Autoencoder KQM Segmentation model as a Keras model.
'''
class AEQMPatchSegmModel(BaseModel):
    def __init__(self, 
                number_of_classes,
                patch_size, 
                pred_strides,
                n_comp,
                sigma_ini,
                enc_weights_file=None,
                kqmcx_file=None,
                encoded_size=64):
        super().__init__()
        autoinit(self)
        self.sigma = tf.Variable(self.sigma_ini,
                                 dtype=tf.float32,
                                 name="sigma", 
                                 trainable=True)       
        kernel_x = kqmcomps.create_rbf_kernel(self.sigma)
        self.kqmu = kqmcomps.KQMUnit(kernel_x,
                            dim_x=self.encoded_size,
                            dim_y=self.number_of_classes,
                            n_comp=self.n_comp
                            )
        self.encoder = kqmcomps.create_resize_encoder((self.patch_size, self.patch_size, 3), encoded_size=self.encoded_size)
        self.patch_extr = kqmcomps.Patches(self.patch_size, 96, self.pred_strides)

    def produces_segmentation_probabilities(self):
        return True    

    def produces_label_proportions(self):
        return True

    def init_model_params(self, run):
        dataloader = run.tr

        batch_size_backup = dataloader.batch_size
        dataloader.batch_size = self.n_comp
        dataloader.on_epoch_end()
        gen_tr = iter(dataloader) 
        tr_x, (tr_p, tr_l) = gen_tr.__next__()
        tr_x, tr_l = run.normitem(tr_x, tr_l)
        self.predict(tr_x)
        if self.enc_weights_file is not None:
            self.encoder.load_weights(self.enc_weights_file)
        if self.kqmcx_file is not None:
            patches = np.load(self.kqmcx_file)
            assert patches.shape[0] == self.n_comp and  patches.shape[1] == self.encoded_size, \
                    "Shape of kqm_cx matrix in disk must coincide with model parameter size" 
        else:
            patch_extr = kqmcomps.Patches(self.patch_size, 96, self.patch_size)
            patches = patch_extr(tr_x)
            idx = np.random.randint(low=0, high=patch_extr.num_patches ** 2, size=(self.n_comp,))
            patches = tf.gather(patches, idx, axis=1, batch_dims=1)
            patches = self.encoder(tf.reshape(patches, (-1, self.patch_size, self.patch_size, 3)))
        self.kqmu.c_x.assign(patches)
        #y = tf.concat([tr_p[:,2:3], 1. - tr_p[:,2:3]], axis=1)
        #y = tf.gather(tr_p, self.metrics.class_ids, axis=1)
        #self.kqmu.c_y.assign(y)
        # restore val dataset config
        dataloader.batch_size = batch_size_backup
        dataloader.on_epoch_end()
        return 

    def get_name(self):
        return f"AE_KQM_classifier"

    def call(self, inputs):
        patches = self.patch_extr(inputs)
        patches = self.encoder(tf.reshape(patches, (-1, self.patch_size, self.patch_size, 3)))
        patches = tf.reshape(patches, (-1, 
                                       self.patch_extr.num_patches ** 2, 
                                       self.encoded_size))
        w = tf.ones_like(patches[:, :, 0]) / (self.patch_extr.num_patches ** 2)
        rho_x = kqmcomps.comp2dm(w, patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = kqmcomps.dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        return probs

    def predict_segmentation(self, inputs):
        '''
        Predicts an output image in contrast to the `call` method 
        that outputs label proportions.
        '''
        patches = self.patch_extr(inputs)
        batch_size = tf.shape(patches)[0]
        indiv_patches = tf.reshape(patches, [batch_size * (self.patch_extr.num_patches ** 2), 
                                             self.patch_size, self.patch_size, 3])
        indiv_patches = self.encoder(indiv_patches)
        rho_x = kqmcomps.pure2dm(indiv_patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = kqmcomps.dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        # Construct image from label proportions
        conv2dt = tf.keras.layers.Conv2DTranspose(filters=1,
                        kernel_size=self.patch_size,
                        strides=self.pred_strides,
                        kernel_initializer=tf.keras.initializers.Ones(),
                        bias_initializer=tf.keras.initializers.Zeros(),
                        trainable=False)
        probs = tf.reshape(probs, [-1, 
                                   self.patch_extr.num_patches, 
                                   self.patch_extr.num_patches, 
                                   self.number_of_classes])
        ones = tf.ones_like(probs[..., 0:1])
        outs = []
        for i in range(self.number_of_classes):
            out_i = conv2dt(probs[..., i:i + 1]) / conv2dt(ones)
            outs.append(out_i)
        out = tf.concat(outs, axis=3)
        return out

