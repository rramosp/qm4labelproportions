from . import models_multiclass, metrics
import numpy as np
import tensorflow as tf

# Patch extraction as a layer 
# from https://keras.io/examples/vision/mlp_image_classification/
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, image_size, strides):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.strides = strides
        self.num_patches = (image_size - patch_size) // strides + 1 

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.strides, self.strides, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches ** 2, patch_dims])
        return patches


def dm2comp(dm):
    '''
    Extract vectors and weights from a factorized density matrix representation
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
    Returns:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    '''
    return dm[:, :, 0], dm[:, :, 1:]


def comp2dm(w, v):
    '''
    Construct a factorized density matrix from vectors and weights
    Arguments:
     w: tensor of shape (bs, n)
     v: tensor of shape (bs, n, d)
    Returns:
     dm: tensor of shape (bs, n, d + 1)
    '''
    return tf.concat((w[:, :, tf.newaxis], v), axis=2)


def pure2dm(psi):
    '''
    Construct a factorized density matrix to represent a pure state
    Arguments:
     psi: tensor of shape (bs, d)
    Returns:
     dm: tensor of shape (bs, 1, d + 1)
    '''
    ones = tf.ones_like(psi[:, 0:1])
    dm = tf.concat((ones[:,tf.newaxis, :],
                    psi[:,tf.newaxis, :]),
                   axis=2)
    return dm


def dm2distrib(dm, sigma):
    '''
    Creates a Gaussian mixture distribution from the components of a density
    matrix with an RBF kernel 
    Arguments:
     dm: tensor of shape (bs, n, d + 1)
     sigma: sigma parameter of the RBF kernel 
    Returns:
     gm: mixture of Gaussian distribution with shape (bs, )
    '''
    w, v = dm2comp(dm)
    gm = tfd.MixtureSameFamily(reparameterize=True,
            mixture_distribution=tfd.Categorical(
                                    probs=w),
            components_distribution=tfd.Independent( tfd.Normal(
                    loc=v,  # component 2
                    scale=sigma * np.sqrt(2.)),
                    reinterpreted_batch_ndims=1))
    return gm


def pure_dm_overlap(x, dm, kernel):
    '''
    Calculates the overlap of a state  \phi(x) with a density 
    matrix in a RKHS defined by a kernel
    Arguments:
      x: tensor of shape (bs, d)
     dm: tensor of shape (bs, n, d + 1)
     kernel: kernel function 
              k: (bs, d) x (bs, n, d) -> (bs, n)
    Returns:
     overlap: tensor with shape (bs, )
    '''
    w, v = dm2comp(dm)
    overlap = tf.einsum('...i,...i->...', w, kernel(x, v) ** 2)
    return overlap

## Kernels

def create_comp_trans_kernel(transform, kernel):
    '''
    Composes a transformation and a kernel to create a new
    kernel.
    Arguments:
        transform: a function f that transform the input before feeding it to the 
                   kernel
                   f:(bs, d) -> (bs, D) 
        kernel: a kernel function
                k:(bs, n, D)x(m, D) -> (bs, n, m)
    Returns:
        a function that receives 2 tensors with the following shapes
            Input:
                A: tensor of shape (bs, n, d)
                B: tensor of shape (m, d)
            Result:
                K: tensor of shape (bs, n, m)
    '''
    def comp_kernel(A, B):
        shape = tf.shape(A) # (bs, n, d)
        A = tf.reshape(A, [shape[0] * shape[1], shape[2]])
        A = transform(A)
        dim_out = tf.shape(A)[1]
        A = tf.reshape(A, [shape[0], shape[1], dim_out])
        B = transform(B)
        return kernel(A, B)
    return comp_kernel

def create_rbf_kernel(sigma):
    '''
    Builds a function that calculates the rbf kernel between two set of vectors
    Arguments:
        sigma: RBF scale parameter
    Returns:
        a function that receives 2 tensors with the following shapes
        Input:
            A: tensor of shape (bs, n, d)
            B: tensor of shape (m, d)
        Result:
            K: tensor of shape (bs, n, m)
    '''

    def rbf_kernel(A, B):
        shape_A = tf.shape(A)
        shape_B = tf.shape(B)
        A_norm = tf.norm(A, axis=-1)[..., tf.newaxis] ** 2
        B_norm = tf.norm(B, axis=-1)[tf.newaxis, tf.newaxis, :] ** 2
        A_reshaped = tf.reshape(A, [-1, shape_A[2]])
        AB = tf.matmul(A_reshaped, B, transpose_b=True) 
        AB = tf.reshape(AB, [shape_A[0], shape_A[1], shape_B[0]])
        dist2 = A_norm + B_norm - 2. * AB
        dist2 = tf.clip_by_value(dist2, 0., np.inf)
        K = tf.exp(-dist2 / (2 * sigma ** 2))
        return K
    return rbf_kernel


## Layers and models

def l1_loss(vals):
    '''
    Calculate the l1 loss for a batch of vectors
    Arguments:
        vals: tensor with shape (b_size, n)
    '''
    b_size = tf.cast(tf.shape(vals)[0], dtype=tf.float32)
    vals = vals / tf.norm(vals, axis=1)[:, tf.newaxis]
    loss = tf.reduce_sum(tf.abs(vals)) / b_size
    return loss

class KQMUnit(tf.keras.layers.Layer):
    """Kernel Quantum Measurement Unit
    Receives as input a factored density matrix represented by a set of vectors
    and weight values. 
    Returns a resulting factored density matrix.
    Input shape:
        (batch_size, n_comp_in, dim_x + 1)
        where dim_x is the dimension of the input state
        and n_comp_in is the number of components of the input factorization. 
        The weights of the input factorization of sample i are [i, :, 0], 
        and the vectors are [i, :, 1:dim_x + 1].
    Output shape:
        (batch_size, n_comp, dim_y)
        where dim_y is the dimension of the output state
        and n_comp is the number of components used to represent the train
        density matrix. The weights of the
        output factorization for sample i are [i, :, 0], and the vectors
        are [i, :, 1:dim_y + 1].
    Arguments:
        dim_x: int. the dimension of the input state
        dim_y: int. the dimension of the output state
        x_train: bool. Whether to train or not the x compoments of the train
                       density matrix.
        x_train: bool. Whether to train or not the y compoments of the train
                       density matrix.
        w_train: bool. Whether to train or not the weights of the compoments 
                       of the train density matrix. 
        n_comp: int. Number of components used to represent 
                 the train density matrix
        l1_act: float. Coefficient of the regularization term penalizing the l1
                       norm of the activations.
        l1_x: float. Coefficient of the regularization term penalizing the l1
                       norm of the x components.
        l1_y: float. Coefficient of the regularization term penalizing the l1
                       norm of the y components.
    """
    def __init__(
            self,
            kernel,
            dim_x: int,
            dim_y: int,
            x_train: bool = True,
            y_train: bool = True,
            w_train: bool = True,
            n_comp: int = 0, 
            l1_x: float = 0.,
            l1_y: float = 0.,
            l1_act: float = 0.,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = w_train
        self.n_comp = n_comp
        self.l1_x = l1_x
        self.l1_y = l1_y
        self.l1_act = l1_act

    def build(self, input_shape):
        if (input_shape[1] and input_shape[2] != self.dim_x + 1 
            or len(input_shape) != 3):
            raise ValueError(
                f'Input dimension must be (batch_size, m, {self.dim_x + 1} )'
                f' but it is {input_shape}'
                )
        self.c_x = self.add_weight(
            "c_x",
            shape=(self.n_comp, self.dim_x),
            #initializer=tf.keras.initializers.orthogonal(),
            initializer=tf.keras.initializers.random_normal(),
            trainable=self.x_train)
        self.c_y = self.add_weight(
            "c_y",
            shape=(self.n_comp, self.dim_y),
            initializer=tf.keras.initializers.Constant(0.05),
            #initializer=tf.keras.initializers.random_normal(),
            trainable=self.y_train)
        self.comp_w = self.add_weight(
            "comp_w",
            shape=(self.n_comp,),
            initializer=tf.keras.initializers.constant(1./self.n_comp),
            trainable=self.w_train) 
        self.eps = 1e-10
        self.built = True

    def call(self, inputs):
        
        # Weight regularizers
        if self.l1_x != 0:
            self.add_loss(self.l1_x * l1_loss(self.c_x))
        if self.l1_y != 0:
            self.add_loss(self.l1_y * l1_loss(self.c_y))
        comp_w = tf.nn.softmax(self.comp_w)
        in_w = inputs[:, :, 0]  # shape (b, n_comp_in)
        in_v = inputs[:, :, 1:] # shape (b, n_comp_in, dim_x)
        out_vw = self.kernel(in_v, self.c_x)  # shape (b, n_comp_in, n_comp)
        out_w = (tf.expand_dims(tf.expand_dims(comp_w, axis=0), axis=0) *
                 tf.square(out_vw)) # shape (b, n_comp_in, n_comp)
        out_w = tf.maximum(out_w, self.eps) #########
        # out_w_sum = tf.maximum(tf.reduce_sum(out_w, axis=2), self.eps)  # shape (b, n_comp_in)
        out_w_sum = tf.reduce_sum(out_w, axis=2) # shape (b, n_comp_in)
        out_w = out_w / tf.expand_dims(out_w_sum, axis=2)
        out_w = tf.einsum('...i,...ij->...j', in_w, out_w, optimize="optimal")
                # shape (b, n_comp)
        if self.l1_act != 0:
            self.add_loss(self.l1_act * l1_loss(out_w))
        out_w = tf.expand_dims(out_w, axis=-1) # shape (b, n_comp, 1)
        out_y_shape = tf.shape(out_w) + tf.constant([0, 0, self.dim_y - 1])
        out_y = tf.broadcast_to(tf.expand_dims(self.c_y, axis=0), out_y_shape)
        out = tf.concat((out_w, out_y), 2)
        return out

    def get_config(self):
        config = {
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "n_comp": self.n_comp,
            "x_train": self.x_train,
            "y_train": self.y_train,
            "w_train": self.w_train,
            "l1_x": self.l1_x,
            "l1_y": self.l1_y,
            "l1_act": self.l1_act,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (self.dim_y + 1, self.n_comp)


class KQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 n_comp,
                 x_train=True):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.sigma = tf.Variable(0.1, dtype=tf.float32)
        kernel_x = create_rbf_kernel(self.sigma)
        self.kqmu = KQMUnit(kernel_x,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            n_comp=n_comp,
                            x_train=x_train)

    def call(self, inputs):
        rho_x = pure2dm(inputs)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        return probs

class BagKQMClassModel(tf.keras.Model):
    def __init__(self,
                 dim_x,
                 dim_y,
                 n_comp_in,
                 n_comp,
                 x_train=True,
                 l1_y=0.):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.n_comp_in = n_comp_in
        self.sigma = tf.Variable(0.1, dtype=tf.float32)
        kernel_x = create_rbf_kernel(self.sigma)
        self.kqmu = KQMUnit(kernel_x,
                            dim_x=dim_x,
                            dim_y=dim_y,
                            n_comp=n_comp,
                            x_train=x_train,
                            l1_y=l1_y)

    def call(self, inputs):
        w = tf.ones_like(inputs[:, :, 0]) / self.n_comp_in
        rho_x = comp2dm(w, inputs)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        rho_y = comp2dm(y_w, y_v)
        return rho_y


@tf.function
def overlap_kernel(A, B):
    '''
    Calculates the identity kernel between 
    two set of vectors.
    Input:
        A: tensor of shape (bs, d)
        B: tensor of shape (bs, n, d)
    Result:
        K: tensor of shape (bs, n)
    '''
    K = tf.einsum("...d,...nd->...n", A, B)
    return K

def overlap_loss(y_true, y_pred):
    y_true = tf.math.sqrt(y_true)
    overlap = pure_dm_overlap(y_true, y_pred, overlap_kernel)
    #return -tf.reduce_mean(tf.math.log(overlap + 0.0000001), axis=-1) 
    return -tf.reduce_mean(overlap , axis=-1) 


from keras.utils.layer_utils import count_params

tfkl = tf.keras.layers


class QMPatchSegmentation(models_multiclass.GenericUnet):

    def __init__(self, 
                input_shape,
                patch_size, 
                pred_strides,
                n_comp,
                sigma_ini,
                deep):
        self.input_shape = input_shape
        self.patch_size = patch_size 
        self.pred_strides = pred_strides
        self.dim_x = patch_size ** 2 * 3
        self.n_comp = n_comp
        self.sigma_ini = sigma_ini
        self.deep = deep

    def get_wandb_config(self):
        if self.deep:
            self.trainable_params = sum(count_params(layer) for layer in (
                self.model[0].trainable_weights + self.deep_model.trainable_weights))
            self.non_trainable_params = sum(count_params(layer) for 
                layer in ( self.model[0].non_trainable_weights + 
                          self.deep_model.non_trainable_weights))
        else:
            self.trainable_params = sum(count_params(layer) for layer in (
                self.model[0].trainable_weights))
            self.non_trainable_params = sum(count_params(layer) for 
                layer in self.model[0].non_trainable_weights)
            
        wconfig = {
            "learning_rate": self.opt.learning_rate,
            "batch_size": self.tr.batch_size,
            'trainable_params': self.trainable_params,
            'non_trainable_params': self.non_trainable_params,
            'loss': self.loss_name,
            'pred_strides': self.pred_strides,
            'patch_size':self.patch_size, 
            'n_comp':self.n_comp,
            'sigma_ini':self.sigma_ini,
            'deep':self.deep}
        return wconfig

    def predict(self, x):
        return self.model[1](x)

    def fit(self, epochs=10):
        gen_val = iter(self.val) 
        for epoch in range(epochs):
            print ("\nepoch", epoch, flush=True)
            losses = []
            for x,(p,l) in pbar(self.tr):
                # trim to unet input shape
                x,l = self.normitem(x,l)
                p = tf.gather(p, self.metrics.class_ids, axis=1)
                # compute loss
                with tf.GradientTape() as t:
                    out = self.model[0](x)
                    loss = tf.keras.losses.mse(out,p)
                if self.deep:
                    grads = t.gradient(loss, self.model[0].trainable_variables + 
                                    [self.sigma] + self.deep_model.trainable_variables)
                    self.opt.apply_gradients(zip(grads, self.model[0].trainable_variables +
                                                [self.sigma]+ self.deep_model.trainable_variables))
                else:
                    grads = t.gradient(loss, self.model[0].trainable_variables + 
                                    [self.sigma])
                    self.opt.apply_gradients(zip(grads, self.model[0].trainable_variables +
                                                [self.sigma]))

                losses.append(loss.numpy())
            tr_loss = np.mean(losses)
            losses = []
            ious = []
            mseps = []
            print ("\nvalidation", flush=True)
            for x, (p,l) in pbar(self.val):
                x,l = self.normitem(x,l)
                out = self.predict(x)
                loss = self.get_loss(out,p,l).numpy()
                iou = self.metrics.compute_iou(l, out)
                losses.append(loss)
                ious.append(iou)
                mseps.append(self.metrics.multiclass_proportions_mse_on_chip(l, out))
            wandb.log({"train/loss": tr_loss,
                       "val/loss": np.mean(losses),
                       "val/iou": np.mean(ious),
                       "val/msep": np.mean(mseps)})

    def set_kqm_params(self):
        batch_size_backup = self.tr.batch_size
        self.tr.batch_size = self.n_comp
        self.tr.on_epoch_end()
        gen_tr = iter(self.tr) 
        tr_x, (tr_p, tr_l) = gen_tr.__next__()
        tr_x, tr_l = self.normitem(tr_x, tr_l)
        self.predict(tr_x)
        patch_extr = Patches(self.patch_size, 96, self.patch_size)
        patches = patch_extr(tr_x)
        idx = np.random.randint(low=0, high=patch_extr.num_patches ** 2, size=(self.n_comp,))
        patches = tf.gather(patches, idx, axis=1, batch_dims=1)
        self.kqmu.c_x.assign(patches)
        #y = tf.concat([tr_p[:,2:3], 1. - tr_p[:,2:3]], axis=1)
        #y = tf.gather(tr_p, self.metrics.class_ids, axis=1)
        #self.kqmu.c_y.assign(y)
        # restore val dataset config
        self.tr.batch_size = batch_size_backup
        self.tr.on_epoch_end()
        return 

    def get_model(self):
        self.sigma = tf.Variable(self.sigma_ini, dtype=tf.float32, trainable=True)       
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
            kernel_x = create_comp_trans_kernel(self.deep_model, create_rbf_kernel(self.sigma))
        else:
            kernel_x = create_rbf_kernel(self.sigma)
        self.kqmu = KQMUnit(kernel_x,
                            dim_x=self.dim_x,
                            dim_y=len(self.class_weights),
                            n_comp=self.n_comp
                            )
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # Create patches.
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        # Model for training
        w = tf.ones_like(patches[:, :, 0]) / (patch_extr.num_patches ** 2)
        rho_x = comp2dm(w, patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        model_train =  tf.keras.models.Model([inputs], [probs])

        # Model for prediction
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        batch_size = tf.shape(patches)[0]
        indiv_patches = tf.reshape(patches, [batch_size * (patch_extr.num_patches ** 2), 
                                             self.dim_x])
        rho_x = pure2dm(indiv_patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = dm2comp(rho_y)
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
        probs = tf.reshape(probs, [-1, patch_extr.num_patches, patch_extr.num_patches, len(self.class_weights)])
        ones = tf.ones_like(probs[..., 0:1])
        outs = []
        for i in range(len(self.class_weights)):
            out_i = conv2dt(probs[..., i:i + 1]) / conv2dt(ones)
            outs.append(out_i)
        out = tf.concat(outs, axis=3)
        model_predict = tf.keras.models.Model([inputs], [out])
        return model_train, model_predict

    def get_name(self):
        return f"KQM_classifier"