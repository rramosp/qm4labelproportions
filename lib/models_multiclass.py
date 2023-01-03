from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Input, Flatten, concatenate, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
from progressbar import progressbar as pbar
from datetime import datetime
from keras.utils.layer_utils import count_params
import numpy as np
import matplotlib.pyplot as plt
import wandb
from . import data
from . import metrics
from rlxutils import subplots
import segmentation_models as sm
import pandas as pd
import gc

def get_sorted_class_weights(class_weights):
    # normalize weights to sum up to 1
    class_weights = {k:v/sum(class_weights.values()) for k,v in class_weights.items()}

    # make sure class ids are ordered
    class_ids = np.sort(list(class_weights.keys()))
    class_w   = np.r_[[class_weights[i] for i in class_ids]]       

    return class_ids, class_w

def multiclass_proportions_mse_on_chip(y_true, y_pred, class_weights):
    k = [tf.reduce_mean(tf.cast(y_true==i, tf.float32), axis=[1,2]) for i in range(12)]
    proportions_y_true = tf.stack(k, axis=1)
    return multiclass_proportions_mse(y_pred, proportions_y_true, class_weights)

def multiclass_proportions_mse(y_pred, proportions, class_weights):
    class_ids, class_w = get_sorted_class_weights(class_weights)

    # select only the proportions of the specified classes (columns)
    proportions_selected = tf.gather(proportions, class_ids, axis=1)

    # compute the proportions on prediction
    proportions_y_pred = tf.reduce_mean(y_pred, axis=[1,2])

    # compute mse using class weights
    r = tf.reduce_mean(
            tf.reduce_sum(
                (proportions_selected - proportions_y_pred)**2 * class_w, 
                axis=-1
            )
    )
    return r

def compute_iou(y_true, y_pred, class_weights):
    class_ids, class_w = get_sorted_class_weights(class_weights)

    # resize smallest
    if y_true.shape[-1]<y_pred.shape[-1]:
        y_true = tf.image.resize(y_true, y_pred.shape[1:], method='nearest')

    if y_pred.shape[-1]<y_true.shape[-1]:
        y_pred = tf.image.resize(y_pred, y_true.shape[1:], method='nearest')
    
    # convert selected class numbers to 0..n for keras MeanIoU to work
    cy_true = tf.reduce_sum(tf.stack([tf.cast(y_true==class_id, tf.int32)*i for i, class_id in enumerate(class_ids)], axis=-1), axis=-1)   
    cy_pred = tf.argmax(y_pred, axis=-1)    
    
    # compute iou
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=len(class_weights))
    iou_metric.reset_states()
    iou = iou_metric(cy_true,  cy_pred)
    return iou


def multiclass_proportions_mse(true_proportions, y_pred, class_weights, binarize=False):
    """
    computes the mse between proportions on probability predictions (y_pred)
    and target_proportions, using the class_weights in this instance.
    
    y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, len(class_ids)] with probability predictions
    target_proportions: a tf tensor of shape [batch_size, n_classes]
    binarize: if true converts any probability >0.5 to 1 and any probability <0.5 to 0 using a steep sigmoid
    
    returns: a float with mse.
    """
    
    #assert len(y_pred.shape)==4 and y_pred.shape[-1]==len(self.class_ids)
    #assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.n_classes
    
    if binarize:            
        y_pred = tf.sigmoid(50*(y_pred-0.5))
    
    # select only the proportions of the specified classes (columns)
    proportions_selected = tf.gather(true_proportions, list(class_weights.keys()), axis=1)

    # compute the proportions on prediction
    proportions_y_pred = tf.reduce_mean(y_pred, axis=[1,2])

    # compute mse using class weights
    r = tf.reduce_mean(
            tf.reduce_sum(
                (proportions_selected - proportions_y_pred)**2 ,
                axis=-1
            )
    )
    return r

class GenericUnet:

    def __init__(self, *args, **kwargs):
      # just to have a constructor accepting parameters
      pass

    def normitem(self, x,l):
        x = x[:,2:-2,2:-2,:]
        l = l[:,2:-2,2:-2]
        return x,l

    def init_run(self, datadir, 
                 learning_rate, 
                 batch_size=32, 
                 train_size=.7, 
                 val_size=0.2, 
                 test_size=0.1,
                 loss='multiclass_proportions_mse',
                 wandb_project = 'qm4labelproportions',
                 wandb_entity = 'rramosp',
                 partitions_id = 'aschips',
                 cache_size = 10000,
                 class_weights = {2: 1, 11:1}):

        self.learning_rate = learning_rate
        self.loss_name = loss
        self.cache_size = cache_size
        self.partitions_id = partitions_id
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.class_weights = class_weights

        self.run_name = f"{self.get_name()}-{self.partitions_id}-{self.loss_name}-{datetime.now().strftime('%Y%m%d[%H%M]')}"
        self.model = self.get_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        self.train_size = train_size
        self.val_size   = val_size
        self.test_size  = test_size
        self.batch_size = batch_size
        self.datadir = datadir

        self.metrics = metrics.ProportionsMetrics(class_weights = class_weights)

        self.tr, self.ts, self.val = data.S2LandcoverDataGenerator.split(
                basedir = self.datadir,
                partitions_id = partitions_id,
                batch_size = self.batch_size,
                train_size = self.train_size,
                test_size = self.test_size, 
                val_size = self.val_size,
                cache_size = cache_size,
                shuffle = True                                             
            )

        if wandb_project is not None:
            wconfig = self.get_wandb_config()
            wandb.init(project=wandb_project, entity=wandb_entity, 
                        name=self.run_name, config=wconfig)
        print ()
        return self

    def empty_caches(self):
        self.tr.empty_cache()
        self.val.empty_cache()
        self.ts.empty_cache()
        gc.collect()


    def measure_iou(self):
        return True

    def get_wandb_config(self):
        self.trainable_params = sum(count_params(layer) for layer in self.model.trainable_weights)
        self.non_trainable_params = sum(count_params(layer) for layer in self.model.non_trainable_weights)
        wconfig = {
            "learning_rate": self.opt.learning_rate,
            "batch_size": self.tr.batch_size,
            'trainable_params': self.trainable_params,
            'non_trainable_params': self.non_trainable_params,
            'loss': self.loss_name
        }
        return wconfig

    def get_val_sample(self, n=10):
        batch_size_backup = self.val.batch_size
        self.val.batch_size = n
        self.val.on_epoch_end()
        gen_val = iter(self.val) 
        val_x, (val_p, val_l) = gen_val.__next__()
        val_x,val_l = self.normitem(val_x,val_l)
        val_out = self.predict(val_x).numpy()

        # restore val dataset config
        self.val.batch_size = batch_size_backup
        self.val.on_epoch_end()
        return val_x, val_p, val_l, val_out

    def plot_val_sample(self, n=10):
        val_x, val_p, val_l, val_out = self.get_val_sample()
        tval_out = (val_out>0.5).astype(int)
        chip_props_on_out = [i.mean() for i in val_out>0.5]
        chip_props_on_label = [i.mean() for i in val_l>0.5]
        y_true = val_l
        y_pred = tval_out
        if self.measure_iou():
            ious = []
            for i in range(len(y_true)):
                iou = compute_iou(y_true[i:i+1], y_pred[i:i+1], self.class_weights).numpy()
                ious.append(iou)

        #ious = np.r_[[get_iou(class_number=i, y_true=val_l, y_pred=tval_out) for i in range(2)]].mean(axis=0)
        for ax,i in subplots(len(val_x)):
            plt.imshow(val_x[i])        
            if i==0: 
                plt.ylabel("input rgb")
                plt.title(f"{self.partitions_id} | {self.loss_name}")

        for ax,i in subplots(len(val_l)):
            plt.imshow(val_l[i])
            if i==0: plt.ylabel("labels")
            plt.title(f"chip props {chip_props_on_label[i]:.2f}")

        for ax,i in subplots(len(val_out)):
            plt.imshow(tval_out[i].argmax(axis=-1))
            title = f"chip props {chip_props_on_out[i]:.2f}"
            if self.measure_iou():
                title += f"  iou {ious[i]:.2f}"
            plt.title(title)
            if i==0: plt.ylabel("thresholded output")

        return val_x, val_p, val_l, val_out

    def get_name(self):
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()

    def get_loss(self, out, p, l):
        if self.loss_name == 'multiclass_proportions_mse':
            return self.metrics.multiclass_proportions_mse(p, out)

        raise ValueError(f"unkown loss '{self.loss_name}'")

    def predict(self, x):
        out = self.model(x)
        #out = tf.keras.layers.Softmax(axis=-1)(out)
        return out

    def fit(self, epochs=10, max_steps=np.inf):

        gen_val = iter(self.val) 
        loss = 0
        for epoch in range(epochs):
            print (f"\nepoch {epoch},  loss {loss:.5f}", flush=True)
            for step_nb,(x,(p,l)) in enumerate(pbar(self.tr)):

                # trim to unet input shape
                x,l = self.normitem(x,l)

                # compute loss
                with tf.GradientTape() as t:
                    out = self.predict(x)
                    loss = self.get_loss(out,p,l)

                grads = t.gradient(loss, self.model.trainable_variables)
                self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

                if self.wandb_project is not None:
                    wandb.log({"train/loss": loss})

                    if self.measure_iou():
                        tr_iou = self.metrics.compute_iou(l, out)
                        wandb.log({"train/iou": tr_iou})

                try:
                    val_x, (val_p, val_l) = gen_val.__next__()
                except:
                    gen_val = iter(self.val) 
                    val_x, (val_p, val_l) = gen_val.__next__()

                val_x,val_l = self.normitem(val_x,val_l)
                val_out = self.predict(val_x)
                val_loss = self.get_loss(val_out,val_p,val_l)

                if self.wandb_project is not None:
                    wandb.log({"val/loss": val_loss})

                    if self.measure_iou():
                        val_iou = self.metrics.compute_iou(val_l, val_out)
                        wandb.log({"val/iou": val_iou})

                    wandb.log({'train/mseprops_on_chip': 
                                    self.metrics.multiclass_proportions_mse_on_chip(l, out)})
                    wandb.log({'val/mseprops_on_chip': 
                                    self.metrics.multiclass_proportions_mse_on_chip(val_l, val_out)})


    def summary_dataset(self, dataset_name):
        assert dataset_name in ['train', 'val', 'test']

        if dataset_name == 'train':
            dataset = self.tr
        elif dataset_name == 'val':
            dataset = self.val
        else:
            dataset = self.ts

        losses, ious, mseps = [], [], []
        for x, (p,l) in pbar(dataset):
            x,l = self.normitem(x,l)
            out = self.predict(x)
            loss = self.get_loss(out,p,l).numpy()
            if self.measure_iou():
                iou = self.metrics.compute_iou(l, out).numpy()

            msep =  self.metrics.multiclass_proportions_mse_on_chip(l, out).numpy()
            losses.append(loss)
            if self.measure_iou():
                ious.append(iou)
            mseps.append(msep)
        if self.measure_iou():
            return {'loss': np.mean(losses), 'iou': np.mean(ious), 'mseprops_on_chip': np.mean(mseps)}
        else:
            return {'loss': np.mean(losses), 'mseprops_on_chip': np.mean(mseps)}
            
    def summary_result(self):
        """
        runs summary_dataset over train, val and test
        returns a dataframe with dataset rows and loss/metric columns
        """
        r = [self.summary_dataset(i) for i in ['train', 'val', 'test']]
        r = pd.DataFrame(r, index = ['train', 'val', 'test'])
        return r


class CustomUnetSegmentation(GenericUnet):

    def get_name(self):
        return "custom_unet"

    def get_model(self):
        input_shape=(96,96,3)
        # Build U-Net model
        inputs = Input(input_shape)
        s = Lambda(lambda x: x / 255) (inputs)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

        outputs = Conv2D(len(self.class_weights), (1, 1), activation='softmax') (c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        
        return model


class SMUnetSegmentation(GenericUnet):

    def __init__(self, **sm_keywords):
        self.sm_keywords = sm_keywords
        self.backbone = self.sm_keywords['backbone_name']

    def get_wandb_config(self):
      w = super().get_wandb_config()
      w.update(self.sm_keywords)
      return w

    def get_model(self):
        unet = model = sm.Unet(input_shape=(None,None,3), 
                               **self.sm_keywords)

        inp = tf.keras.layers.Input(shape=(None, None, 3))
        out = unet(inp)
        out = tf.keras.layers.Conv2D(len(self.class_weights), (1,1), padding='same', activation='softmax')(out)
        m = tf.keras.models.Model([inp], [out])
        return m

    def get_name(self):
        return f"{self.backbone}_unet"


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


class PatchProportionsRegression(GenericUnet):

    def __init__(self, input_shape,
                     patch_size, 
                     pred_strides,
                     num_units,
                     dropout_rate):
        self.input_shape = input_shape
        self.patch_size = patch_size 
        self.pred_strides = pred_strides
        self.num_units = num_units
        self.dropout_rate = dropout_rate

    def get_wandb_config(self):
      w = super().get_wandb_config()
      w.update({'patch_size':self.patch_size, 
                'pred_strides': self.pred_strides,
                'num_units':self.num_units})
      return w

    def get_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # Create patches.
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        # Process x using the module blocks.
        x = tf.keras.layers.Dense(self.num_units, activation="gelu")(patches)
        # Apply dropout.
        x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)
        # Label proportions prediction layer
        probs = tf.keras.layers.Dense(len(self.class_weights), activation="softmax")(x)
        out   = tf.reshape(probs, [-1, patch_extr.num_patches, patch_extr.num_patches, len(self.class_weights)])

        m = tf.keras.models.Model([inputs], [out])
        return m

    def get_name(self):
        return f"patch_classifier"

class PatchClassifierSegmentation(GenericUnet):

    def __init__(self, input_shape,
                     patch_size, 
                     pred_strides,
                     num_units,
                     dropout_rate):
        self.input_shape = input_shape
        self.patch_size = patch_size 
        self.pred_strides = pred_strides
        self.num_units = num_units
        self.dropout_rate = dropout_rate

    def get_wandb_config(self):
      w = super().get_wandb_config()
      w.update({'patch_size':self.patch_size, 
                'pred_strides': self.pred_strides,
                'num_units':self.num_units})
      return w

    def get_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # Create patches.
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        # Process x using the module blocks.
        x = tf.keras.layers.Dense(self.num_units, activation="gelu")(patches)
        # Apply dropout.
        x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)
        # Label proportions prediction layer
        probs = tf.keras.layers.Dense(len(self.class_weights), activation="softmax")(x)
        # Construct image from label proportions
        conv2dt = tf.keras.layers.Conv2DTranspose(filters=len(self.class_weights),
                        kernel_size=self.patch_size,
                        strides=self.pred_strides,
                        trainable=True,
                        activation='softmax')
        probs = tf.reshape(probs, [-1, patch_extr.num_patches, patch_extr.num_patches, len(self.class_weights)])
        out = conv2dt(probs)
        m = tf.keras.models.Model([inputs], [out])
        return m

        # conv2dt = tf.keras.layers.Conv2DTranspose(filters=len(self.class_weights),
        #                 kernel_size=self.patch_size,
        #                 strides=self.pred_strides,
        #                 kernel_initializer=tf.keras.initializers.Ones(),
        #                 bias_initializer=tf.keras.initializers.Zeros(),
        #                 trainable=False)        
        #ones = tf.ones_like(probs)
        #out = conv2dt(probs) / conv2dt(ones)
        #m = tf.keras.models.Model([inputs], [out])
        #return m

    def get_name(self):
        return f"patch_classifier_segm"
