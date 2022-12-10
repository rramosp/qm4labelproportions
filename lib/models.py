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
from rlxutils import subplots
import segmentation_models as sm
import pandas as pd



def mse_proportions_on_chip(y_true, y_pred): 
    return tf.reduce_mean(
                    (tf.reduce_mean(y_pred, axis=[1,2]) \
                     - tf.reduce_mean(y_true, axis=[1,2]))**2
                )

def get_iou(class_number, y_true, y_pred):
    """
    assumes y_true/y_pred contain a batch of size (batch_size, other_dims)
    returns a list of ious of size batch_size
    """
    iou_batch = []
    cy_pred = (y_pred == class_number).astype(int)
    cy_true = (y_true == class_number).astype(int)
    for i in range(len(cy_pred)):
        union        = ((cy_true[i]+cy_pred[i])>=1).sum()
        intersection = ((cy_true[i]+cy_pred[i])==2).sum()
        iou = 1 if union==0 else intersection/union
        iou_batch.append(iou)
    return np.r_[iou_batch]

class GenericUnet:

    def normitem(self, x,l):
        x = x[:,2:-2,2:-2,:]
        l = l[:,2:-2,2:-2]
        l = (l==2).astype(np.float32)
        return x,l

    def init_run(self, datadir, 
                 learning_rate, 
                 batch_size=32, 
                 train_size=.7, 
                 val_size=0.2, 
                 test_size=0.1,
                 loss='mse',
                 wandb_project = 'qm4labelproportions',
                 wandb_entity = 'rramosp',
                 partitions_id = 'aschips',
                 cache_size = 10000,
                 measure_iou=True):
        self.learning_rate = learning_rate
        self.measure_iou = measure_iou
        self.loss_name = loss
        self.cache_size = cache_size
        self.partitions_id = partitions_id
        self.run_name = f"{self.get_name()}-{self.partitions_id}-{self.loss_name}-{datetime.now().strftime('%Y%m%d[%H%M]')}"
        self.model = self.get_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.dice_loss = sm.losses.DiceLoss()
        self.binxe_loss = tf.keras.losses.BinaryCrossentropy()

        self.train_size = train_size
        self.val_size   = val_size
        self.test_size  = test_size
        self.batch_size = batch_size
        self.datadir = datadir

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

        self.iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)

        wconfig = self.get_wandb_config()
        wandb.init(project=wandb_project, entity=wandb_entity, 
                    name=self.run_name, config=wconfig)
        print ()
        return self

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

        y_true = val_l
        y_pred = tval_out
        if self.measure_iou:
            ious = []
            for i in range(len(y_true)):
                self.iou_metric.reset_state()
                self.iou_metric.update_state(y_true[i:i+1], y_pred[i:i+1])
                ious.append(self.iou_metric.result().numpy())

        #ious = np.r_[[get_iou(class_number=i, y_true=val_l, y_pred=tval_out) for i in range(2)]].mean(axis=0)
        for ax,i in subplots(len(val_x)):
            plt.imshow(val_x[i])        
            if i==0: plt.ylabel("input rgb")

        for ax,i in subplots(len(val_l)):
            plt.imshow(val_l[i])
            if i==0: plt.ylabel("labels")

        for ax,i in subplots(len(val_out)):
            plt.imshow(tval_out[i])
            if self.measure_iou:
                plt.title(f"iou {ious[i]:.2f}")
            if i==0: plt.ylabel("thresholded output")

        for ax,i in subplots(len(val_out)):
            plt.imshow(val_out[i])
            if i==0: plt.ylabel("unthreshold output")

        return val_x, val_p, val_l, val_out

    def get_name(self):
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()

    def get_loss(self, out, p, l):
        if self.loss_name == 'mse':
          return tf.reduce_mean( (l-out)**2)
        elif self.loss_name == 'dice':
          out = tf.sigmoid(50*(out-0.5))
          return self.dice_loss(l, out)
        elif self.loss_name == 'binxe':
          return self.binxe_loss(l, out)
        elif self.loss_name == 'mse_on_proportions':
            out = tf.sigmoid(50*(out-0.5))
            return tf.reduce_mean(
                        (tf.reduce_mean(out, axis=[1,2]) - p[:,2])**2)
        elif self.loss_name == 'pmse_on_proportions':
            return tf.reduce_mean(
                        (tf.reduce_mean(out, axis=[1,2]) - p[:,2])**2)
        elif self.loss_name == 'binxe_on_proportions':
            o = tf.sigmoid(50*(out-0.5))
            return self.binxe_loss(p[:,2], tf.reduce_mean(o, axis=[1,2]))

        raise ValueError(f"unkown loss '{self.loss_name}'")

    def predict(self, x):
        return self.model(x)[:,:,:,0]

    def fit(self, epochs=10):

        gen_val = iter(self.val) 
        
        for epoch in range(epochs):
            print ("\nepoch", epoch, flush=True)
            for x,(p,l) in pbar(self.tr):
                # trim to unet input shape
                x,l = self.normitem(x,l)

                # compute loss
                with tf.GradientTape() as t:
                    out = self.predict(x)
                    loss = self.get_loss(out,p,l)

                grads = t.gradient(loss, self.model.trainable_variables)
                self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
                wandb.log({"train/loss": loss})

                if self.measure_iou:
                    self.iou_metric.reset_states()
                    tr_iou = self.iou_metric(
                                y_true = l, 
                                y_pred = tf.cast(out>0.5, dtype=tf.int32)
                    )
                    wandb.log({"train/iou": tr_iou})

                try:
                    val_x, (val_p, val_l) = gen_val.__next__()
                except:
                    gen_val = iter(self.val) 
                    val_x, (val_p, val_l) = gen_val.__next__()

                val_x,val_l = self.normitem(val_x,val_l)
                val_out = self.predict(val_x)
                val_loss = self.get_loss(val_out,val_p,val_l)
                wandb.log({"val/loss": val_loss})

                if self.measure_iou:
                    self.iou_metric.reset_states()
                    val_iou = self.iou_metric(
                                y_true = val_l, 
                                y_pred = tf.cast(val_out>0.5, dtype=tf.int32)
                    )
                    wandb.log({"val/iou": val_iou})

                wandb.log({'train/mseprops_on_chip': 
                                mse_proportions_on_chip(l, out)})
                wandb.log({'val/mseprops_on_chip': 
                                mse_proportions_on_chip(val_l, val_out)})


    def summary_dataset(self, dataset_name):
        assert dataset_name in ['train', 'val', 'test']

        self.iou_metric.reset_states()
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
            iou = self.iou_metric(
                            y_true = l, 
                            y_pred = tf.cast(out>0.5, dtype=tf.int32)
            ).numpy()
            msep =  mse_proportions_on_chip(l, out).numpy()
            losses.append(loss)
            ious.append(iou)
            mseps.append(msep)
        return {'loss': np.mean(losses), 'iou': ious[-1], 'mseprops_on_chip': np.mean(mseps)}

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
        activation='sigmoid'
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

        outputs = Conv2D(1, (1, 1), activation=activation) (c9)

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
        out = tf.keras.layers.Conv2D(1, (1,1), padding='same', activation='sigmoid')(out)
        m = tf.keras.models.Model([inp], [out])
        return m

    def get_name(self):
        return f"{self.backbone}_unet"

