from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, Input, Flatten, concatenate, Lambda, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from progressbar import progressbar as pbar
from datetime import datetime
from keras.utils.layer_utils import count_params
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import wandb
from . import data
from . import metrics
from rlxutils import subplots
import segmentation_models as sm
import pandas as pd
import gc
import os


def get_next_file_path(path, base_name, ext):
    i = 1
    while True:
        file_name = f"{base_name}{i}.{ext}"
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1

def get_sorted_class_weights(class_weights):
    # normalize weights to sum up to 1
    class_weights = {k:v/sum(class_weights.values()) for k,v in class_weights.items()}

    # make sure class ids are ordered
    class_ids = np.sort(list(class_weights.keys()))
    class_w   = np.r_[[class_weights[i] for i in class_ids]]       

    return class_ids, class_w

class GenericUnet:

    def __init__(self, *args, **kwargs):
      # just to have a constructor accepting parameters
      pass

    def normitem(self, x,l):
        x = x[:,2:-2,2:-2,:]
        l = l[:,2:-2,2:-2]
        return x,l

    def init_run(self, datadir,
                 outdir,
                 learning_rate, 
                 data_generator_class = data.S2LandcoverDataGenerator,
                 batch_size=32, 
                 train_size=.7, 
                 val_size=0.2, 
                 test_size=0.1,
                 loss='multiclass_proportions_mse',
                 wandb_project = 'qm4labelproportions',
                 wandb_entity = 'rramosp',
                 partitions_id = 'aschips',
                 cache_size = 10000,
                 class_weights = None,
                 n_batches_online_val = np.inf,
                 max_chips = None
                ):

        print (f"initializing {self.get_name()}")

        self.learning_rate = learning_rate
        self.loss_name = loss
        self.cache_size = cache_size
        self.partitions_id = partitions_id
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.class_weights = class_weights
        self.n_batches_online_val = n_batches_online_val 

        self.train_size = train_size
        self.val_size   = val_size
        self.test_size  = test_size
        self.batch_size = batch_size
        self.datadir = datadir
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        self.tr, self.ts, self.val = data_generator_class.split(
                basedir = self.datadir,
                partitions_id = partitions_id,
                batch_size = self.batch_size,
                train_size = self.train_size,
                test_size = self.test_size, 
                val_size = self.val_size,
                cache_size = cache_size,
                shuffle = True,
                max_chips = max_chips
            )

        if self.class_weights is None:
            nclasses = self.tr.number_of_classes
            self.class_weights = {i:1/nclasses for i in range(nclasses)}

        self.number_of_classes = len(self.class_weights)

        self.run_name = f"{self.get_name()}-{self.partitions_id}-{self.loss_name}-{datetime.now().strftime('%Y%m%d[%H%M]')}"
        self.train_model, self.val_model = self.get_models()
        self.opt = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        self.metrics = metrics.ProportionsMetrics(class_weights = class_weights, number_of_classes=self.tr.number_of_classes)

        # if there are no class weights, assume equal weight for all classes
        # as defined in the dataloader
        if wandb_project is not None:
            wconfig = self.get_wandb_config()
            wandb.init(project=wandb_project, entity=wandb_entity, 
                        name=self.run_name, config=wconfig)
            self.run_id = wandb.run.id
            self.run_file_path = os.path.join(self.outdir, self.run_id + ".h5")            
        else:
            self.run_file_path = get_next_file_path(self.outdir, "run","h5")
            self.run_id = os.path.basename(self.run_file_path)[:-3]
        self.init_model_params()
        print ()
        return self

    def init_model_params(self):
        '''
        Perform a model parameter initialization if needed
        '''
        pass

    def empty_caches(self):
        self.tr.empty_cache()
        self.val.empty_cache()
        self.ts.empty_cache()
        gc.collect()
    
    def produces_pixel_predictions(self):
        return True    
    
    def get_wandb_config(self):
        self.trainable_params = sum(count_params(layer) for layer in self.train_model.trainable_weights)
        self.non_trainable_params = sum(count_params(layer) for layer in self.train_model.non_trainable_weights)
        wconfig = {
            "learning_rate": self.learning_rate,
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
        val_x, val_p, val_l, val_out = self.get_val_sample(n=n)
        tval_out = np.argmax(val_out, axis=-1)
        cmap=matplotlib.colors.ListedColormap([plt.cm.tab20(i) for i in range(self.number_of_classes)])

        accs = []
        ious = []
        mseprops_onchip = []
        for i in range(len(val_x)):
            msep = self.metrics.multiclass_proportions_rmse_on_chip(val_l[i:i+1], val_out[i:i+1])
            mseprops_onchip.append(msep)
            if self.produces_pixel_predictions():
                acc = self.metrics.compute_accuracy(val_l[i:i+1], val_out[i:i+1])
                accs.append(acc)
                iou = self.metrics.compute_iou(val_l[i:i+1], val_out[i:i+1])
                ious.append(iou)
                
        for ax,i in subplots(len(val_x)):
            plt.imshow(val_x[i])        
            if i==0: 
                plt.ylabel("input rgb")
                plt.title(f"{self.partitions_id}\n{self.loss_name}")

        for ax,i in subplots(len(val_l)):            
            plt.imshow(val_l[i], vmin=0, vmax=self.number_of_classes, cmap=cmap, interpolation='none')
            if i==0: plt.ylabel("labels")

        for ax,i in subplots(len(val_out)):
            title = f"onchip: rmseprop {mseprops_onchip[i]:.4f}"
            if self.produces_pixel_predictions():
                title += f"\nacc {accs[i]:.3f}"
                title += f"  iou {ious[i]:.3f}"
                plt.imshow(tval_out[i], vmin=0, vmax=self.number_of_classes, cmap=cmap, interpolation='none')
                if i==len(val_out)-1:
                    cbar = plt.colorbar(ax=ax, ticks=range(self.number_of_classes))
                    cbar.ax.set_yticklabels([f"{i}" for i in range(self.number_of_classes)])  # vertically oriented colorbar
            plt.title(title)
            if i==0: plt.ylabel("thresholded output")

        n = self.number_of_classes
        y_pred_proportions = self.metrics.get_y_pred_as_proportions(val_out, argmax=True)
        onchip_proportions = self.metrics.get_class_proportions_on_masks(val_l)
        for ax, i in subplots(len(val_x)):
            plt.bar(np.arange(n)-.2, val_p[i], 0.2, label="on partition", alpha=.5)
            plt.bar(np.arange(n), onchip_proportions[i], 0.2, label="on chip", alpha=.5)
            plt.bar(np.arange(n)+.2, y_pred_proportions[i], 0.2, label="pred", alpha=.5)
            if i==len(val_x)-1:
                plt.legend()
            plt.grid();
            plt.xticks(np.arange(n), np.arange(n));
            plt.title("proportions per class")            

        return val_x, val_p, val_l, val_out
    
    def get_name(self):
        raise NotImplementedError()

    def get_models(self):
        raise NotImplementedError()

    def get_loss(self, out, p, l): 
        if self.loss_name == 'multiclass_proportions_mse':
            return self.metrics.multiclass_proportions_mse(p, out)
        if self.loss_name == 'multiclass_proportions_rmse':
            return self.metrics.multiclass_proportions_rmse(p, out)
        if self.loss_name == 'multiclass_LSRN_loss':
            return self.metrics.multiclass_LSRN_loss(p, out)
        raise ValueError(f"unkown loss '{self.loss_name}'")

    def predict(self, x):
        out = self.val_model(x)
        return out

    def get_trainable_variables(self):
        return self.train_model.trainable_variables

    def fit(self, epochs=10, max_steps=np.inf):
        tr_loss = 0
        min_val_loss = np.inf
        for epoch in range(epochs):
            print (f"\nepoch {epoch}", flush=True)
            losses = []
            for step_nb,(x,(p,l)) in enumerate(pbar(self.tr)):
                # trim to unet input shape
                x,l = self.normitem(x,l)
                # compute loss
                with tf.GradientTape() as t:
                    out = self.train_model(x)
                    loss = self.get_loss(out,p,l)
                grads = t.gradient(loss, self.get_trainable_variables())
                self.opt.apply_gradients(zip(grads, self.get_trainable_variables()))
                losses.append(loss.numpy())

            log_dict = {}
            tr_loss = np.mean(losses)
            log_dict['train/loss'] = tr_loss
            
            # measure stuff on validation for reporting
            losses, accs, ious, rmseps = [], [], [], []
            max_value = np.min([len(self.val),self.n_batches_online_val ])
            for i, (x, (p,l)) in pbar(enumerate(self.val), max_value=max_value):
                if i>=self.n_batches_online_val:
                    break
                x,l = self.normitem(x,l)
                out = self.predict(x)
                loss = self.get_loss(out,p,l).numpy()
                losses.append(loss)
                rmseps.append(self.metrics.multiclass_proportions_rmse_on_chip(l, out))
                if self.produces_pixel_predictions():
                    accs.append(self.metrics.compute_accuracy(l, out))
                    ious.append(self.metrics.compute_iou(l,out))
                    
            # summarize validation stuff
            val_loss = np.mean(losses)
            val_mean_rmse = np.mean(rmseps)
            txt_metrics = f"rmse {val_mean_rmse:.5f}"
            if self.produces_pixel_predictions():
                val_mean_acc = np.mean(accs)
                val_mean_iou = np.mean(ious)
                txt_metrics += f" acc {val_mean_acc:.5f} iou {val_mean_iou:.5f}"
                
            # save model if better val loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                self.train_model.save_weights(self.run_file_path)
                
            # log to wandb
            if self.wandb_project is not None:
                log_dict["val/loss"] = val_loss
                if self.produces_pixel_predictions():
                    log_dict["val/acc"] = val_mean_acc
                    log_dict["val/iou"] = val_mean_iou
                log_dict["val/rmseprops_on_chip"] = val_mean_rmse
                wandb.log(log_dict)

            # log to screen
            print (f"epoch {epoch:3d}, train loss {tr_loss:.5f}", flush=True)
            print (f"epoch {epoch:3d},   val loss {val_loss:.5f} {txt_metrics}", flush=True)

    def summary_dataset(self, dataset_name):
        assert dataset_name in ['train', 'val', 'test']

        if dataset_name == 'train':
            dataset = self.tr
        elif dataset_name == 'val':
            dataset = self.val
        else:
            dataset = self.ts

        losses, accs, mseps, ious = [], [], [], []
        for x, (p,l) in pbar(dataset):
            x,l = self.normitem(x,l)
            out = self.predict(x)
            loss = self.get_loss(out,p,l).numpy()
            if self.produces_pixel_predictions():
                acc = self.metrics.compute_accuracy(l, out)
                accs.append(acc)
                iou = self.metrics.compute_iou(l, out)
                ious.append(iou)
                
            msep =  self.metrics.multiclass_proportions_rmse_on_chip(l, out).numpy()
            losses.append(loss)
            mseps.append(msep)
            
        r = {'loss': np.mean(losses), 'rmseprops_on_chip': np.mean(mseps)}
        
        if self.produces_pixel_predictions():
            r['accuracy'] = np.mean(accs)
            r['iou'] = np.mean(iou)

        return r
    
    def summary_result(self):
        """
        runs summary_dataset over train, val and test
        returns a dataframe with dataset rows and loss/metric columns
        """
        self.train_model.load_weights(self.run_file_path)
        r = [self.summary_dataset(i) for i in ['train', 'val', 'test']]
        r = pd.DataFrame(r, index = ['train', 'val', 'test'])
        csv_path = self.run_file_path[:-3] + ".csv"
        r.to_csv(csv_path)
        params_path = self.run_file_path[:-3] + ".params"
        with open(params_path, 'w') as f:
            f.write(repr(self.get_wandb_config()))
        return r


class CustomUnetSegmentation(GenericUnet):

    def __init__(self, initializer='he_normal', input_shape=(96,96,3)):
        self.initializer = initializer    
        self.input_shape = input_shape
    
    def get_name(self):
        return "custom_unet"

    def get_models(self):
        input_shape=self.input_shape
        # Build U-Net model
        inputs = Input(input_shape)
        s = Lambda(lambda x: x / 255) (inputs)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (s)
        c1 = Dropout(0.1) (c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (p1)
        c2 = Dropout(0.1) (c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (p2)
        c3 = Dropout(0.2) (c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (p3)
        c4 = Dropout(0.2) (c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c4)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (p4)
        c5 = Dropout(0.3) (c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (u6)
        c6 = Dropout(0.2) (c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (u7)
        c7 = Dropout(0.2) (c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (u8)
        c8 = Dropout(0.1) (c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (u9)
        c9 = Dropout(0.1) (c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=self.initializer, padding='same') (c9)

        outputs = Conv2D(len(self.class_weights), (1, 1), activation='softmax') (c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        
        return model, model


class SMUnetSegmentation(GenericUnet):

    def __init__(self, **sm_keywords):
        self.sm_keywords = sm_keywords
        self.backbone = self.sm_keywords['backbone_name']

    def get_wandb_config(self):
      w = super().get_wandb_config()
      w.update(self.sm_keywords)
      return w

    def get_models(self):
        self.unet = sm.Unet(input_shape=(None,None,3), 
                            classes = self.number_of_classes, 
                            activation = 'softmax',
                            **self.sm_keywords)

        inp = tf.keras.layers.Input(shape=(None, None, 3))
        out = self.unet(inp)
        #out = tf.keras.layers.Conv2D(len(self.class_weights), (1,1), padding='same', activation='softmax')(out)
        m = tf.keras.models.Model([inp], [out])
        return m, m


    def get_name(self):
        r = f"segmnt_{self.backbone}"

        if 'encoder_weights' in self.sm_keywords.keys() and self.sm_keywords['encoder_weights'] is not None:
            r += f"_{self.sm_keywords['encoder_weights']}"

        return r


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
                     dropout_rate,
                     activation='softmax'):
        self.input_shape = input_shape
        self.patch_size = patch_size 
        self.pred_strides = pred_strides
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.activation = activation

    def get_wandb_config(self):
      w = super().get_wandb_config()
      w.update({'patch_size':self.patch_size, 
                'pred_strides': self.pred_strides,
                'num_units':self.num_units})
      return w

    def get_models(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # Create patches.
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        # Process x using the module blocks.
        x = tf.keras.layers.Dense(self.num_units, activation="gelu")(patches)
        # Apply dropout.
        x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)
        # Label proportions prediction layer
        probs = tf.keras.layers.Dense(len(self.class_weights), activation=self.activation)(x)
        out   = tf.reshape(probs, [-1, patch_extr.num_patches, patch_extr.num_patches, len(self.class_weights)])

        m = tf.keras.models.Model([inputs], [out])
        return m, m

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

    def get_models(self):
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
        return m, m

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


class ConvolutionsRegression(GenericUnet):
    
    def __init__(self, backbone, backbone_kwargs={'weights': None}, input_shape=(96,96,3)):
        """
        backbone: a class under tensorflow.keras.applications
        """
        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.input_shape = input_shape
    
    def get_name(self):
        r = f"convregr_{self.backbone.__name__}"

        if 'weights' in self.backbone_kwargs.keys() and self.backbone_kwargs['weights'] is not None:
            r += f"_{self.backbone_kwargs['weights']}"

        return r

    def produces_pixel_predictions(self):
        return False    
    
    def get_models(self):
        inputs = Input(self.input_shape)
        backcone_output  = self.backbone(include_top=False, input_tensor=inputs, **self.backbone_kwargs)(inputs)
        flat   = Flatten()(backcone_output)
        dense1 = Dense(1024, activation="relu")(flat)
        dense2 = Dense(1024, activation="relu")(dense1)
        outputs = Dense(self.number_of_classes, activation='softmax')(dense2)
        model = Model([inputs], [outputs])
        return model, model