import tensorflow as tf
tfkl = tf.keras.layers
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Resizing, InputLayer, \
                                    Conv2DTranspose, Input, Flatten, concatenate, Lambda, Dense
from tensorflow.keras.models import Model, Sequential
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
import seaborn as sns
import pandas as pd
import gc
import os
from skimage.io import imread



def get_next_file_path(path, base_name, ext):
    i = 1
    while True:
        file_name = f"{base_name}{i}.{ext}"
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1

class GenericUnet:

    def __init__(self, *args, **kwargs):
      # just to have a constructor accepting parameters
      pass

    def get_wandb_config(self):
        self.trainable_params = sum(count_params(layer) for layer in self.train_model.trainable_weights)
        self.non_trainable_params = sum(count_params(layer) for layer in self.train_model.non_trainable_weights)
        wconfig = {
            "model_class":self.__class__.__name__,
            "learning_rate":self.learning_rate,
            "batch_size":self.tr.batch_size,
            'trainable_params':self.trainable_params,
            'non_trainable_params':self.non_trainable_params,
            'class_weights':{str(k):v for k,v in self.class_weights.items()},
            'loss': self.loss_name,
            'partitions_id':self.partitions_id
        }
        return wconfig


    @staticmethod
    def restore_init_from_file(
                 outdir, 
                 file_run_id,
                 data_generator_split_method = data.S2_ESAWorldCover_DataGenerator.split,
                 data_generator_split_args = None,
                 wandb_project = None,
                 wandb_entity = None,                 
                 n_batches_online_val = np.inf):
        with open(outdir + '/' + file_run_id + '.params') as f:
            init_args = eval(f.read())
        model_class =eval(init_args.pop('model_class'))
        loss = init_args.pop('loss', 'multiclass_proportions_mse')
        learning_rate = init_args.pop('learning_rate', 1e-4)
        batch_size = init_args.pop('batch_size', 32)
        class_weights = init_args.pop('class_weights', None)
        partitions_id = init_args.pop('partitions_id', 'aschip')
        init_args.pop('trainable_params') 
        init_args.pop('non_trainable_params')
        model = model_class(**init_args)
        model.init_run(outdir=outdir,
                 learning_rate=learning_rate,
                 file_run_id=file_run_id,
                 loss=loss,
                 wandb_project = wandb_project,
                 wandb_entity = wandb_entity,
                 data_generator_split_method=data_generator_split_method,
                 data_generator_split_args=data_generator_split_args,
                 class_weights=class_weights,
                 n_batches_online_val=n_batches_online_val
                )
        return model

    def normitem(self, x,l):
        x = x[:,2:-2,2:-2,:]
        l = l[:,2:-2,2:-2]
        return x,l

    def init_run(self,
                 outdir,
                 learning_rate,
                 file_run_id=None,
                 loss='multiclass_proportions_mse',
                 wandb_project = 'qm4labelproportions',
                 wandb_entity = 'mindlab',

                 data_generator_split_method = data.S2_ESAWorldCover_DataGenerator.split,
                 data_generator_split_args   = dict(
                        basedir = None,
                        partitions_id = 'aschip',
                        batch_size = 32,
                        train_size = 0.7,
                        test_size = 0.1, 
                        val_size = 0.2,
                        cache_size = 10000,
                        shuffle = True,
                        max_chips = None
                 ),
                 metrics_args = {},
                 class_weights = None,
                 n_batches_online_val = np.inf,
                 n_val_samples = 10,
                 log_imgs = False,
                 log_perclass = False,
                 log_confusion_matrix = False
                ):

        print (f"initializing {self.get_name()}")

        assert 'partitions_id' in data_generator_split_args.keys(), \
               "'data_generator_split_args' must have 'partitions_id'"

        self.log_imgs = log_imgs
        self.log_perclass = log_perclass
        self.log_confusion_matrix = log_confusion_matrix
        self.partitions_id = data_generator_split_args['partitions_id']
        self.learning_rate = learning_rate
        self.loss_name = loss
        self.data_generator_split_method = data_generator_split_method
        self.data_generator_split_args = data_generator_split_args
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.class_weights = class_weights
        self.n_batches_online_val = n_batches_online_val 
        self.metrics_args = metrics_args
        self.outdir = outdir
        self.n_val_samples = n_val_samples
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if self.class_weights is None:
            data_generator_split_args['class_groups'] = None
        else:
            self.class_weights = {k:v/sum(self.class_weights.values()) for k,v in self.class_weights.items()}
            data_generator_split_args['class_groups'] = [i for i in self.class_weights.keys() if i!=0]

        self.tr, self.ts, self.val = data_generator_split_method(**data_generator_split_args)

        self.number_of_classes = self.tr.number_of_output_classes

        # if there are no classweights, distribute class weights evently across all classes
        if self.class_weights is None:
            n = self.number_of_classes
            self.class_weights = {i: 1/n for i in range(n)}

        self.class_weights_values = list(self.class_weights.values())
        
        # if no zero in class weights set its weight to zero
        if not 0 in self.class_weights.keys():
            self.class_weights_values = [0] + self.class_weights_values    

        self.run_name = f"{self.get_name()}-{self.partitions_id}-{self.loss_name}-{datetime.now().strftime('%Y%m%d[%H%M]')}"
        self.train_model, self.val_model = self.get_models()
        self.opt = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        self.metrics = metrics.ProportionsMetrics(class_weights_values = self.class_weights_values, **self.metrics_args)

        if file_run_id is None:
            self.init_model_params()
            if wandb_project is not None:
                wconfig = self.get_wandb_config()
                wandb.init(project=wandb_project, entity=wandb_entity, 
                            name=self.run_name, config=wconfig)
                self.run_id = wandb.run.id
            else:
                run_file_path = get_next_file_path(self.outdir, "run","h5")
                self.run_id = os.path.basename(run_file_path)[:-3]
        else:
            self.run_id = file_run_id
            run_file_path = os.path.join(self.outdir, self.run_id + ".h5")            
            self.train_model.load_weights(run_file_path)

        self.classification_metrics = metrics.ClassificationMetrics(
            number_of_classes=self.number_of_classes
        )

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

    def plot_val_sample(self, n=10, return_fig = False):
        
        shuffle = self.val.shuffle
        self.val.shuffle = False
        val_x, val_p, val_l, val_out = self.get_val_sample(n=n)
        self.val.shuffle = shuffle
        
        n = len(val_x)
        
        tval_out = np.argmax(val_out, axis=-1)
        cmap=matplotlib.colors.ListedColormap([plt.cm.gist_ncar(i/self.number_of_classes) \
                                                for i in range(self.number_of_classes)])

        n_rows = 4 if self.produces_pixel_predictions() else 3
        for ax, ti in subplots(range(n*n_rows), n_cols=n, usizex=3.5):
            i = ti % n
            row = ti // n
            
            if row==0:
                plt.imshow(val_x[i])        
                if i==0: 
                    plt.ylabel("input rgb")
                    plt.title(f"{self.partitions_id}\n{self.loss_name}")

            if row==1:
                plt.imshow(val_l[i], vmin=0, vmax=self.number_of_classes, cmap=cmap, interpolation='none')
                if i==0: 
                    plt.ylabel("labels")
                    
                cbar = plt.colorbar(ax=ax, ticks=range(self.number_of_classes))
                cbar.ax.set_yticklabels([f"{i}" for i in range(self.number_of_classes)])  # vertically oriented colorbar
                    
            if row==2 and self.produces_pixel_predictions():
                self.classification_metrics.reset_state()
                self.classification_metrics.update_state(val_l[i:i+1], val_out[i:i+1])
                f1 = self.classification_metrics.result('f1', 'micro')
                iou = self.metrics.compute_iou(val_l[i:i+1], val_out[i:i+1])
                
                title = f"onchip f1 {f1:.3f} iou {iou:.3f}"
                plt.imshow(tval_out[i], vmin=0, vmax=self.number_of_classes, cmap=cmap, interpolation='none')
                cbar = plt.colorbar(ax=ax, ticks=range(self.number_of_classes))
                cbar.ax.set_yticklabels([f"{i}" for i in range(self.number_of_classes)])  # vertically oriented colorbar
                plt.title(title)
                if i==0: 
                    plt.ylabel("thresholded output")
                
            if (row==2 and not self.produces_pixel_predictions()) or row==3:
                nc = self.number_of_classes
                y_pred_proportions = self.metrics.get_y_pred_as_proportions(val_out[i:i+1], argmax=True)[0]
                onchip_proportions = self.metrics.get_class_proportions_on_masks(val_l[i:i+1])[0]

                maec = self.metrics.multiclass_proportions_mae_on_chip(val_l[i:i+1], val_out[i:i+1])

                plt.bar(np.arange(nc)-.2, val_p[i], 0.2, label="on partition", alpha=.5)
                plt.bar(np.arange(nc), onchip_proportions, 0.2, label="on chip", alpha=.5)
                plt.bar(np.arange(nc)+.2, y_pred_proportions, 0.2, label="pred", alpha=.5)
                if i in [0, n//2, n-1]:
                    plt.legend()
                plt.grid();
                plt.xticks(np.arange(nc), np.arange(nc));
                plt.title(f"maeprops {maec:.3f}")            
                plt.xlabel("class number")
                plt.ylim(0,1)
                plt.ylabel("proportions")
                
        if return_fig:
            fig = plt.gcf()
            plt.close(fig)
            return fig
        else:
            return val_x, val_p, val_l, val_out
    
    def get_name(self):
        r = self.__get_name__()
        if 'tr' in dir(self):
            r = self.tr.basedir.split("/")[-2][:4]+"_"+r
        return r

    def __get_name__(self):
        raise NotImplementedError()


    def get_models(self):
        raise NotImplementedError()

    def get_loss(self, out, p, l): 
        if self.loss_name in ['multiclass_proportions_mse', 'mse']:
            return self.metrics.multiclass_proportions_mse(p, out)
        if self.loss_name in ['multiclass_LSRN_loss', 'lsrn']:
            return self.metrics.multiclass_LSRN_loss(p, out)
        raise ValueError(f"unkown loss '{self.loss_name}'")

    def predict(self, x):
        out = self.val_model(x)
        return out

    def get_trainable_variables(self):
        return self.train_model.trainable_variables

    def fit(self, epochs=10, max_steps=np.inf):
        run_file_path = os.path.join(self.outdir, self.run_id + ".h5")            
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
            losses, ious, maeps, maeps_perclass = [], [], [], []
            max_value = np.min([len(self.val),self.n_batches_online_val ]).astype(int)
            self.classification_metrics.reset_state()
            for i, (x, (p,l)) in pbar(enumerate(self.val), max_value=max_value):
                if i>=self.n_batches_online_val:
                    break
                x,l = self.normitem(x,l)
                out = self.predict(x)
                loss = self.get_loss(out,p,l).numpy()
                losses.append(loss)
                maeps.append(self.metrics.multiclass_proportions_mae_on_chip(l, out))
                maeps_perclass.append(self.metrics.multiclass_proportions_mae_on_chip(l, out, perclass=True).numpy())
                if self.produces_pixel_predictions():
                    ious.append(self.metrics.compute_iou(l,out))

                self.classification_metrics.update_state(l,out)
                    
            # summarize validation stuff
            val_loss = np.mean(losses)
            val_mean_mae = np.mean(maeps)
            txt_metrics = f"mae {val_mean_mae:.5f}"
            if self.produces_pixel_predictions():
                val_mean_f1 = np.mean(self.classification_metrics.result('f1', 'micro'))
                val_mean_iou = np.mean(ious)
                txt_metrics += f" f1 {val_mean_f1:.5f} iou {val_mean_iou:.5f}"
                
            # save model if better val loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                self.train_model.save_weights(run_file_path)
                
            # assemble per class metrics
            r = {'loss': np.mean(losses), 'maeprops_on_chip::global': np.mean(maeps)}
            r.update({f'maeprops_on_chip::class_{k}':v for k,v in zip(range(0, self.number_of_classes), np.r_[maeps_perclass].mean(axis=0))})

            if self.produces_pixel_predictions(): 
                r['f1::global']  = self.classification_metrics.result('f1', 'micro').numpy()
                r['iou::global'] = np.mean(ious)
                r.update({f'f1::class_{k}':tf.constant(v).numpy() for k,v in self.classification_metrics.result('f1', 'per_class').items()})
                r.update({f'iou::class_{k}':tf.constant(v).numpy() for k,v in self.classification_metrics.result('iou', 'per_class').items()})

            df_perclass = pd.DataFrame([{k:v for k,v in r.items() if 'global' not in k and k!='loss'}], index=["val"]).T


            # log to wandb
            if self.wandb_project is not None:
                log_dict["val/loss"] = val_loss
                if self.produces_pixel_predictions():
                    log_dict["val/f1"] = val_mean_f1
                    log_dict["val/iou"] = val_mean_iou
                log_dict["val/maeprops_on_chip"] = val_mean_mae
                if self.log_imgs:
                    log_dict['val/sample'] = self.plot_val_sample(self.n_val_samples, return_fig=True)
                if self.log_perclass:
                    log_dict['val/perclass'] = \
                        wandb.Table(columns = ['metric', 'val'], 
                                    data=[[i,j[0]] for i,j in zip (df_perclass.index, df_perclass.values)])    
                if self.log_confusion_matrix:
                    cm = self.classification_metrics.cm
                    tmpfname = f"/tmp/{np.random.randint(1000000)}.png"
                    fig, ax = plt.subplots(figsize=(5+self.number_of_classes/5,5+self.number_of_classes/5))
                    sns.heatmap(cm/np.sum(cm), fmt='.1%', annot=True, 
                                cmap='Blues', cbar=False, ax=ax,
                                annot_kws={"size": 8})
                    plt.xlabel("y_pred")
                    plt.ylabel("y_true")
                    plt.savefig(tmpfname)
                    plt.close(fig)       
                    img = imread(tmpfname)
                    os.remove(tmpfname)
                    log_dict['val/confusion_matrix'] = wandb.Image(img, caption="confusion matrix")
         
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

        losses, maeps, ious = [], [], []
        mae_perclass = []
        self.classification_metrics.reset_state()
        for x, (p,l) in pbar(dataset):
            x,l = self.normitem(x,l)
            out = self.predict(x)
            if self.produces_pixel_predictions():
                iou = self.metrics.compute_iou(l, out)
                ious.append(iou)
                self.classification_metrics.update_state(l, out)
                
            losses.append(self.get_loss(out,p,l).numpy())
            maeps.append(self.metrics.multiclass_proportions_mae_on_chip(l, out).numpy())
            mae_perclass.append(self.metrics.multiclass_proportions_mae_on_chip(l, out, perclass=True).numpy())
            
        r = {'loss': np.mean(losses), 'maeprops_on_chip::global': np.mean(maeps)}
        r.update({f'maeprops_on_chip::class_{k}':v for k,v in zip(range(0, self.number_of_classes), np.r_[mae_perclass].mean(axis=0))})

        if self.produces_pixel_predictions(): 
            r['f1::global']  = self.classification_metrics.result('f1', 'micro').numpy()
            r['iou::global'] = np.mean(ious)
            r.update({f'f1::class_{k}':tf.constant(v).numpy() for k,v in self.classification_metrics.result('f1', 'per_class').items()})
            r.update({f'iou::class_{k}':tf.constant(v).numpy() for k,v in self.classification_metrics.result('iou', 'per_class').items()})

        return r
    
    def summary_result(self):
        """
        runs summary_dataset over train, val and test
        returns a dataframe with dataset rows and loss/metric columns
        """
        run_file_path = os.path.join(self.outdir, self.run_id + ".h5")            
        self.train_model.load_weights(run_file_path)
        r = [self.summary_dataset(i) for i in ['train', 'val', 'test']]
        r = pd.DataFrame(r, index = ['train', 'val', 'test'])

        r = r[['loss'] + sorted([c for c in r.columns if c!='loss'])]
        r.columns = [" ".join(c.split("::")) for c in r.columns]        
        return r.T


class CustomUnetSegmentation(GenericUnet):

    def __init__(self, initializer='he_normal', input_shape=(96,96,3)):
        self.initializer = initializer    
        self.input_shape = input_shape
    
    def get_wandb_config(self):
        wconfig = super().get_wandb_config()
        wconfig['initializer'] = self.initializer
        wconfig['input_shape'] = self.input_shape
        return wconfig

    def __get_name__(self):
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

        outputs = Conv2D(self.number_of_classes, (1, 1), activation='softmax') (c9)

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
        m   = tf.keras.models.Model([inp], [out])
        return m, m


    def __get_name__(self):
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
        w.update({'input_shape':self.input_shape,
                    'patch_size':self.patch_size, 
                    'pred_strides': self.pred_strides,
                    'num_units':self.num_units,
                    'dropout_rate':self.dropout_rate,
                    'activation':self.activation})
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
        probs = tf.keras.layers.Dense(self.number_of_classes, activation=self.activation)(x)
        out   = tf.reshape(probs, [-1, patch_extr.num_patches, patch_extr.num_patches, self.number_of_classes])

        m = tf.keras.models.Model([inputs], [out])
        return m, m

    def __get_name__(self):
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
        w.update({'input_shape':self.input_shape,
                    'patch_size':self.patch_size, 
                    'pred_strides': self.pred_strides,
                    'num_units':self.num_units,
                    'dropout_rate':self.dropout_rate})
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
        probs = tf.keras.layers.Dense(self.number_of_classes, activation="softmax")(x)
        # Construct image from label proportions
        conv2dt = tf.keras.layers.Conv2DTranspose(filters=self.number_of_classes,
                        kernel_size=self.patch_size,
                        strides=self.pred_strides,
                        trainable=True,
                        activation='softmax')
        probs = tf.reshape(probs, [-1, patch_extr.num_patches, patch_extr.num_patches, self.number_of_classes])
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

    def __get_name__(self):
        return f"patch_classifier_segm"


class ConvolutionsRegression(GenericUnet):
    
    def __init__(self, backbone, backbone_kwargs={'weights': None}, input_shape=(96,96,3)):
        """
        backbone: a class under tensorflow.keras.applications
        """
        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.input_shape = input_shape
    
    def get_wandb_config(self):
        w = super().get_wandb_config()
        w.update({'input_shape':self.input_shape,
                    'backbone':self.backbone, 
                    'backbone_kwargs': self.backbone_kwargs})
        return w

    def __get_name__(self):
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

from . import kqm 
class QMPatchSegmentation(GenericUnet):

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
                self.train_model.trainable_weights + self.deep_model.trainable_weights))
            self.non_trainable_params = sum(count_params(layer) for 
                layer in ( self.train_model.non_trainable_weights + 
                          self.deep_model.non_trainable_weights))
        else:
            self.trainable_params = sum(count_params(layer) for layer in (
                self.train_model.trainable_weights))
            self.non_trainable_params = sum(count_params(layer) for 
                layer in self.train_model.non_trainable_weights)
        w = super().get_wandb_config()
        w.update({'input_shape':self.input_shape,
                    'patch_size':self.patch_size, 
                    'pred_strides': self.pred_strides,
                    'n_comp':self.n_comp,
                    'sigma_ini':self.sigma_ini,
                    'deep':self.deep,
                    'trainable_params':self.trainable_params,
                    'non_trainable_params':self.non_trainable_params})
        return w

    def get_trainable_variables(self):
        if self.deep:
            return (self.train_model.trainable_variables +
                    [self.sigma] +
                    self.deep_model.trainable_variables)
        else:
            return (self.train_model.trainable_variables +
                    [self.sigma])

    def init_model_params(self):
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

    '''
    def get_loss(self, out, p, l):
        if self.loss_name == 'multiclass_proportions_mse':
            p = tf.gather(p, self.metrics.class_ids, axis=1)
            return tf.keras.losses.mse(out,p)
    '''
    def get_models(self):
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
            kernel_x = kqm.create_comp_trans_kernel(self.deep_model, 
                                                    kqm.create_rbf_kernel(self.sigma))
        else:
            kernel_x = kqm.create_rbf_kernel(self.sigma)
        self.kqmu = kqm.KQMUnit(kernel_x,
                            dim_x=self.dim_x,
                            dim_y=self.number_of_classes,
                            n_comp=self.n_comp
                            )
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # Create patches.
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        # Model for training
        w = tf.ones_like(patches[:, :, 0]) / (patch_extr.num_patches ** 2)
        rho_x = kqm.comp2dm(w, patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = kqm.dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        # probs = probs[:, tf.newaxis, tf.newaxis, :]
        train_model =  tf.keras.models.Model([inputs], [probs])

        # Model for prediction
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        batch_size = tf.shape(patches)[0]
        indiv_patches = tf.reshape(patches, [batch_size * (patch_extr.num_patches ** 2), 
                                             self.dim_x])
        rho_x = kqm.pure2dm(indiv_patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = kqm.dm2comp(rho_y)
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
        probs = tf.reshape(probs, [-1, patch_extr.num_patches, patch_extr.num_patches, self.number_of_classes])
        ones = tf.ones_like(probs[..., 0:1])
        outs = []
        for i in range(self.number_of_classes):
            out_i = conv2dt(probs[..., i:i + 1]) / conv2dt(ones)
            outs.append(out_i)
        out = tf.concat(outs, axis=3)
        predict_model = tf.keras.models.Model([inputs], [out])
        return train_model, predict_model


    def __get_name__(self):
        return f"KQM_classifier"

def create_resize_encoder(input_shape, encoded_size=64, filter_size=[32, 64, 128]):
    encoder = Sequential([
        InputLayer(input_shape=input_shape),
        #tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
        Resizing(16, 16),
        Conv2D(filter_size[0], 3, 2,
                    padding='same', activation=tf.nn.relu),
        Conv2D(filter_size[1], 3, 2,
                    padding='same', activation=tf.nn.relu),
        Conv2D(filter_size[2], 3, 2,
                    padding='same', activation=tf.nn.relu),
        #tfk.layers.LayerNormalization(),
        Flatten(),
        Dense(encoded_size,
                activation=None),
                #activity_regularizer=tf.keras.regularizers.l2(1e-4)),
    ])
    return encoder

class AEQMPatchSegmentation(QMPatchSegmentation):

    def __init__(self, enc_weights_file=None, kqmcx_file=None, **kwargs):
        super().__init__(**kwargs)
        self.enc_weights_file = enc_weights_file
        self.kqmcx_file = kqmcx_file

    def __get_name__(self):
        return f"AE_KQM_classifier"

    def get_trainable_variables(self):
        return (self.train_model.trainable_variables +
                [self.sigma])

    def get_models(self):
        self.encoded_size = 64
        self.sigma = tf.Variable(self.sigma_ini, dtype=tf.float32, trainable=True)       
        kernel_x = kqm.create_rbf_kernel(self.sigma)
        self.encoder = create_resize_encoder((self.patch_size, self.patch_size, 3), encoded_size=self.encoded_size)
        self.dim_x = self.encoded_size
        self.kqmu = kqm.KQMUnit(kernel_x,
                            dim_x=self.dim_x,
                            dim_y=self.number_of_classes,
                            n_comp=self.n_comp
                            )
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # Create patches.
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        patches = self.encoder(tf.reshape(patches, (-1, self.patch_size, self.patch_size, 3)))
        patches = tf.reshape(patches, (-1, patch_extr.num_patches ** 2, self.encoded_size))
        # Model for training
        w = tf.ones_like(patches[:, :, 0]) / (patch_extr.num_patches ** 2)
        rho_x = kqm.comp2dm(w, patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = kqm.dm2comp(rho_y)
        norms_y = tf.expand_dims(tf.linalg.norm(y_v, axis=-1), axis=-1)
        y_v = y_v / norms_y
        probs = tf.einsum('...j,...ji->...i', y_w, y_v ** 2, optimize="optimal")
        # probs = probs[:, tf.newaxis, tf.newaxis, :]
        train_model =  tf.keras.models.Model([inputs], [probs])

        # Model for prediction
        patch_extr = Patches(self.patch_size, 96, self.pred_strides)
        patches = patch_extr(inputs)
        batch_size = tf.shape(patches)[0]
        indiv_patches = tf.reshape(patches, [batch_size * (patch_extr.num_patches ** 2), 
                                             self.patch_size, self.patch_size, 3])
        indiv_patches = self.encoder(indiv_patches)
        rho_x = kqm.pure2dm(indiv_patches)
        rho_y = self.kqmu(rho_x)
        y_w, y_v = kqm.dm2comp(rho_y)
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
        probs = tf.reshape(probs, [-1, patch_extr.num_patches, patch_extr.num_patches, self.number_of_classes])
        ones = tf.ones_like(probs[..., 0:1])
        outs = []
        for i in range(self.number_of_classes):
            out_i = conv2dt(probs[..., i:i + 1]) / conv2dt(ones)
            outs.append(out_i)
        out = tf.concat(outs, axis=3)
        predict_model = tf.keras.models.Model([inputs], [out])
        return train_model, predict_model

    def init_model_params(self):
        batch_size_backup = self.tr.batch_size
        self.tr.batch_size = self.n_comp
        self.tr.on_epoch_end()
        gen_tr = iter(self.tr) 
        tr_x, (tr_p, tr_l) = gen_tr.__next__()
        tr_x, tr_l = self.normitem(tr_x, tr_l)
        self.predict(tr_x)
        if self.enc_weights_file is not None:
            self.encoder.load_weights(self.enc_weights_file)
        if self.kqmcx_file is not None:
            patches = np.load(self.kqmcx_file)
            assert patches.shape[0] == self.n_comp and  patches.shape[1] == self.encoded_size, \
                    "Shape of kqm_cx matrix in disk must coincide with model parameter size" 
        else:
            patch_extr = Patches(self.patch_size, 96, self.patch_size)
            patches = patch_extr(tr_x)
            idx = np.random.randint(low=0, high=patch_extr.num_patches ** 2, size=(self.n_comp,))
            patches = tf.gather(patches, idx, axis=1, batch_dims=1)
            patches = self.encoder(tf.reshape(patches, (-1, self.patch_size, self.patch_size, 3)))
        self.kqmu.c_x.assign(patches)
        #y = tf.concat([tr_p[:,2:3], 1. - tr_p[:,2:3]], axis=1)
        #y = tf.gather(tr_p, self.metrics.class_ids, axis=1)
        #self.kqmu.c_y.assign(y)
        # restore val dataset config
        self.tr.batch_size = batch_size_backup
        self.tr.on_epoch_end()
        return 
