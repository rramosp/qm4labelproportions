import tensorflow as tf
import numpy as np
from rlxutils import subplots
import matplotlib.pyplot as plt

class ClassificationMetrics:
    """
    accumulates tp,tn,fp,fn per class and computes metrics on them
    accepts batches of images
    
    y_true: [batch_size, sizex, sizey]                     class labels
    y_pred: [batch_size, sizex, sizey, number_of_classes]  probabilities per class
    
    if sizex,sizey are different in y_true and y_pred the smaller is resized to the larger
    
    """
    def __init__(self, number_of_classes, exclude_classes=[]):
        self.number_of_classes = number_of_classes
        self.classes = [i for i in range(self.number_of_classes) if not i in exclude_classes]
        self.exclude_classes = exclude_classes
        self.reset_state()
        
    def reset_state(self):
        self.tp = {i:0 for i in self.classes}
        self.tn = {i:0 for i in self.classes}
        self.fp = {i:0 for i in self.classes}
        self.fn = {i:0 for i in self.classes}
        self.number_of_pixels = {i:0 for i in self.classes}
        self.total_pixels = 0
        
    def update_state(self, y_true, y_pred):
        # resize smallest
        if y_true.shape[-1]<y_pred.shape[-1]:
            y_true = tf.image.resize(y_true[..., tf.newaxis], y_pred.shape[1:], method='nearest')[:,:,:,0]

        if y_pred.shape[-1]<y_true.shape[-1]:
            y_pred = tf.image.resize(y_pred, y_true.shape[1:], method='nearest')
        
        # choose class with highest probability for prediction
        y_pred = tf.argmax(y_pred, axis=-1)

        self.total_pixels += tf.cast(tf.reduce_prod(y_true.shape), tf.float32)
        for i in self.classes:
            y_true_ones  = tf.cast(y_true==i, tf.float32)
            y_true_zeros = tf.cast(y_true!=i, tf.float32)
            y_pred_ones  = tf.cast(y_pred==i, tf.float32)
            y_pred_zeros = tf.cast(y_pred!=i, tf.float32)
            
            self.tp[i] += tf.reduce_sum(y_true_ones  * y_pred_ones)
            self.tn[i] += tf.reduce_sum(y_true_zeros * y_pred_zeros)
            self.fp[i] += tf.reduce_sum(y_true_zeros * y_pred_ones)
            self.fn[i] += tf.reduce_sum(y_true_ones  * y_pred_zeros)
            self.number_of_pixels[i] += tf.reduce_sum(y_true_ones)
            
    def accuracy(self, tp, tn, fp, fn):
        denominator = tp + tn + fp + fn
        if denominator == 0:
            return np.nan
        else:
            return (tp + tn)/denominator
    
    def f1(self, tp, tn, fp, fn):
        denominator = tp + 0.5*(fp+fn)
        if denominator == 0:
            return np.nan
        return tp / denominator
    
    def precision(self, tp, tn, fp, fn):
        denominator = tp+fp
        if denominator == 0:
            return np.nan
        return tp / denominator

    def iou(self, tp, tn, fp, fn):
        denominator = tp + fp + fn
        if denominator == 0:
            return np.nan
        else:
            return tp / denominator

    def result(self, metric_name, mode='micro'):
        if not mode in ['per_class', 'micro', 'macro', 'weighted']:
            raise ValueError(f"invalid mode '{mode}'")
        
        m = eval(f'self.{metric_name}')
        
        if mode=='per_class':
            r = {i: m(self.tp[i], self.tn[i], self.fp[i], self.fn[i]) for i in self.classes}
            return r
        
        if mode=='macro':
            r = {i: m(self.tp[i], self.tn[i], self.fp[i], self.fn[i]) for i in self.classes}
            r = [i for i in r.values() if not np.isnan(i)]
            if len(r)==0:
                return 0.
            else:
                return tf.reduce_mean(r)

        if mode=='weighted':
            total_pixels = tf.reduce_sum(list(self.number_of_pixels.values()))
            if total_pixels == 0:
                return 1.
            r = []
            for i in self.classes:
                metric = m(self.tp[i], self.tn[i], self.fp[i], self.fn[i])
                if not np.isnan(metric):
                    r.append(metric*self.number_of_pixels[i] / total_pixels)

            if len(r)==0:
                return 0.
            else:
                return tf.reduce_sum(r)
        
        if mode=='micro':
            tp = tf.reduce_sum(list(self.tp.values()))
            tn = tf.reduce_sum(list(self.tn.values()))
            fp = tf.reduce_sum(list(self.fp.values()))
            fn = tf.reduce_sum(list(self.fn.values()))
            r = m(tp, tn, fp, fn)
            if np.isnan(r):
                return 0
            else:
                return r

class ProportionsMetrics:
    """
    class containing methods for label proportions metrics and losses
    """

    def __init__(self, class_weights, proportions_argmax=False):
        """
        proportions_argmax: see get_y_pred_as_proportions
        """
        self.class_weights = class_weights
        self.number_of_classes = len(self.class_weights)
        self.proportions_argmax = proportions_argmax
        self.get_sorted_class_weights()
        
    def get_sorted_class_weights(self):
        """
        separates class ids and class weights, normalizes weights and sorts by class_id
        """
        # normalize weights to sum up to 1
        class_weights = {k:v/sum(self.class_weights.values()) for k,v in self.class_weights.items()}

        # make sure class ids are ordered
        self.class_ids = np.sort(list(class_weights.keys()))
        self.class_w   = np.r_[[class_weights[i] for i in self.class_ids]]       

        # sanity checks after code refactoring (should remove self.class_ids completly
        # since we assume they are consecuive starting at zero)
        if self.class_ids[0] != 0:
            raise ValueError("there must be a class zero")

        if not np.allclose(self.class_ids, np.arange(self.number_of_classes)):
            raise ValueError("class_ids must be consecutive starting at zero")

    def generate_y_true(self, batch_size, pixel_size=96, max_samples=5):
        """
        generate a sample of label masks of shape batch_size x pixel_size x pixel_size
        each pixel being an integer value in self.class_ids
        returns a numpy array
        """
        y_true = np.ones((batch_size,pixel_size,pixel_size))
        y_true = y_true * self.class_ids[0]
        for i in range(batch_size):
            for j, class_id in enumerate(self.class_ids[1:]):
                for _ in range(np.random.randint(max_samples-j)+1):
                    size = np.random.randint(30)+10
                    x,y = np.random.randint(y_true.shape[1]-size, size=2)
                    y_true[i,y:y+size,x:x+size] = class_id
        return y_true.astype(np.float32)

    def generate_y_pred(self, batch_size, pixel_size=96, max_samples=5, noise=2):
        """
        generate a sample of probability predictions of shape [batch_size, pixel_size, pixel_size, len(class_ids)]
        each pixel being a probability in [0,1] softmaxed so the each pixel probabilities add up to 1.
        returns a numpy array
        """        
        y_true = self.generate_y_true(batch_size=batch_size, pixel_size=pixel_size, max_samples=max_samples)
        y_pred = np.zeros((batch_size, pixel_size, pixel_size, self.number_of_classes))

        # set classes
        for i, class_id in enumerate(self.class_ids):
            y_pred[:,:,:,i][y_true==class_id] = 1.

        # add random noise    
        y_pred += np.random.random(y_pred.shape)*noise

        # softmax for probabilities across pixels
        y_pred = np.exp(y_pred) / np.exp(y_pred).sum(axis=-1)[:,:,:,np.newaxis]
        return y_pred.astype(np.float32)

    def get_class_proportions_on_masks(self, y_true, dense=True):
        """
        obtains the class proportions in a label mask.
        y_true: int array of shape [batch_size, pixel_size, pixel_size]
        if dense==True: returns an array [batch_size, len(class_ids)]
        else          : returns an array [batch_size, number_of_classes]
        """
        if dense:
            return np.r_[[[np.mean(y_true[i]==class_id)  for class_id in self.class_ids] for i in range(len(y_true))]].astype(np.float32)
        else:
            return np.r_[[[np.mean(y_true[i]==class_id)  for class_id in range(self.number_of_classes)] for i in range(len(y_true))]].astype(np.float32)

    def get_class_proportions_on_probabilities(self, y_pred):
        """
        obtains the class proportions on probability predictions
        y_pred: float array of shape [batch_size, pixel_size, pixel_size, len(class_ids)]
        returns: a tf tensor of shape [batch_size, len(class_ids)]
        """
        assert y_pred.shape[-1] == len(self.class_ids)
        return tf.reduce_mean(y_pred, axis=[1,2])

    def to_dense_proportions(self, proportions):
        """
        converts sparse proportions to dense
        """
        return tf.gather(proportions, self.class_ids, axis=1)
    
    def get_y_pred_as_masks(self, y_pred):
        """
        converts probability predictions to masks by selecting in each pixel the class with highest probability.
        """
        assert y_pred.shape[-1] == len(self.class_ids)
        y_pred_as_label = np.zeros(y_pred.shape[:-1]).astype(int)
        t = y_pred.argmax(axis=-1)
        for i in range(len(self.class_ids)):
            y_pred_as_label[t==i] = self.class_ids[i]
        return y_pred_as_label
    
    def get_y_true_as_probabilities(self, y_true):
        """
        converts masks to probability predictions by setting prob=1 or 0
        """
        r = np.zeros(list(y_true.shape) + [len(self.class_ids)])
        for i,class_id in enumerate(self.class_ids):
            r[:,:,:,i] = y_true==class_id
        return r.astype(np.float32)
    
    
    def get_y_pred_as_proportions(self, y_pred, argmax=None):
        """
        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, number_of_classes] with 
                probability predictions per pixel (such as the output of a softmax layer,so that class 
                proportions will be computed from it), or with shape [batch_size, number_of_classes] 
                directly with the class proportions (must add up to 1).
        argmax: if true compute proportions by selecting the class with highest assigned probability 
                in each pixel and then computing the proportions of selected classes across each image.
                If False, the class proportions will be computed by averaging the probabilities in 
                each class channel. If none, it will use self.proportions_argmax
        
        returns: a tf tensor of shape [batch_size, number_of_classes]
                 if input has shape [batch_size, number_of_classes], the input is returned untouched
        """
        assert (len(y_pred.shape)==4 or len(y_pred.shape)==2) and y_pred.shape[-1]==len(self.class_ids)

        if argmax is None:
            argmax = self.proportions_argmax

        # compute the proportions on prediction
        if len(y_pred.shape)==4:
            # if we have probability predictions per pixel (softmax output)
            if argmax:
                # compute proportions by selecting the class with highest assigned probability in each pixel
                # and then computing the proportions of selected classes across each image
                y_pred_argmax = tf.argmax(y_pred, axis=-1)                
                r = tf.convert_to_tensor([tf.reduce_sum(tf.cast(y_pred_argmax==class_id, tf.float32), axis=[1,2]) \
                                          for class_id in self.class_ids]) / np.prod(y_pred_argmax.shape[-2:])
                r = tf.transpose(r, [1,0])
            else:
                # compute the proportions by averaging each class. Softmax output guarantees all will add up to one.
                r = tf.reduce_mean(y_pred, axis=[1,2])
        else:
            # if we already have a probabilities vector return it as such
            r = y_pred

        return r        


    def multiclass_proportions_mse(self, true_proportions, y_pred, argmax=None):
        """
        computes the mse between proportions on probability predictions (y_pred)
        and target_proportions, using the class_weights in this instance.
        
        y_pred:  see y_pred in get_y_pred_as_proportions
        argmax: see get_y_pred_as_proportions

        returns: a float with mse.
        """
                        
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax)

        # compute mse using class weights
        r = tf.reduce_mean(
                tf.reduce_sum(
                    (true_proportions - proportions_y_pred)**2 * self.class_w, 
                    axis=-1
                )
        )
        return r
        
    def multiclass_proportions_mae(self, true_proportions, y_pred, argmax=None, perclass=False):
        """
        computes the mae between proportions on probability predictions (y_pred)
        and target_proportions. NO CLASS WEIGHTS ARE USED.

        y_pred: see y_pred in get_y_pred_as_proportions
        argmax: see get_y_pred_as_proportions
        perclass: if true returns a vector of length num_classes with the mae on each class

        returns: a float with mse if perclass=False, otherwise a vector
        """
                        
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes

        # compute the proportions on prediction
        proportions_y_pred = self.get_y_pred_as_proportions(y_pred, argmax)

        # compute mae per class
        r = tf.reduce_mean(
            tf.sqrt((true_proportions - proportions_y_pred)**2),
            axis=0
        )

        # return mean if perclass is not required
        if not perclass:
            r = tf.reduce_mean(r)
            
        return r


    def multiclass_LSRN_loss(self, true_proportions, y_pred):
        """
        computes the loss proposed in:
        
         Malkin, Kolya, et al. "Label super-resolution networks." 
         International Conference on Learning Representations. 2018

        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, len(class_ids)] with probability predictions
        target_proportions: a tf tensor of shape [batch_size, number_of_classes]
        
        
        returns: a float with the loss.
        """

        assert len(y_pred.shape)==4 and y_pred.shape[-1]==len(self.class_ids)
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.number_of_classes
        
        eta = true_proportions

        # compute the proportions on prediction (mu)
        mu = tf.reduce_mean(y_pred, axis=[1,2])

        # compute variances (sigma^2)
        block_size = y_pred.shape[1] * y_pred.shape[2]
        sigma_2 = (tf.reduce_sum(y_pred * (1 - y_pred), 
                                axis=[1,2]) / block_size ** 2)
        # compute loss
        loss = tf.reduce_mean(
                tf.reduce_sum(
                    0.01 * (eta - mu)**2 / (sigma_2) +
                    0.99 * tf.math.log(2 * np.pi * sigma_2), 
                    axis=-1
                )
        )
        return loss

    def multiclass_proportions_mae_on_chip(self, y_true, y_pred, argmax=None, perclass=False):
        """
        computes the mse between the proportions observed in a prediction wrt to a mask
        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, len(class_ids)] with probability predictions
        y_true: int array of shape [batch_size, pixel_size, pixel_size]
        perclass: see multiclass_proportions_mae
        argmax: see multiclass_proportions_mae

        returns: a float with mse
        """
        p_true = self.get_class_proportions_on_masks(y_true, dense=False)
        return self.multiclass_proportions_mae(p_true, y_pred, argmax=argmax, perclass=perclass)
    
    def compute_iou(self, y_true, y_pred):
        """
        computes iou using the formula tp / (tp + fp + fn) for each individual image.
        for each image, it computes the iou for each class and then averages only over
        the classes containing pixels in that image in y_true or y_pred.
        NO CLASS WEIGHTS ARE USED.
        """

        if y_true.shape[1]<y_pred.shape[1]:
            y_true = tf.image.resize(y_true[..., tf.newaxis], y_pred.shape[1:], method='nearest')[:,:,:,0]

        if y_pred.shape[1]<y_true.shape[1]:
            y_pred = tf.image.resize(y_pred, y_true.shape[1:], method='nearest')

        itemclass_iou = []
        itemclass_true_or_pred_ones = []
        y_pred = tf.argmax(y_pred, axis=-1)
        for i in range(self.number_of_classes):
            class_id = self.class_ids[i]
            y_true_ones  = tf.cast(y_true==class_id, tf.float32) 
            y_pred_ones  = tf.cast(y_pred==i, tf.float32)
            y_true_zeros = tf.cast(y_true!=class_id, tf.float32) 
            y_pred_zeros = tf.cast(y_pred!=i, tf.float32)

            tp = tf.reduce_sum(y_true_ones  * y_pred_ones, axis=[1,2])
            fp = tf.reduce_sum(y_true_zeros * y_pred_ones, axis=[1,2])
            fn = tf.reduce_sum(y_true_ones  * y_pred_zeros, axis=[1,2])

            true_or_pred_ones = tf.cast(tf.reduce_sum(y_true_ones + y_pred_ones, axis=[1,2])>0, tf.float32)
            iou_this_class = tp / (tp + fp + fn)

            # substitute nans with zeros
            iou_this_class = tf.where(tf.math.is_nan(iou_this_class), tf.zeros_like(iou_this_class), iou_this_class)

            itemclass_iou.append(iou_this_class)
            itemclass_true_or_pred_ones.append(true_or_pred_ones)
            

        itemclass_iou = tf.convert_to_tensor(itemclass_iou)
        itemclass_true_or_pred_ones = tf.convert_to_tensor(itemclass_true_or_pred_ones)
        # only compute the mean of the classes with pixels in y_true or y_pred
        per_item_iou = tf.reduce_sum(itemclass_iou, axis=0)/tf.reduce_sum(itemclass_true_or_pred_ones, axis=0)
        per_item_iou = tf.where(tf.math.is_nan(per_item_iou), tf.ones_like(per_item_iou), per_item_iou)

        return tf.reduce_mean(per_item_iou)   

    def compute_accuracy(self, y_true, y_pred):
        """
        accuracy is computed by summing up the true positives of each class present in the y_true
        and dividing by the total number of pixels. NO CLASS WEIGHTS ARE USED
        
        formally:
        Li: number of total pixels in y_true belonging to class i
        Pi: number of class i pixels in y_true correctly predicted in y_pred (true positives)
                  
           acc =  (sum_i Pi)/ (sum_i Li)
        
        """
        # resize smallest
        if y_true.shape[-1]<y_pred.shape[-1]:
            y_true = tf.image.resize(y_true[..., tf.newaxis], y_pred.shape[1:], method='nearest')[:,:,:,0]

        if y_pred.shape[-1]<y_true.shape[-1]:
            y_pred = tf.image.resize(y_pred, y_true.shape[1:], method='nearest')
        
        # choose class with highest probability for prediction
        y_pred = tf.argmax(y_pred, axis=-1)

        # compute accuracy per class
        hits = []
        total_pixels = []
        for i, class_id in enumerate(self.class_ids):
            y_true_ones = tf.cast(y_true==class_id, tf.float32)
            y_pred_ones = tf.cast(y_pred==i, tf.float32)
            nb_pixels_correct  = tf.reduce_sum( y_true_ones * y_pred_ones ) 
            nb_pixels_in_class = tf.reduce_sum( y_true_ones )  
            hits.append(nb_pixels_correct)
            total_pixels.append(nb_pixels_in_class)

        accuracy = tf.reduce_sum(hits) / tf.reduce_sum(total_pixels)
        return accuracy

    def show_y_pred(self, y_pred):
        for n in range(len(y_pred)):
            for ax,i in subplots(self.number_of_classes, usizex=4):
                plt.imshow(y_pred[n,:,:,i]>=0.5)
                plt.title(f"item {n}, class {self.class_ids[i]}, m {np.mean(y_pred[n,:,:,i]>=0.5):.3f}")
                plt.colorbar();

    def show_y_true(self, y_true):
        for ax,i in subplots(len(y_true)):
            plt.imshow(y_true[i], vmin=0, vmax=11, cmap=plt.cm.tab20b, interpolation="none")
            plt.colorbar()
            plt.title(f"item {i}")

