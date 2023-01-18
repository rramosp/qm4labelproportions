import tensorflow as tf
import numpy as np
from rlxutils import subplots
import matplotlib.pyplot as plt

class ProportionsMetrics:
    """
    class containing methods for label proportions metrics and losses
    """

    def __init__(self, class_weights, n_classes=12):
        self.class_weights = class_weights
        self.get_sorted_class_weights()
        self.n_classes = n_classes
        
    def get_sorted_class_weights(self):
        """
        separates class ids and class weights, normalizes weights and sorts by class_id
        """
        # normalize weights to sum up to 1
        class_weights = {k:v/sum(self.class_weights.values()) for k,v in self.class_weights.items()}

        # make sure class ids are ordered
        self.class_ids = np.sort(list(class_weights.keys()))
        self.class_w   = np.r_[[class_weights[i] for i in self.class_ids]]       
    
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
        y_pred = np.zeros((batch_size, pixel_size, pixel_size, len(self.class_weights)))

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
        else          : returns an array [batch_size, n_classes]
        """
        if dense:
            return np.r_[[[np.mean(y_true[i]==class_id)  for class_id in self.class_ids] for i in range(len(y_true))]].astype(np.float32)
        else:
            return np.r_[[[np.mean(y_true[i]==class_id)  for class_id in range(self.n_classes)] for i in range(len(y_true))]].astype(np.float32)

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
    
    
    def multiclass_proportions_mse(self, true_proportions, y_pred, binarize=False):
        """
        computes the mse between proportions on probability predictions (y_pred)
        and target_proportions, using the class_weights in this instance.
        
        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, len(class_ids)] with probability predictions
        target_proportions: a tf tensor of shape [batch_size, n_classes]
        binarize: if true converts any probability >0.5 to 1 and any probability <0.5 to 0 using a steep sigmoid
        
        returns: a float with mse.
        """
        
        assert len(y_pred.shape)==4 and y_pred.shape[-1]==len(self.class_ids)
        assert len(true_proportions.shape)==2 and true_proportions.shape[-1]==self.n_classes
        
        if binarize:            
            y_pred = tf.sigmoid(50*(y_pred-0.5))
        
        # select only the proportions of the specified classes (columns)
        proportions_selected = tf.gather(true_proportions, self.class_ids, axis=1)

        # compute the proportions on prediction
        proportions_y_pred = tf.reduce_mean(y_pred, axis=[1,2])

        # compute mse using class weights
        r = tf.reduce_mean(
                tf.reduce_sum(
                    (proportions_selected - proportions_y_pred)**2 * self.class_w, 
                    axis=-1
                )
        )
        return r
    
    def multiclass_proportions_mse_on_chip(self, y_true, y_pred, binarize=False):
        """
        computes the mse between the proportions observed in a prediction wrt to a mask
        y_pred: a tf tensor of shape [batch_size, pixel_size, pixel_size, len(class_ids)] with probability predictions
        y_true: int array of shape [batch_size, pixel_size, pixel_size]
        
        returns: a float with mse
        """
        p_true = self.get_class_proportions_on_masks(y_true, dense=False)
        return self.multiclass_proportions_mse (p_true, y_pred, binarize=binarize)
    
    def compute_iou_batch(self, y_true, y_pred):
        """
        computes MeanIoU just like https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU
        but uses the class weights when averaging per-class IoUs to get a single number to return
        """
        
        # resize smallest
        if y_true.shape[1]<y_pred.shape[1]:
            y_true = tf.image.resize(y_true[..., tf.newaxis], y_pred.shape[1:], method='nearest')[:,:,:,0]

        if y_pred.shape[1]<y_true.shape[1]:
            y_pred = tf.image.resize(y_pred, y_true.shape[1:], method='nearest')
        
        weighted_iou = 0
        weights_used = []
        for i, class_id in enumerate(self.class_ids):
            y_true_ones  = y_true==class_id 
            y_pred_ones  = np.argmax(y_pred, axis=-1)==i
            y_true_zeros = y_true!=class_id 
            y_pred_zeros = np.argmax(y_pred, axis=-1)!=i

            tp = np.sum(y_true_ones  * y_pred_ones)
            fp = np.sum(y_true_zeros * y_pred_ones)
            fn = np.sum(y_true_ones  * y_pred_zeros)

            # only compute IoU for classes with pixels on y_true
            if np.sum(y_true_ones)>0:
                weighted_iou += self.class_w[i] * tp / (tp + fp + fn)
                weights_used.append(self.class_w[i])
                 
        # normalize by the weights of the classes used (sum=1 if all classes are used)
        weighted_iou = weighted_iou / sum(weights_used)
        return weighted_iou

    def compute_iou(self, y_true, y_pred):
        per_item_iou = []
        for i in range(len(y_true)):
            per_item_iou.append(self.compute_iou_batch(y_true[i:i+1], y_pred[i:i+1]).numpy())

        return tf.reduce_mean(per_item_iou)        

    def show_y_pred(self, y_pred):
        for n in range(len(y_pred)):
            for ax,i in subplots(len(self.class_ids), usizex=4):
                plt.imshow(y_pred[n,:,:,i]>=0.5)
                plt.title(f"item {n}, class {self.class_ids[i]}, m {np.mean(y_pred[n,:,:,i]>=0.5):.3f}")
                plt.colorbar();

    def show_y_true(self, y_true):
        for ax,i in subplots(len(y_true)):
            plt.imshow(y_true[i], vmin=0, vmax=11, cmap=plt.cm.tab20b, interpolation="none")
            plt.colorbar()
            plt.title(f"item {i}")

