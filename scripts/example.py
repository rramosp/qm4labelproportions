import sys
import os
import tensorflow as tf

sys.path.insert(0, "..")
os.environ['SM_FRAMEWORK']='tf.keras' 
print ("available GPUs", tf.config.list_physical_devices('GPU'))

from lib.experiments import runs
from lib.data import dataloaders
import numpy as np

# load preset models
from lib.confs import kqm as kqmconfs
from lib.confs import classic as classicconfs
from lib.confs import nlbe

wandb_project = 'qm4lp-test-experiments'
wandb_entity  = 'mindlab'

model = classicconfs.downsampl01
print (model['model_init_args'])


# -----------------------------------
# change these dirs to your settings
# -----------------------------------
basedir = "/home/rlx/data/nlbe"
outdir = "/tmp"

run = runs.Run(
               **model,
               dataloader_split_method = dataloaders.S2_ESAWorldCover_DataLoader.split_per_partition,
               dataloader_split_args = dict (
                    basedir = f'{basedir}/nlbe_sentinel2rgb_2020_landcover/',
                    partitions_id = 'communes',
                    batch_size = 32,
                    cache_size = 1000,
                    shuffle = True,
                    max_chips = 500
                ),
               
               class_weights=nlbe.nlbe_class_weights,
    
               outdir = outdir,
               wandb_project = wandb_project,
               wandb_entity = wandb_entity,
               log_imgs = True,
               log_confusion_matrix = True,
    
               loss = 'kldiv',
               learning_rate = 0.001,
            
               epochs = 2
              )

run.initialize()
run.model.summary()
run.run()
