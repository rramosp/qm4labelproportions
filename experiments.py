import psutil
import os
import numpy as np
import pandas as pd
from .lib import data
from sklearn.model_selection import ParameterSampler
import matplotlib.pyplot as plt
import wandb

def exp_summary(run_ids, outdir, sort_by=[]):
    '''
    Generates a data frame from a list of run_ids. The information is grabed 
    from files in outdir

    Arguments:
        run_ids: a list of strings with run identificators.
        out_dir: path to a folder containing the experimentation files
    Returns:
        A dataframe sorted by val|rmse
    '''
    res_df = None
    for run_id in run_ids:
        df = pd.read_csv(outdir + '/' + run_id + '.csv')
        df1 = df.rename(columns={"maeprops_on_chip":"mae"})
        df1 = df1.set_index('Unnamed: 0').stack().to_frame()
        df1.index = df1.index.map('|'.join)
        with open(outdir + '/' + run_id + '.params') as f:
            params = eval(f.read())
        params['run_id'] = run_id
        df2 = pd.Series(params).to_frame()
        df = pd.concat([df1, df2])
        if res_df is None:
            res_df = df.T
        else:
            res_df = pd.concat([res_df, df.T])
    return res_df.sort_values(by=sort_by)

def parameter_sweep(
                   data_generator_split_method,
                   data_generator_split_args,
                   outdir,
                   model_class,
                   n_iter,
                   init_args,
                   learning_rate=0.0001,
                   loss='multiclass_proportions_mse', 
                   epochs=10, 
                   class_weights=None,
                   wproject=None,
                   wentity=None,
                   n_batches_online_val = np.inf,
                   log_imgs=False,
                   log_perclass=False,
                   metrics_args={},
                   wandb_tags=[]):
    '''
    parameter_sweep works similar as run_experiment but the hyperparameters
    are lists or distributions instead of individual values.
    Same as it happens in sklearn.model_selection.RandomizedSearchCV
    https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-search
    This applies for all the components of init_args, as well as for learning_rate
    and batch_size. n_iter specifies the number of experiments each one corresponds
    to a set of random parameters sampled from the corresponding distributions. 
    '''
    batch_size = data_generator_split_args['batch_size']
    param_grid = dict(init_args)
    if type(learning_rate) is float or type(learning_rate) is int:
        param_grid['lr'] = [learning_rate]
    else:
        param_grid['lr'] = learning_rate
    if type(batch_size) is int:
        param_grid['batch_size'] = [batch_size]
    else:
        param_grid['batch_size'] = batch_size
    run_ids = []
    param_samples = list(ParameterSampler(param_grid, n_iter=n_iter))
    for sampl in param_samples:
        print(sampl)
    for i, params in enumerate(param_samples):
        print(f"\n\n\nSweep: {i}")
        print(params)
        lr = params.pop('lr')
        data_generator_split_args['batch_size'] = params.pop('batch_size')
        clf = run_experiment(
                   data_generator_split_method = data_generator_split_method,
                   data_generator_split_args = data_generator_split_args,
                   outdir=outdir,
                   model_class=model_class, 
                   init_args=params,
                   learning_rate=lr,
                   loss=loss, 
                   epochs=epochs, 
                   class_weights=class_weights,
                   wproject=wproject,
                   wentity=wentity,
                   n_batches_online_val=n_batches_online_val,
                   log_imgs=log_imgs,
                   log_perclass=log_perclass,
                   metrics_args=metrics_args,
                   wandb_tags=wandb_tags
                  )
        if hasattr(clf, 'run_id'):
            run_ids.append(clf.run_id)
        del clf
    return run_ids

def run_experiment(data_generator_split_method,
                   data_generator_split_args,
                   outdir,
                   model_class, 
                   init_args,
                   learning_rate=0.0001,
                   loss='multiclass_proportions_mse', 
                   epochs=10, 
                   class_weights=None,
                   wproject=None,
                   wentity=None,
                   n_batches_online_val = np.inf,
                   log_imgs = False,
                   log_perclass = False,
                   log_confusion_matrix = False,
                   metrics_args = {},
                   wandb_tags = []
                   ):
    print ("\n---------", data_generator_split_args['partitions_id'], "------------")
    print ("using loss", loss) 

    pcs = model_class(**init_args)

    print ("-----", psutil.virtual_memory())
    pcs.init_run(data_generator_split_method = data_generator_split_method,
                 data_generator_split_args = data_generator_split_args,
                 outdir=outdir,
                 learning_rate=learning_rate, 
                 loss=loss,
                 wandb_project = wproject,
                 wandb_entity = wentity,
                 class_weights = class_weights,
                 n_batches_online_val=n_batches_online_val,
                 log_perclass = log_perclass,
                 log_imgs = log_imgs,
                 log_confusion_matrix = log_confusion_matrix,
                 metrics_args = metrics_args
                 )
    if wproject is not None:
        wandb.run.tags = wandb.run.tags + tuple(wandb_tags)             
    print ("-----", psutil.virtual_memory())
    pcs.fit(epochs=epochs)
    pcs.plot_val_sample(10); plt.show()
    r = pcs.summary_result()
    csv_path = os.path.join(outdir, pcs.run_id + '.csv')
    r.to_csv(csv_path)
    if wproject is not None:
        config = {}
        for part in ['train', 'val', 'test']:
            config[f"mae::{part}"] = r[part]['maeprops_on_chip global']
            if pcs.produces_pixel_predictions():
                config[f"f1::{part}"] = r[part]['f1 global']
                config[f"iou::{part}"] = r[part]['iou global']
        wandb.config.update(config)
    params_path = os.path.join(outdir, pcs.run_id + ".params")
    with open(params_path, 'w') as f:
        f.write(repr(pcs.get_wandb_config()))
    pcs.empty_caches()
    
    print ("--------")
    print (r[[i=='loss' or 'global' in i for i in r.index]])
    print ("--------")
    print (r[[i!='loss' and 'global' not in i for i in r.index]])
    print ("--------")

    wandb.finish(quiet=True)
    return pcs
