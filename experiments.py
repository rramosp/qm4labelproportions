import psutil
import numpy as np
from .lib import data
from sklearn.model_selection import ParameterSampler
import matplotlib.pyplot as plt


def parameter_sweep(datadir,
                   outdir,
                   model_class,
                   n_iter,
                   init_args,
                   learning_rate=0.0001,
                   batch_size=32,
                   loss='multiclass_proportions_mse', 
                   partitions_id =  'aschips',
                   epochs=10, 
                   cache_size=40000,
                   class_weights=None,
                   wproject=None,
                   wentity=None,
                   data_generator_class = data.S2LandcoverDataGenerator):
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
    for i, params in enumerate(ParameterSampler(param_grid, n_iter=n_iter)):
        print(f"Sweep: {i}")
        print(params)
        lr = params.pop('lr')
        batch_size = params.pop('batch_size')
        # clf = run_experiment(datadir=datadir,
        #            outdir=outdir,
        #            model_class=model_class, 
        #            init_args=params,
        #            learning_rate=lr,
        #            batch_size=batch_size,
        #            loss=loss, 
        #            partitions_id=partitions_id,
        #            epochs=epochs, 
        #            cache_size=cache_size,
        #            class_weights=class_weights,
        #            wproject=wproject,
        #            wentity=wentity,
        #            data_generator_class=data_generator_class
        #           )
        # if hasattr(clf, 'run_id'):
        #     run_ids.append(clf.run_id)
    return run_ids

def run_experiment(datadir,
                   outdir,
                   model_class, 
                   init_args,
                   learning_rate=0.0001,
                   batch_size=32,
                   loss='multiclass_proportions_mse', 
                   partitions_id =  'aschips',
                   epochs=10, 
                   cache_size=40000,
                   class_weights=None,
                   wproject=None,
                   wentity=None,
                   data_generator_class = data.S2LandcoverDataGenerator,
                   n_batches_online_val = np.inf
                  ):
    print ("XXXX", loss, "XXXX") 
    print ("\n---------", partitions_id, "------------")

    pcs = model_class(**init_args)

    print ("-----", psutil.virtual_memory())
    pcs.init_run(datadir=datadir,
                outdir=outdir,
                learning_rate=learning_rate, 
                batch_size=batch_size, 
                loss=loss,
                partitions_id = partitions_id,
                wandb_project = wproject,
                wandb_entity = wentity,
                cache_size = cache_size, 
                class_weights = class_weights,
                data_generator_class = data_generator_class,
                n_batches_online_val=n_batches_online_val)
    print ("-----", psutil.virtual_memory())
    pcs.fit(epochs=epochs)
    pcs.plot_val_sample(10); plt.show()
    r = pcs.summary_result()
    pcs.empty_caches()
    print (r)
    return pcs
