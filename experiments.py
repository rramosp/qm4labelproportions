import psutil
from .lib import data


def run_experiment(datadir,
                   outdir,
                   model_class, 
                   init_args,
                   loss='multiclass_proportions_mse', 
                   partitions_id =  'aschips',
                   epochs=10, 
                   batch_size=32,
                   cache_size=40000,
                   learning_rate=0.0001,
                   class_weights=None,
                   wproject=None,
                   wentity=None,
                   data_generator_class = data.S2LandcoverDataGenerator
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
                data_generator_class = data_generator_class)
    print ("-----", psutil.virtual_memory())
    pcs.fit(epochs=epochs)
    pcs.plot_val_sample(10);
    r = pcs.summary_result()
    pcs.empty_caches()
    print (r)
    return pcs
