from ..models import kqm

qmp01 = dict(
            model_class = kqm.QMPatchSegmentation,
            model_init_args = dict(
                                patch_size=6,
                                pred_strides=2,
                                n_comp=64, 
                                sigma_ini=None,
                                deep=False
                            )
            )                            

aeqm = dict(
            model_class = kqm.AEQMPatchSegm,
            model_init_args = dict(
                        patch_size=6,
                        pred_strides=2,
                        n_comp=64, 
                        sigma_ini=None
                    )
        )


qmr01 = dict( 
            model_class = kqm.QMRegression,
            model_init_args = dict(n_comp = 64,sigma_ini = None)
            )
