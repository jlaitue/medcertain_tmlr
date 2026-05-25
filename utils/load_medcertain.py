from MedFuse.models.medcertain import MedcertainModule
import jax
from pprint import pprint

def setup_medcertain(*args, rng_key, **kwargs):
    kwargs["exmp_inputs"] = jax.device_put(next(iter(kwargs["train_loader"]))[0])

    medcertain = MedcertainModule(*args, **kwargs)
    
    del kwargs["exmp_inputs"]
    del kwargs["train_loader"]
    del kwargs["val_loader"]
    del kwargs["test_loader"]
    del kwargs["context_loader"]

    pprint(kwargs)
        
    return medcertain, kwargs