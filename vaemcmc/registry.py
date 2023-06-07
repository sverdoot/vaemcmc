from torch import nn
from torch.distributions import Distribution
from typing import Any, Dict, Optional, Type

# from vaemcmc.mcmc import MCMCKernel
# from vaemcmc import distributions
# from vaemcmc


class Registry:
    registry: Dict[str, Any] = {}
    model_registry: Dict[str, Type[nn.Module]] = {}
    dist_registry: Dict[str, Type[Distribution]] = {}
    mcmc_registry: Dict[str, Type[object]] = {}
    callback_registry: Dict[str, Type[object]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        def inner_wrapper(wrapped_class):
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper
    
    @classmethod
    def register_model(cls, name: Optional[str] = None):
        def inner_wrapper(wrapped_class: Type[nn.Module]):
            if name is None:
                name_ : str = wrapped_class.__name__
            else:
                name_ = name
            cls.model_registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper
    
    @classmethod
    def register_dist(cls, name: Optional[str] = None):
        def inner_wrapper(wrapped_class):
            if name is None:
                name_ : str = wrapped_class.__name__
            else:
                name_ = name
            cls.dist_registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper
    
    @classmethod
    def register_mcmc(cls, name: Optional[str] = None):
        def inner_wrapper(wrapped_class: Type[object]):
            if name is None:
                name_ : str = wrapped_class.__name__
            else:
                name_ = name
            cls.mcmc_registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs):
        model = cls.registry[name]
        model = model(**kwargs)
        return model
    