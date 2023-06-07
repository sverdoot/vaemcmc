from typing import Dict, Union
from dataclasses import dataclass
from pathlib import Path
from munch import Munch
from ruamel.yaml import YAML, RoundTripLoader, RoundTripDumper


class Config(Munch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self = self.munch(self)

    @staticmethod  
    def munch(d: Dict):
        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(v, Dict):
                    d[k] = Config.munch(v)
            d = Munch(d)
        return d
    
    @staticmethod
    def dict(d: Munch):
        if hasattr(d, 'items'):
            for k, v in d.items():
                if isinstance(v, Munch):
                    d[k] = Config.dict(v)
            d = dict(d)
        return d
             
    @staticmethod 
    def load(cfg_path: Union[Path, str]):
        yaml = YAML(typ='safe')
        return Config(yaml.load(Path(cfg_path)))
        
    def dump(self, cfg_path: Union[Path, str]):
        d = self.dict(dict(self))
        yaml = YAML(typ='safe')
        yaml.dump(d, Path(cfg_path))
        
            
    