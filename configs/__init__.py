
from importlib import import_module

def get(arg=None):

    config_name = 'get_cfg_defaults'
    module_name = 'configs.config'
    module = import_module(module_name)
    get_config = getattr(module, config_name)
    cfg = get_config()

    if arg is not None:
        cfg.defrost()
        cfg.merge_from_file('configs/' + arg.configuration)
        cfg.num_gpus = len(cfg.gpus)
        cfg.data_folder = arg.data_folder
        cfg.test_model = arg.test_model
        cfg.test_dir = arg.test_dir
        cfg.freeze()
        args_config = cfg
    else:
        args_config = cfg

    return args_config

