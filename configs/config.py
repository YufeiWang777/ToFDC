from yacs.config import CfgNode as CN

cfg = CN()

# TRAINING SETTINGS
cfg.debug_dp = False
cfg.debug_loss_txt = ''

# Hardware
cfg.seed = 7240
cfg.gpus = (8, )
cfg.port = 29000
cfg.num_threads = 0
cfg.no_multiprocessing = False
cfg.syncbn = False
cfg.cudnn_deterministic = False
cfg.cudnn_benchmark = True

# Dataset
cfg.data_folder = ''
cfg.dataset = ['dep', 'gt', 'rgb']
cfg.grid_spot = True
cfg.cut_mask = False
cfg.max_depth = 80.0
cfg.num_sample = 0
cfg.rgb_noise = 0.05
cfg.noise = 0.01

cfg.fill_type = ''

cfg.toy_test = True
cfg.toy_test_number = 30

cfg.hflip = False
cfg.colorjitter = False
cfg.rotation = False
cfg.resize = False
cfg.normalize = False
cfg.scale_depth = False
cfg.rgb_dropout = False

cfg.val_h = 352
cfg.val_w = 1216
cfg.crop_and_pad = True
cfg.crop_orientation = 'xy'
cfg.softspace = 10
cfg.mode = 'mean'
cfg.random_crop_height = 320
cfg.random_crop_width = 1216
cfg.train_bottom_crop = False
cfg.train_random_crop = False
cfg.val_bottom_crop = False
cfg.val_random_crop = False
cfg.test_bottom_crop = False
cfg.test_random_crop = False

cfg.val_epoch = 10
cfg.val_iters = 500

# Network
cfg.model = ''
cfg.depth_norm = True
cfg.kernel_size = 3
cfg.filter_size = 15
cfg.dkn_residual = True
cfg.preserve_input = True
cfg.activate = 'relu'
cfg.summary_name = ''


cfg.loss_weight = 0.0
cfg.output = ''
cfg.supervised = ''


# Resume
cfg.resume = False
cfg.load_model_strict = True
cfg.selected_layers = []
cfg.pretrain = ''
cfg.wandb_id_resume = ''

# Test
cfg.test = False
cfg.test_option = ''
cfg.test_name = ''
cfg.tta = True
cfg.test_not_random_crop = False
cfg.wandb_id_test = ''
cfg.test_dir = ''
cfg.test_model = ''
cfg.save_test_image = False

# Training
cfg.log_itr = 1
cfg.start_epoch = 0
cfg.epochs = 0
cfg.batch_size = 0

cfg.accumulation_gradient = False
cfg.accumulation_steps = 0

# warm_up
cfg.warm_up = False
cfg.no_warm_up = True

# Loss
cfg.loss_fixed = True
cfg.partial_supervised_index = 0.0
cfg.loss_ben = ''
cfg.loss_jin = ''
cfg.loss_an = ''

# Optimizer
cfg.lr = 0.01
cfg.optimizer = 'ADAM'
# * ADAM
cfg.momentum = 0.9
cfg.betas = (0.9, 0.999)
cfg.epsilon = 1e-8
cfg.weight_decay = 0.0
cfg.scheduler = 'stepLR'    # choices:(stepLR, lambdaLR)
# * lambdaLR
cfg.decay = (7, )
cfg.gamma = (7, )
# * stepLR
cfg.decay_step = 3
cfg.decay_factor = 0.1

# Logs
cfg.vis_step = 10
cfg.num_summary = 4
cfg.record_by_wandb_online = True
cfg.test_record_by_wandb_online = True

cfg.ben_online_loss=True
cfg.ben_online_metric=True
cfg.ben_online_rmse_only=True
cfg.ben_online_img=True

cfg.save_result_only = True


def get_cfg_defaults():
    """
    :return: global local has an error (2020.12.30)
    """
    return cfg.clone()


if __name__ == '__main__':
    my_cfg = get_cfg_defaults()
    print(my_cfg)
