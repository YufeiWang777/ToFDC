# CONFIG
import argparse
arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-d', '--data_folder', type=str, default='./testv2')
arg.add_argument('-m', '--test_model', type=str, default='./pretrained_model/best_mae_model.pt')
arg.add_argument('-o', '--test_dir', type=str, default='./')
arg.add_argument('-c', '--configuration', type=str, default='val_with_self_ensemble.yml')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)

# ENVIRONMENT SETTINGS
import os
rootPath = os.path.abspath(os.path.dirname(__file__))
import functools
if len(config.gpus) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus[0])
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = functools.reduce(lambda x, y: str(x) + ',' + str(y), config.gpus)
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# BASIC PACKAGES
import emoji
import time
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

# MODULES
from dataloaders.codalab_loader import CodalabDepth as KittiDepth
from model import get as get_model
from summary import get as get_summary
from metric import get as get_metric
from utility import *

# VARIANCES
sample_, output_ = None, None
metric_txt_dir = None

# MINIMIZE RANDOMNESS
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)


def test(args):

    # DATASET
    print(emoji.emojize('Prepare data... :writing_hand:', variant="emoji_type"), end=' ')
    global sample_, output_, metric_txt_dir
    data_test = KittiDepth(args.test_option, args)
    loader_test = DataLoader(dataset=data_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
    print('Done!')

    # NETWORK
    print(emoji.emojize('Prepare model... :writing_hand:', variant="emoji_type"), end=' ')
    model = get_model(args)
    net = model(args)
    net.cuda()
    print('Done!')
    # total_params = count_parameters(net)

    # METRIC
    print(emoji.emojize('Prepare metric... :writing_hand:', variant="emoji_type"), end=' ')
    metric = get_metric(args)
    metric = metric(args)
    print('Done!')

    # SUMMARY
    print(emoji.emojize('Prepare summary... :writing_hand:', variant="emoji_type"), end=' ')
    summary = get_summary(args)
    try:
        if not os.path.isdir(args.test_dir):
            os.makedirs(args.test_dir)
        os.makedirs(args.test_dir, exist_ok=True)
        os.makedirs(args.test_dir + '/output', exist_ok=True)
        metric_txt_dir = os.path.join(args.test_dir + '/output/result_metric.txt')
        with open(metric_txt_dir, 'w') as f:
            f.write('test_model: {} \ntest_option: {} \nval:{} \ntest_name: {} \n'
                    'test_not_random_crop: {} \n'
                    'tta: {}\n \n'.format(args.test_model, args.test_option, 'val', args.test_name,
                                       args.test_not_random_crop,
                                       args.tta))
    except OSError:
        pass
    writer_test = summary(args.test_dir, 'test', args, None, metric.metric_name)
    print('Done!')

    # LOAD MODEL
    print(emoji.emojize('Load model... :writing_hand:', variant="emoji_type"), end=' ')
    if len(args.test_model) != 0:
        assert os.path.exists(args.test_model), \
            "file not found: {}".format(args.test_model)

        checkpoint_ = torch.load(args.test_model)
        model = remove_moudle(checkpoint_['net'])
        key_m, key_u = net.load_state_dict(model, strict=True)

        if key_u:
            print('Unexpected keys :')
            print(key_u)
            raise KeyError

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError

    net = nn.DataParallel(net)
    net.eval()
    print('Done!')

    num_sample = len(loader_test) * loader_test.batch_size
    pbar_ = tqdm(total=num_sample)
    t_total = 0
    with torch.no_grad():
        for batch_, sample_ in enumerate(loader_test):

            torch.cuda.synchronize()
            t0 = time.time()
            if args.tta:

                samplep = {key: val.float().cuda() for key, val in sample_.items()
                           if torch.is_tensor(val)}
                samplep['d_path'] = sample_['d_path']

                dep = sample_['dep']
                rgb = sample_['rgb']
                s1 = sample_['s1']

                depf = torch.flip(sample_['dep'], [-1])
                rgbf = torch.flip(sample_['rgb'], [-1])
                s1f = torch.flip(sample_['s1'], [-1])

                deps = torch.cat((dep, depf), dim=0)
                rgbs = torch.cat((rgb, rgbf), dim=0)
                s1s = torch.cat((s1, s1f), dim=0)

                sample = {'rgb': rgbs, 'dep':deps, 's1':s1s}
                sample = {key: val.float().cuda() for key, val in sample.items()
                           if torch.is_tensor(val)}

                outputp = net(sample)
                predp = outputp[args.test_name]

                depth_pred, depth_predf = predp.split(predp.shape[0] // 2)
                depth_predf = torch.flip(depth_predf, [-1])
                depth_pred = (depth_pred + depth_predf) / 2.

                output_ = {args.test_name: depth_pred}

            else:
                # sample_['rgb'] = sample_['rgb'].type(torch.FloatTensor)
                samplep = {key: val.float().cuda() for key, val in sample_.items()
                           if torch.is_tensor(val)}
                samplep['d_path'] = sample_['d_path']
                output_ = net(samplep)

            torch.cuda.synchronize()
            t1 = time.time()
            t_total += (t1 - t0)
            if 'test' not in args.test_option:
                metric_test = metric.evaluate(output_[args.test_name], samplep['gt'], 'test')
            else:
                metric_test = metric.evaluate(output_[args.test_name], samplep['dep'], 'test')

            depth_validpoint_number = count_validpoint(samplep['dep'])
            # ben_mask
            depth_validpoint_number_clear = count_validpoint(samplep['dep'])
            with open(metric_txt_dir, 'a') as f:
                f.write('{}; RMSE:{}; MAE:{}; vp_pre:{}; vp_post:{}\n'.format(samplep['d_path'][0].split('/')[-1],
                                                                         metric_test.data.cpu().numpy()[0, 0]*1000,
                                                                         metric_test.data.cpu().numpy()[0, 1]*1000,
                                                                         depth_validpoint_number,
                                                                         depth_validpoint_number_clear))

            writer_test.add(None, metric_test)
            if args.save_test_image:
                writer_test.save(args.epochs, batch_, samplep, output_)

            current_time = time.strftime('%y%m%d@%H:%M:%S')
            error_str = '{} | Test'.format(current_time)
            pbar_.set_description(error_str)
            pbar_.update(loader_test.batch_size)

    pbar_.close()
    _ = writer_test.update(args.epochs, samplep, output_,
                           online_loss=False, online_metric=False, online_rmse_only=False, online_img=False)
    t_avg = t_total / num_sample
    with open(metric_txt_dir, 'a') as f:
        f.write('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))

    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))


if __name__ == '__main__':
    test(config)
