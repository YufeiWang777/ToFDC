import glob
import os.path

from dataloaders.utils import *

import torchvision.transforms.functional as TF


glob_dep, glob_gt, glob_K, glob_rgb = None, None, None, None
glob_penettruth, glob_s2dtruth = None, None
get_rgb_paths, get_penettruth_paths, get_s2dtruth_paths, get_K_paths = None, None, None, None

def get_codalabpaths(split, args):

    # data_dir = os.path.join(args.data_folder, 'train')
    # data_list = os.path.join(args.data_folder, 'train', 'data_train.list')

    # sample_list = []
    # with open(data_list, 'r') as f:
    #     lines = f.readlines()
    #     for l in lines:
    #         paths = l.rstrip().split()
    #         sample_list.append(
    #             {'rgb': os.path.join(data_dir, paths[0]), 'depth': os.path.join(data_dir, paths[1])})

    if 'test' in split:
        data_testv1_lists = os.path.join(args.data_folder, '*/data.list')
        data_testv1_pathes = sorted(glob.glob(data_testv1_lists))
        if len(data_testv1_pathes) == 0:
            raise ValueError ('The path to the dataset may be wrong!')
        sample_list_testv1 = []
        for data_testv1_path in data_testv1_pathes:
            with open(data_testv1_path, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    paths = l.rstrip().split()
                    sample_list_testv1.append(
                        {'rgb': data_testv1_path.replace('data.list', paths[0]),
                         'depth': data_testv1_path.replace('data.list', paths[1])})
        sample_list = sample_list_testv1
    else:
        raise ValueError('The code currently only supports testing!')

    # if split == 'train':
    #     sample_list = sample_list[2000:]
    # elif split == 'val':
    #     sample_list = sample_list[:2000]
    # elif 'test' in split:
    #     sample_list = sample_list
    # else:
    #     raise ValueError

    if 'test' in split:
        sample_list = sample_list
    else:
        raise ValueError('The code currently only supports testing!')

    return sample_list

def codalabtransforms(split, args, dep, rgb,
                    bottom_crop=False, hflip=False, colorjitter=False,
                    rotation=False, resize=False, random_crop=False,
                    normalize=False, scale_depth=False, noise_num=0.0, rgb_noise_num=0.0):
    if split == 'train':
        bottom_crop = args.train_bottom_crop
        hflip = args.hflip
        colorjitter = args.colorjitter
        rotation = args.rotation
        resize = args.resize
        random_crop = args.train_random_crop
        normalize = args.normalize
        scale_depth = args.scale_depth
        noise_num = args.noise
        rgb_noise_num = args.rgb_noise
    elif split == 'val':
        resize = args.resize
        normalize = args.normalize
        bottom_crop = args.val_bottom_crop
        random_crop = args.val_random_crop
    elif 'test' in split:
        normalize = args.normalize

    width, height = dep.size

    flip = torch.FloatTensor(1).uniform_(0, 1).item()
    degree = torch.FloatTensor(1).uniform_(-5.0, 5.0).item()
    _scale = torch.FloatTensor(1).uniform_(1.0, 1.5).item()

    if bottom_crop:
        oheight, owidth = args.val_h, args.val_w
        h_init = height - oheight
        w_init = (width - owidth) // 2
        dep = Crop(dep, h_init, w_init, oheight, owidth) if (dep is not None) else None
        rgb = Crop(rgb, h_init, w_init, oheight, owidth) if (rgb is not None) else None

    if colorjitter:
        rgb = ColorJitter(rgb) if (rgb is not None) else None

    if rotation:
        dep = Rotation(dep, degree) if (dep is not None) else None
        rgb = Rotation(rgb, degree) if (rgb is not None) else None

    if resize:
        oheight, owidth = args.val_h, args.val_w
        scale = np.int(oheight)

        dep = Resize(dep, scale, Image.NEAREST) if (dep is not None) else None
        rgb = Resize(rgb, scale, Image.BILINEAR) if (rgb is not None) else None


    if hflip:
        dep = Hflip(dep, flip) if (dep is not None) else None
        rgb = Hflip(rgb, flip) if (rgb is not None) else None


    if random_crop:
        width_, height_ = dep.size

        rwidth, rheight = args.random_crop_width, args.random_crop_height
        h_init = (height_ - rheight) // 2
        w_init = (width_ - rwidth) // 2
        dep = Crop(dep, h_init, w_init, rheight, rwidth) if (dep is not None) else None
        rgb = Crop(rgb, h_init, w_init, rheight, rwidth) if (rgb is not None) else None

    dep = TF.to_tensor(np.array(dep)) if (dep is not None) else None
    rgb = TF.to_tensor(rgb) if (rgb is not None) else None

    if normalize:
        if rgb is not None:
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

    if scale_depth:
        dep = dep / _scale if (dep is not None) else None

    rgb_n = torch.FloatTensor(1).uniform_(0, 1).item()
    if rgb_n > 0.2 and rgb_noise_num > 0:
        rgb_noise = torch.normal(mean=torch.zeros_like(rgb), std=args.rgb_noise * torch.FloatTensor(1).uniform_(0.5, 1.5).item())
        rgb = rgb + rgb_noise

    if noise_num:
        reflection = np.clip(np.random.normal(1, scale=0.333332, size=(1, 1)), 0.01, 3)[0, 0]
        noise = torch.normal(mean=0.0, std=dep * reflection * noise_num)
        dep_noise = dep + noise
        dep_noise[dep_noise < 0] = 0
    else:
        dep_noise = dep.clone()

    if args.grid_spot:
        dep_sp = get_sparse_depth_grid(dep_noise)
    else:
        dep_sp = get_sparse_depth(dep_noise, args.num_sample)

    return dep_sp, dep_noise, rgb


