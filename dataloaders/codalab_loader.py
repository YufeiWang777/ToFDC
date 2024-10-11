import torch.utils.data as data

from dataloaders.paths_and_transform import *
from dataloaders.NNfill import *


class CodalabDepth(data.Dataset):
    """A data loader for the CodalabDepth dataset
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        self.paths = get_codalabpaths(split, args)
        self.transforms = codalabtransforms
        if args.fill_type == 'NN_fillv1':
            self.fill = NN_fillv1
        elif args.fill_type == 'NN_fillv2':
            self.fill = NN_fillv2
        else:
            raise ValueError('fill type error')


    def __getraw__(self, index):

        rgb = Image.open(self.paths[index]['rgb']).convert('RGB')
        dep = cv2.imread(self.paths[index]['depth'], cv2.IMREAD_ANYDEPTH)
        dep = dep.astype(np.float32)
        dep[dep>self.args.max_depth] = 0
        dep = Image.fromarray(dep)

        return dep, rgb, self.paths[index]['rgb']

    def __getitem__(self, index):

        dep, rgb, paths = self.__getraw__(index)
        dep_sp, dep, rgb = self.transforms(self.split, self.args, dep, rgb)

        # NNI fill
        dep_np = dep.numpy().squeeze(0)
        dep_sp_np = dep_sp.numpy().squeeze(0)
        dep_np_nn = np.copy(dep_np)

        if 'test' not in self.split:
            mask = dep_sp_np > 0
            S1, _ = NN_fillv2(dep_np_nn, mask)
        else:
            mask = dep_np > 0
            S1, _ = NN_fillv2(dep_np_nn, mask)

        if 'test' in self.split and self.args.crop_and_pad:
            S1 = crop_and_pad(S1.squeeze(0), dep_np, crop_orientation=self.args.crop_orientation,
                              softspace=self.args.softspace, mode=self.args.mode)
            S1 = S1[np.newaxis, :]

        S1_torch = torch.from_numpy(S1)
        S1_torch = S1_torch.to(dtype=torch.float32)

        if self.args.cut_mask and self.split=='train':
            dep_sp = cut_mask(dep_sp)

        if 'test' not in self.split:
            candidates = {'dep':dep_sp, 'gt':dep, 'rgb':rgb, 's1':S1_torch}
        else:
            candidates = {'dep': dep, 'gt': dep, 'rgb': rgb, 's1':S1_torch}

        items = {
            key: val
            for key, val in candidates.items() if val is not None
        }
        if self.args.debug_dp or self.args.test:
            items['d_path'] = paths

        return items

    def __len__(self):
        if self.args.toy_test:
            return self.args.toy_test_number
        else:
            return len(self.paths)


if __name__ == '__main__':
    import argparse
    arg = argparse.ArgumentParser(description='depth completion')
    arg.add_argument('-p', '--project_name', type=str, default='Codalab')
    arg.add_argument('-n', '--model_name', type=str, default='test')
    arg.add_argument('-o', '--opt_level', type=str, default='O0')
    arg.add_argument('-c', '--configuration', type=str, default='train.yml')
    arg = arg.parse_args()
    from configs import get as get_cfg
    config = get_cfg(arg)

    data_train = CodalabDepth('testv1', config)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=2,
                                               shuffle=False, num_workers=10,
                                               pin_memory=False, drop_last=True)

    i = 0
    for batch, sample in enumerate(loader_train):
        # print(i)
        i += 1

