from . import BaseSummary
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
# import wandb
import cv2
# from PIL import Image

cm = plt.get_cmap('plasma')
log_metric_val = None
log_metric_val_rmse, log_metric_val_mae = None, None


class Summary(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):

        super(Summary, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def add(self, loss=None, metric=None, log_itr=None):
        # loss and metric should be numpy arrays
        if loss is not None:
            self.loss.append(loss.data.cpu().numpy())
        if metric is not None:
            self.metric.append(metric.data.cpu().numpy())

        if 'train' in self.mode and log_itr % self.args.vis_step == 0:
            log_dict = {}
            for idx, loss_type in enumerate(self.loss_name):
                val = loss.data.cpu().numpy()[0, idx]
                log_dict[self.mode + '_all_' + loss_type] = val

                # log by tb
                self.add_scalar('All/' + loss_type, val, log_itr)
            log_dict['custom_step_loss'] = log_itr
            # wandb.log(log_dict)

    def update(self, global_step, sample, output,
               online_loss=True, online_metric=True, online_rmse_only=True, online_img=True):
        """
        update results
        """
        global log_metric_val, log_metric_val_rmse, log_metric_val_mae
        log_dict = {}
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format(self.mode + '_Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                if online_loss:
                    log_dict[self.mode + '_' + loss_type] = val
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format(self.mode + '_Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                if online_metric:
                    if online_rmse_only:
                        if name == 'RMSE':
                            log_metric_val = val
                            log_dict[self.mode + '_' + name] = val
                        else:
                            pass
                    else:
                        if name == 'RMSE':
                            log_metric_val_rmse = val
                        elif name == "MAE":
                            log_metric_val_mae = val
                        log_dict[self.mode + '_' + name] = val
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

            if self.args.test:
                f_metric = open(os.path.join(self.args.test_dir + '/output/result_metric.txt'), 'a')
                f_metric.write('\n{:04d} | {}\n'.format(global_step, msg))
                f_metric.close()

        # if 'train' in self.mode:
        #     log_dict['custom_step_train'] = global_step
        # elif 'val' in self.mode:
        #     log_dict['custom_step_val'] = global_step
        # elif 'test' in self.mode:
        #     log_dict['custom_step_test'] = global_step
        #
        # # Log by wandb
        # if len(log_dict) != 0 and 'test' not in self.mode:
        #         wandb.log(log_dict)

        # Reset
        self.loss = []
        self.metric = []

        return log_metric_val_rmse, log_metric_val_mae

    def save(self, epoch, idx, sample, output):
        with torch.no_grad():
            if self.args.save_result_only:
                if not self.args.test:
                    self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                                                              'result_pred', epoch)
                else:
                    self.path_output = '{}/{}/{}'.format(self.log_dir,
                                                                  'output', 'depth_pred')

                test_lists = ['iPhone_static', 'iPhone_dynamic', 'modified_phone_static', 'synthetic']
                if idx == 0:
                    for test_list in test_lists:
                        path_output = os.path.join(self.path_output, test_list)
                        os.makedirs(path_output, exist_ok=True)

                        with open(os.path.join(path_output, 'data.list'), 'w') as f:
                            pass

                for test_list in test_lists:
                    if test_list in sample['d_path'][0]:

                        path_output = os.path.join(self.path_output, test_list)
                        path_save_pred = '{}/{}.exr'.format(path_output, idx)

                        pred = output[self.args.output].detach()
                        pred = torch.clamp(pred, min=1e-8, max=self.args.max_depth)
                        pred = pred[0, 0, :, :].data.cpu().numpy()

                        cv2.imwrite(path_save_pred, pred.astype(np.float32))

                        with open(os.path.join(path_output, 'data.list'), 'a') as f:
                            f.write('{}.exr\n'.format(idx))

            else:
                print('No output results are stored!')





