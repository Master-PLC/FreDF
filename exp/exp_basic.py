import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from data_provider.data_factory import data_provider
from torch.utils.tensorboard import SummaryWriter

from models import MODEL_DICT
from utils.tools import ensure_path, pv


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = MODEL_DICT
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        pv(self.model, args.verbose)
        self.writer = None

        self.epoch = 0
        self.step = 0

        self.output_pred = args.output_pred
        self.output_vis = args.output_vis

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        pretrain_model_path = self.args.pretrain_model_path
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f'Loading pretrained model from {pretrain_model_path}')
            state_dict = torch.load(pretrain_model_path)
            model.load_state_dict(state_dict, strict=False)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) \
                if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _create_writer(self, log_dir):
        item_list = os.listdir(log_dir)
        item_path_list = [os.path.join(log_dir, item) for item in item_list]
        item_path_list = [item_path for item_path in item_path_list if os.path.isfile(item_path)]
        if len(item_path_list) > 0:
            pre_log_dir = os.path.join(log_dir, "pre_logs")
            ensure_path(pre_log_dir)

            item_list = [os.path.basename(item_path) for item_path in item_path_list]
            for item, item_path in zip(item_list, item_path_list):
                shutil.move(item_path, os.path.join(pre_log_dir, item))

        return SummaryWriter(log_dir)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, model=None, lr=None, optim_type=None):
        if model is None:
            model = self.model
        if lr is None:
            lr = self.args.learning_rate
        if optim_type is None:
            optim_type = self.args.optim_type
        if optim_type == 'adam':
            optim_class = optim.Adam
        elif optim_type == 'adamw':
            optim_class = optim.AdamW
        elif optim_type == 'sgd':
            optim_class = optim.SGD
        model_optim = optim_class(model.parameters(), lr=lr)
        return model_optim

    def _select_criterion(self, loss_type=None):
        loss_type = loss_type or self.args.loss
        loss_type = loss_type.lower()
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type == 'mae':
            criterion = nn.L1Loss()
        elif loss_type == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = loss_type
        return criterion

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
