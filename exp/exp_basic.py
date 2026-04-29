import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import shutil
import time
import torch
import yaml

from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data_provider.data_factory import data_provider
from models import MODEL_DICT
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.metrics_torch import metric_torch
from utils.tools import ensure_path, pv, LocalBufferWriter, BufferSummaryWriter, FoolWriter, visual


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
        self.output_log = args.output_log
        self.report_to = args.report_to

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

        if self.report_to == 'tensorboard':
            writer = SummaryWriter(log_dir)
        elif self.report_to == 'local':
            writer = LocalBufferWriter(log_dir)
        elif self.report_to == 'buffer':
            writer = BufferSummaryWriter(log_dir)
        else:
            writer = FoolWriter(log_dir)
        return writer

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
        elif loss_type == 'mape':
            criterion = mape_loss()
        elif loss_type == 'mase':
            criterion = mase_loss()
        elif loss_type == 'smape':
            criterion = smape_loss()
        else:
            criterion = loss_type
        return criterion

    def forward_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE', 'CFPT']:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        model_args = [batch_x, batch_x_mark, dec_inp, batch_y_mark]
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs, attn = self.model(*model_args)
                else:
                    outputs = self.model(*model_args)
                    attn = None
        else:
            if self.args.output_attention:
                outputs, attn = self.model(*model_args)
            else:
                outputs = self.model(*model_args)
                attn = None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        return outputs, batch_y, attn

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        total_loss = torch.mean(torch.stack(total_loss)).item()  # average loss
        self.model.train()
        return total_loss

    def train(self):
        pass

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        inputs, preds, trues = [], [], []
        if self.output_vis:
            folder_path = os.path.join(self.args.test_results, setting)
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    batch_x = batch_x.cpu().numpy()
                    in_shape = batch_x.shape
                    batch_x = test_data.inverse_transform(batch_x.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(batch_x).float().to(self.device)

                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()
                    out_shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs).float().to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)

                inputs.append(batch_x.cpu())
                preds.append(outputs.cpu())
                trues.append(batch_y.cpu())

                if i % 20 == 0 and self.output_vis:
                    gt = np.concatenate((batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        print('test shape:', preds.shape, trues.shape)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        metrics = OrderedDict(zip(['mae', 'mse', 'rmse', 'mape', 'mspe', 'mre'], [mae, mse, rmse, mape, mspe, mre]))

        extra_metrics = OrderedDict()

        full_metrics = OrderedDict(**metrics, **extra_metrics)
        line = f'{self.args.data_id} @ {self.pred_len}\t| mse:{mse} mae:{mae}'
        if self.args.extra_metrics != []:
            extra_line = ', '.join([f'{k}:{v}' for k, v in extra_metrics.items()])
            line = f'{line}\t| {extra_line}'
        print(line)

        for k, v in full_metrics.items():
            self.writer.add_scalar(f'{self.pred_len}/test/{k}', v, self.epoch)
        self.writer.close()

        if self.output_log:
            log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
            payload = f"{setting}\n\n{line}\n\n"
            with open(log_path, mode="a", encoding="utf-8") as f:
                f.write(payload)

        # np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))
        yaml.safe_dump(dict(full_metrics), open(os.path.join(res_path, 'metrics.yaml'), 'w'), default_flow_style=False, sort_keys=False)

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            yaml.dump(vars(self.args), open(os.path.join(res_path, 'config.yaml'), 'w'), default_flow_style=False)

        return
