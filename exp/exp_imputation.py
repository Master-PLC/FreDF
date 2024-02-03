import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.tools import (EarlyStopping, adjust_learning_rate, ensure_path,
                         visual)

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        self.mask_rate = args.mask_rate

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()
                
                if self.args.reconstruction_type == "imputation":
                    loss = criterion(pred[mask == 0], true[mask == 0])
                elif self.args.reconstruction_type == "autoencoder":
                    loss = criterion(pred[mask == 1], true[mask == 1])
                elif self.args.reconstruction_type == "full":
                    loss = criterion(pred, true)
                else:
                    raise NotImplementedError
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        ensure_path(path)
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                
                loss = 0
                if self.args.rec_lambda:
                    if self.args.reconstruction_type == 'imputation':
                        loss_rec = criterion(outputs[mask == 0], batch_x[mask == 0])
                    elif self.args.reconstruction_type == 'autoencoder':
                        loss_rec = criterion(outputs[mask == 1], batch_x[mask == 1])
                    elif self.args.reconstruction_type == 'full':
                        loss_rec = criterion(outputs, batch_x)
                    else:
                        raise NotImplementedError
                    loss += loss_rec * self.args.rec_lambda
                    if (i + 1) % 100 == 0:
                        print(f"\tloss_rec: {loss_rec.item()}")

                    self.writer.add_scalar(f'{self.mask_rate}/train/loss_rec', loss_rec, self.step)

                if self.args.auxi_lambda:
                    # fft shape: [B, T, N]
                    if self.args.auxi_mode == "fft":
                        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_x, dim=1)

                    elif self.args.auxi_mode == "rfft-fill":
                        filled_outputs = torch.zeros_like(outputs)
                        filled_x = torch.zeros_like(batch_x)
                        
                        if self.args.reconstruction_type == 'autoencoder':
                            filled_outputs[mask == 1] = outputs[mask == 1]
                            filled_x[mask == 1] = batch_x[mask == 1]
                        elif self.args.reconstruction_type == 'imputation':
                            filled_outputs[mask == 0] = outputs[mask == 0]
                            filled_x[mask == 0] = batch_x[mask == 0]
                        else:
                            raise NotImplementedError

                        loss_auxi = torch.fft.rfft(filled_outputs, dim=1) - torch.fft.rfft(filled_x, dim=1)

                    elif self.args.auxi_mode == "rfft":
                        loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_x, dim=1)

                    elif self.args.auxi_mode == "temp":
                        loss_auxi = outputs - batch_x
                    else:
                        raise NotImplementedError

                    if self.args.auxi_loss == "MAE":
                        # MAE, 最小化element-wise error的模长
                        loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                    elif self.args.auxi_loss == "MSE":
                        # MSE, 最小化element-wise error的模长
                        loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                    else:
                        raise NotImplementedError

                    loss += loss_auxi * self.args.auxi_lambda
                    if (i + 1) % 100 == 0:
                        print(f"\tloss_auxi: {loss_auxi.item()}")

                    self.writer.add_scalar(f'{self.mask_rate}/train/loss_auxi', loss_auxi, self.step)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.writer.add_scalar(f'{self.mask_rate}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.mask_rate}/vali/loss', vali_loss, self.epoch)
            self.writer.add_scalar(f'{self.mask_rate}/test/loss', test_loss, self.epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))
            print('loading finished')

        preds, trues, masks = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        ensure_path(folder_path)

        self.model.eval()
        metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                outputs = outputs.detach().contiguous()
                true = batch_x.detach().contiguous()
                mask = mask.detach().contiguous()

                if self.args.reconstruction_type in ["imputation", "full"]:
                    metric_collector.update(outputs[mask == 0], true[mask == 0])
                elif self.args.reconstruction_type == "autoencoder":
                    metric_collector.update(outputs[mask == 1], true[mask == 1])
                else:
                    raise NotImplementedError

                if self.output_vis or self.output_pred:
                    pred = outputs.cpu().numpy()
                    true = true.cpu().numpy()
                    mask = mask.cpu().numpy()

                    preds.append(pred)
                    trues.append(true)
                    masks.append(mask)

                if i % 20 == 0 and self.output_vis:
                    filled = true[0, :, -1].copy()
                    if self.args.reconstruction_type in ["imputation", "full"]:
                        filled = filled * mask[0, :, -1] + pred[0, :, -1] * (1 - mask[0, :, -1])
                    elif self.args.reconstruction_type == "autoencoder":
                        filled = filled * (1 - mask[0, :, -1]) + pred[0, :, -1] * mask[0, :, -1]
                    else:
                        raise NotImplementedError
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.pdf'))

        if self.output_vis or self.output_pred:
            preds = np.concatenate(preds, 0)
            trues = np.concatenate(trues, 0)
            masks = np.concatenate(masks, 0)
            print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        m = metric_collector.compute()
        mae, mse, rmse, mape, mspe = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"]
        # mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        self.writer.add_scalar(f'{self.mask_rate}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.mask_rate}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.mask_rate}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.mask_rate}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.mask_rate}/test/mspe', mspe, self.epoch)
        self.writer.close()

        print('{}\t| mse:{}, mae:{}'.format(self.mask_rate, mse, mae))
        log_path = "result_imputation.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        if self.output_pred:
            np.save(os.path.join(res_path, 'pred.npy'), preds)
            np.save(os.path.join(res_path, 'true.npy'), trues)
            np.save(os.path.join(res_path, 'mask.npy'), trues)
        return
