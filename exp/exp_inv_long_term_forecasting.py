import os
import time
import warnings

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import metric
from utils.tools import (EarlyStopping, adjust_learning_rate, ensure_path,
                         visual)

warnings.filterwarnings('ignore')


class Exp_Inv_Long_Term_Forecast(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Inv_Long_Term_Forecast, self).__init__(args)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -(self.pred_len + 2):, f_dim:]
                real, img = torch.chunk(outputs, dim=1, chunks=2)
                outputs_freq = torch.complex(real, img)
                outputs_temp = torch.fft.irfft(outputs_freq, dim=1)
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                pred = outputs_temp.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

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

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

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
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -(self.pred_len + 2):, f_dim:]
                        real, img = torch.chunk(outputs, dim=1, chunks=2)
                        outputs_freq = torch.complex(real, img)
                        outputs_temp = torch.fft.irfft(outputs_freq, dim=1)
                        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs_temp, batch_y)

                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        # outputs shape: [B, P, D]
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -(self.pred_len + 2):, f_dim:]
                    real, img = torch.chunk(outputs, dim=1, chunks=2)
                    outputs_freq = torch.complex(real, img)
                    outputs_temp = torch.fft.irfft(outputs_freq, dim=1)
                    batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                    loss = 0
                    if self.args.rec_lambda:
                        loss_rec = criterion(outputs_temp, batch_y)
                        loss += self.args.rec_lambda * loss_rec

                        self.writer.add_scalar(f'{self._pred_len}/train/loss_rec', loss_rec, self.step)

                    if self.args.auxi_lambda:
                        # fft shape: [B, P, D]
                        if self.args.auxi_mode == "rfft":
                            loss_auxi = outputs_freq - torch.fft.rfft(batch_y, dim=1)
                        else:
                            raise NotImplementedError

                        if self.mask is not None:
                            loss_auxi *= self.mask

                        if self.args.auxi_loss == "MAE":
                            # MAE, 最小化element-wise error的模长
                            loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                        elif self.args.auxi_loss == "MSE":
                            # MSE, 最小化element-wise error的模长
                            loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                        else:
                            raise NotImplementedError

                        loss += self.args.auxi_lambda * loss_auxi

                        self.writer.add_scalar(f'{self._pred_len}/train/loss_auxi', loss_auxi, self.step)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.writer.add_scalar(f'{self._pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self._pred_len}/vali/loss', vali_loss, self.epoch)
            self.writer.add_scalar(f'{self._pred_len}/test/loss', test_loss, self.epoch)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
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

        preds = []
        trues = []
        folder_path = os.path.join(self.args.test_results, setting)
        ensure_path(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros(batch_y.shape[0], self.args.pred_len ,batch_y.shape[-1]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -(self.pred_len + 2):, :]
                real, img = torch.chunk(outputs, dim=1, chunks=2)
                outputs_freq = torch.complex(real, img)
                outputs_temp = torch.fft.irfft(outputs_freq, dim=1)
                batch_y = batch_y[:, -self.pred_len:, :].to(self.device)
                outputs = outputs_temp.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        ensure_path(res_path)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.writer.add_scalar(f'{self._pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self._pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self._pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self._pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self._pred_len}/test/mspe', mspe, self.epoch)
        self.writer.close()

        print('mse:{}, mae:{}'.format(mse, mae))
        log_path = "result_inv_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))

        if self.args.output_pred:
            np.save(os.path.join(res_path, 'pred.npy'), preds)
            np.save(os.path.join(res_path, 'true.npy'), trues)

        return
