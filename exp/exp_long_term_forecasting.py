import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import time
import torch
import warnings

import numpy as np

from exp.exp_basic import Exp_Basic
from utils.polynomial import chebyshev_torch, hermite_torch, laguerre_torch, leg_torch
from utils.tools import EarlyStopping, ensure_path, Scheduler

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.pred_len = args.pred_len

        if args.add_noise and args.noise_amp > 0:
            seq_len = args.pred_len
            cutoff_freq_percentage = args.noise_freq_percentage
            cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
            if args.auxi_mode == "rfft":
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1).to(self.device)
        else:
            self.mask = None

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
        scheduler = Scheduler(model_optim, self.args, train_steps)
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []
            rec_loss, auxi_loss = [], []

            lr_cur = scheduler.get_lr()
            lr_cur = lr_cur[0] if isinstance(lr_cur, list) else lr_cur
            self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

                loss = 0
                if self.args.rec_lambda:
                    loss_rec = criterion(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec
                else:
                    loss_rec = torch.tensor(1e4)

                if self.args.auxi_lambda:
                    # fft shape: [B, P, D]
                    if self.args.auxi_mode == "fft":
                        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)

                    elif self.args.auxi_mode == "rfft":
                        if self.args.auxi_type == 'complex':
                            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
                        elif self.args.auxi_type == 'complex-phase':
                            loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                        elif self.args.auxi_type == 'complex-mag-phase':
                            loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                            loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                        elif self.args.auxi_type == 'phase':
                            loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                        elif self.args.auxi_type == 'mag':
                            loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                        elif self.args.auxi_type == 'mag-phase':
                            loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                            loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "rfft-D":
                        loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)

                    elif self.args.auxi_mode == "rfft-2D":
                        loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)
                    
                    elif self.args.auxi_mode == "legendre":
                        loss_auxi = leg_torch(outputs, self.args.leg_degree, device=self.device) - leg_torch(batch_y, self.args.leg_degree, device=self.device)
                    
                    elif self.args.auxi_mode == "chebyshev":
                        loss_auxi = chebyshev_torch(outputs, self.args.leg_degree, device=self.device) - chebyshev_torch(batch_y, self.args.leg_degree, device=self.device)
                    
                    elif self.args.auxi_mode == "hermite":
                        loss_auxi = hermite_torch(outputs, self.args.leg_degree, device=self.device) - hermite_torch(batch_y, self.args.leg_degree, device=self.device)
                    
                    elif self.args.auxi_mode == "laguerre":
                        loss_auxi = laguerre_torch(outputs, self.args.leg_degree, device=self.device) - laguerre_torch(batch_y, self.args.leg_degree, device=self.device)
                    else:
                        raise NotImplementedError

                    if self.mask is not None:
                        loss_auxi *= self.mask

                    if self.args.offload:
                        loss_auxi = loss_auxi.cpu()

                    if self.args.auxi_loss == "MAE":
                        # MAE, 最小化element-wise error的模长
                        loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                    elif self.args.auxi_loss == "MSE":
                        # MSE, 最小化element-wise error的模长
                        loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                    else:
                        raise NotImplementedError

                    if self.args.offload:
                        loss_auxi = loss_auxi.to(self.device)

                    loss += self.args.auxi_lambda * loss_auxi
                else:
                    loss_auxi = torch.tensor(1e4)

                train_loss.append(loss.item())
                rec_loss.append(loss_rec.item())
                auxi_loss.append(loss_auxi.item())
                self.writer.add_scalar(f'{self.pred_len}/train_iter/loss', loss.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_iter/loss_rec', loss_rec.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_iter/loss_auxi', loss_auxi.item(), self.step)

                if (i + 1) % 100 == 0:
                    print("\titers: {}, epoch: {} | loss_rec: {:.7f} | loss_auxi: {:.7f} | loss: {:.7f}".format(i + 1, self.epoch, loss_rec.item(), loss_auxi.item(), loss.item()))
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                model_optim.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            rec_loss = np.average(rec_loss)
            auxi_loss = np.average(auxi_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', rec_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', auxi_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/test/loss', test_loss, self.epoch)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(vali_loss, self.epoch)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
