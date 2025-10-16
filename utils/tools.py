import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

plt.switch_backend('agg')


def disable_grad(m):
    if isinstance(m, nn.Module):
        for p in m.parameters():
            p.requires_grad = False
    elif isinstance(m, nn.Parameter):
        m.requires_grad = False


def enable_grad(m):
    if isinstance(m, nn.Module):
        for p in m.parameters():
            p.requires_grad = True
    elif isinstance(m, nn.Parameter):
        m.requires_grad = True


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine2":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def adjust_learning_rate_only(epoch, args, init_lr=None):
    if init_lr is None:
        init_lr = args.learning_rate
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: init_lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: init_lr if epoch < 3 else init_lr * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine2":
        lr_adjust = {epoch: init_lr / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == "sigmoid":
        k = 0.5 # logistic growth rate
        s = 10  # decreasing curve smoothing rate
        w = 10  # warm-up coefficient
        lr_adjust = {epoch: args.learning_rate / (1 + np.exp(-k * (epoch - w))) - args.learning_rate / (1 + np.exp(-k/s * (epoch - w*s)))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        print('Updating learning rate to {}'.format(lr))
        return lr
    else:
        return init_lr


class Scheduler:
    def __init__(self, optimizer, args, train_steps, fixed_epoch=None):
        self.optimizer = optimizer
        self.scheduler_type = args.lradj

        self.step_size = args.step_size
        self.lr_decay = args.lr_decay
        self.min_lr = args.min_lr
        self.mode = args.mode
        self.train_epochs = args.train_epochs
        self.train_steps = train_steps
        self.pct_start = args.pct_start
        self.fixed_epoch = 3 if fixed_epoch is None else fixed_epoch

        if self.scheduler_type is None:
            self.scheduler = None

        elif self.scheduler_type == 'reduce':
            _mode = 'min' if self.mode == 0 else 'max'
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=_mode, factor=self.lr_decay, patience=self.step_size, min_lr=self.min_lr)

        elif self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.step_size, eta_min=self.min_lr)

        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.lr_decay)

        elif self.scheduler_type == 'type1':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 1))

        elif self.scheduler_type == 'type2':
            lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            lr_lambda = {epoch: lr / args.learning_rate for epoch, lr in lr_adjust.items()}
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda.get(epoch, 1.0))

        elif self.scheduler_type == 'type3':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 if epoch < self.fixed_epoch else 0.9 ** ((epoch - self.fixed_epoch) // 1))

        elif self.scheduler_type == 'cosine2':
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 + math.cos(epoch / args.train_epochs * math.pi)) / 2)

        elif self.scheduler_type == 'TST':
            max_lr = [g['lr'] for g in optimizer.param_groups]
            if len(max_lr) == 1:
                max_lr = max_lr[0]
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, steps_per_epoch=self.train_steps, epochs=self.train_epochs,
                max_lr=max_lr, pct_start=self.pct_start
            )

        elif self.scheduler_type == 'sigmoid':
            k = 0.5 # logistic growth rate
            s = 10  # decreasing curve smoothing rate
            w = 10
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + np.exp(-k * (epoch - w))) - 1 / (1 + np.exp(-k/s * (epoch - w*s))))

        else:
            raise NotImplementedError

        if self.scheduler is not None:
            self.last_lr = self.scheduler._last_lr[0] if len(self.scheduler._last_lr) == 1 else list(self.scheduler._last_lr)
        else:
            lrs = [g['lr'] for g in optimizer.param_groups]
            self.last_lr = lrs[0] if len(lrs) == 1 else lrs
        print(f'Initial learning rates: {self.last_lr}')

    def get_lr(self):
        return self.last_lr

    def step(self, val_loss=None, epoch=None, verbose=True):
        if self.scheduler_type is None or self.scheduler_type == 'none':
            return
        elif self.scheduler_type == 'reduce':
            self.scheduler.step(val_loss, epoch)
        elif epoch is not None:
            self.scheduler.step(epoch)
        else:
            self.scheduler.step()
        self.lr_info(verbose=verbose)

    def lr_info(self, verbose=True):
        last_lrs = self.scheduler._last_lr[0] if len(self.scheduler._last_lr) == 1 else list(self.scheduler._last_lr)
        if last_lrs != self.last_lr:
            if verbose:
                print(f'Updating learning rate from {self.last_lr} to {last_lrs}')
            self.last_lr = last_lrs


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, **kwargs):
        # 如果val_loss为nan或inf，直接记为一次count
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.counter += 1
            print(f'Validation loss is NaN or Inf. EarlyStopping counter: \033[91m{self.counter} out of {self.patience}\033[0m')
            if self.counter >= self.patience:
                self.early_stop = True
            return False

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, **kwargs)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: \033[91m{self.counter} out of {self.patience}\033[0m')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, **kwargs)
            self.counter = 0
            return True

    def save_checkpoint(self, val_loss, model, path, **kwargs):
        if self.verbose:
            print(f'Validation loss decreased \033[92m({self.val_loss_min:.6f} --> {val_loss:.6f})\033[0m. Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        for key, value in kwargs.items():
            torch.save(value, os.path.join(path, f'{key}.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')


def plot_heatmap(matrix, save_path=None):
    f = plt.figure(dpi=300, figsize=(4, 3))  # 如果只画一条线，就用红色，高改为3->2.5
    f.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.2)
    ax = f.add_subplot(1, 1, 1)
    cax = ax.matshow(matrix, cmap='viridis')
    f.colorbar(cax)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(f)
    else:
        return f


def log_heatmap(writer, matrix, tag, step):
    f = plot_heatmap(matrix)
    writer.add_figure(tag, f, step)
    plt.close(f)


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class EvalAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            values = eval(values)
        except:
            try:
                values = eval(values.lower().capitalize())
            except:
                pass
        setattr(namespace, self.dest, values)


class PParameter(nn.Parameter):
    def __repr__(self):
        tensor_type = str(self.data.type()).split('.')[-1]
        size_str = " x ".join(map(str, self.shape))
        return f"Parameter containing: [{tensor_type} of size {size_str}]"


def ensure_path(path):
    os.makedirs(path, exist_ok=True)


def pv(msg, verbose):
    if verbose:
        print(msg)


def split_dataset_with_overlap(dataset, n, r):
    """
    将 dataset 分成 n 份，每份和下一份重叠 r (0<=r<1) 比例的元素。
    返回 List[Subset]，每个 Subset 可直接喂 DataLoader。
    """
    length = len(dataset)
    if not (0 <= r < 1):
        raise ValueError("r 必须在 [0, 1) 范围内")
    if n <= 0 or length < n:
        raise ValueError("n 参数不合法")

    part_len = int((length + (n - 1) * r) / n)
    if part_len <= 0:
        raise ValueError("每份长度过小")
    overlap_len = int(part_len * r)

    splits = []
    for i in range(n):
        start = i * (part_len - overlap_len)
        end   = start + part_len
        if end > length:
            end = length
        splits.append(Subset(dataset, list(range(start, end))))
    return splits


def split_dataset(dataset, r):
    """
    将 dataset 分成两份，比例为 r:(1-r)。
    返回 List[Subset]，每个 Subset 可直接喂 DataLoader。
    """
    length = len(dataset)
    if not (0 < r < 1):
        raise ValueError("r 必须在 (0, 1) 范围内")

    len1 = int(length * r)
    return Subset(dataset, list(range(0, len1))), Subset(dataset, list(range(len1, length)))


def clip_grads(grads, max_norm):
    valid_grads = [g for g in grads if g is not None]
    if len(valid_grads) == 0:
        return grads
    total_norm = torch.norm(torch.stack([g.norm() for g in valid_grads]))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        return [g * scale if g is not None else None for g in grads]
    return grads
