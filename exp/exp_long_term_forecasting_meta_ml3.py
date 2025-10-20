import os
import time
import warnings
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import yaml
from exp.exp_basic import Exp_Basic
from functorch import make_functional
from torch.utils.data import DataLoader
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.tools import EarlyStopping, Scheduler, clip_grads, disable_grad, enable_grad, log_weight, \
    split_dataset, split_dataset_with_overlap, visual

warnings.filterwarnings('ignore')


class ErrorWeighting(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len
        self.eps = 1e-6
        self.auxi_loss = args.auxi_loss
        assert args.module_first
        assert args.auxi_mode == 'rfft'
        assert args.auxi_type == 'complex'

        self.rec_lambda = args.rec_lambda
        self.auxi_lambda = args.auxi_lambda

        if self.rec_lambda:
            self.rec_weights = nn.Parameter(torch.ones(args.pred_len))
        if self.auxi_lambda:
            self.auxi_weights = nn.Parameter(torch.ones(args.pred_len // 2 + 1))

    def forward(self, pred, target, params=None):
        """ML3 Learned Loss Function"""

        loss = 0
        if self.rec_lambda:
            error = (pred - target) ** 2  # [B, P, D]
            weights = params['rec_weights'] if params else self.rec_weights
            weights = torch.softmax(weights, dim=0) * self.pred_len
            loss_rec = (error * weights.view(1, -1, 1)).mean()
            loss += loss_rec
        else:
            loss_rec = 1000

        if self.auxi_lambda:
            error = torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)  # [B, P//2+1, D]
            weights = params['auxi_weights'] if params else self.auxi_weights
            weights = torch.softmax(weights, dim=0) * (self.pred_len // 2 + 1)
            if self.args.offload:
                error = error.cpu()
                weights = weights.cpu()

            if self.auxi_loss == 'MSE':
                loss_auxi = (error.abs()**2 * weights.view(1, -1, 1)).mean()
            elif self.auxi_loss == 'MAE':
                loss_auxi = (error.abs() * weights.view(1, -1, 1)).mean()

            if self.args.offload:
                loss_auxi = loss_auxi.to(pred.device)
                weights = weights.to(pred.device)

            loss += loss_auxi
        else:
            loss_auxi = 1000

        return loss, loss_rec, loss_auxi


def get_weights(ew, name='rec_weights'):
    with torch.no_grad():
        weights = ew.state_dict()[name]
        weights = torch.softmax(weights, dim=0)
    return weights.detach().cpu().numpy()


def get_param_dict(module, params=None):
    if params:
        return {pair[0]: p for pair, p in zip(module.named_parameters(), params)}
    else:
        return dict(module.named_parameters())


def update_param_dict(param_dict, grads_dict, lr):
    """
    使用梯度字典更新参数字典。
    grads_dict 仅包含有梯度的参数，其他参数视为 grad=0（保持不变）。
    """
    updated_params = {}
    for k, v in param_dict.items():
        if k in grads_dict and grads_dict[k] is not None:
            updated_params[k] = v - lr * grads_dict[k]
        else:
            # 该参数无梯度 → 视为梯度为 0 → 不更新
            updated_params[k] = v  # 保持原参数，等价于 +0
    return updated_params


class Exp_Long_Term_Forecast_META_ML3(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len
        self.n_inner = args.meta_inner_steps
        self.lr = args.learning_rate
        self.inner_lr = args.inner_lr
        self.meta_lr = args.meta_lr
        self.first_order = args.first_order
        self.model_per_task = args.model_per_task
        self.num_tasks = args.num_tasks

        self.error_weighting = ErrorWeighting(self.args).to(self.device)
        self.model_func, _ = make_functional(self.model)
        self.task_models = [self.model]
        if self.model_per_task:
            for _ in range(1, self.num_tasks):
                task_model = self._build_model().to(self.device)
                self.task_models.append(task_model)
        else:
            self.task_models = [self.model] * self.num_tasks

    def forward_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark, params=None, model=None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
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
        if params is None and model is None:
            if self.args.output_attention:
                outputs, attn = self.model(*model_args)
            else:
                outputs, attn = self.model(*model_args), None
        else:
            # model_args = tuple(model_args)
            if self.args.output_attention:
                outputs, attn = model(params, *model_args)
            else:
                outputs, attn = model(params, *model_args), None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        return outputs, batch_y, attn

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_tmp_loss, total_feq_loss = [], []
        total_rec_loss, total_auxi_loss = [], []

        self.model.eval()
        self.error_weighting.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach()
                true = batch_y.detach()

                loss_tmp = criterion(pred, true)  # 标准损失
                loss_feq = torch.fft.rfft(pred, dim=1) - torch.fft.rfft(true, dim=1)
                loss_feq = loss_feq.cpu()
                loss_feq = (loss_feq.abs()**2).mean() if self.args.auxi_loss == 'MSE' else loss_feq.abs().mean()

                loss, loss_rec, loss_auxi = self.error_weighting(pred, true)  # 学习到的损失

                total_loss.append(loss)
                total_tmp_loss.append(loss_tmp); total_feq_loss.append(loss_feq)
                total_rec_loss.append(loss_rec); total_auxi_loss.append(loss_auxi)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        total_loss = torch.stack(total_loss).mean().item()
        total_tmp_loss = torch.stack(total_tmp_loss).mean().item()
        total_feq_loss = torch.stack(total_feq_loss).mean().item()
        total_rec_loss = torch.stack(total_rec_loss).mean().item()
        total_auxi_loss = torch.stack(total_auxi_loss).mean().item()

        self.model.train()
        self.error_weighting.train()
        return total_loss, total_tmp_loss, total_feq_loss, total_rec_loss, total_auxi_loss

    def inner_loop(self, task_id, support_loader, query_loader):
        # 获取当前模型参数（每个meta epoch都从当前状态开始，而不是初始状态）
        task_model = self.task_models[task_id]
        model_func, _ = make_functional(task_model)
        model_params_init = get_param_dict(task_model)

        # 内层循环：使用学习到的损失函数训练模型参数
        fast_model_params = {k: v.clone() for k, v in model_params_init.items()}
        for k in range(self.n_inner):
            bx, by, bx_mark, by_mark = next(support_loader)
            outputs, batch_y, _ = self.forward_step(
                bx, by, bx_mark, by_mark, fast_model_params.values(), model_func
            )
            loss, loss_rec, loss_auxi = self.error_weighting(outputs, batch_y)

            model_grads = torch.autograd.grad(
                loss, fast_model_params.values(), 
                create_graph=not self.first_order, 
                allow_unused=True
            )
            model_grads = clip_grads(model_grads, self.args.max_norm)
            model_grads_dict = {k: g for k, g in zip(fast_model_params.keys(), model_grads)}
            fast_model_params = update_param_dict(fast_model_params, model_grads_dict, self.inner_lr)

        # 外层循环：在query set上使用标准损失评估性能
        bx, by, bx_mark, by_mark = next(query_loader)
        outputs, batch_y, _ = self.forward_step(
            bx, by, bx_mark, by_mark, fast_model_params.values(), model_func
        )
        meta_loss, meta_rec_loss, meta_auxi_loss = self.error_weighting(outputs, batch_y)
        return meta_loss, meta_rec_loss, meta_auxi_loss

    def initialize_meta_tasks(self, train_data):
        task_data_list = split_dataset_with_overlap(train_data, self.num_tasks, self.args.overlap_ratio)
        task_data_list = [split_dataset(task_data, r=0.7) for task_data in task_data_list]

        support_data_list = [td[0] for td in task_data_list]
        support_loader_list = [DataLoader(support_data, batch_size=self.args.auxi_batch_size, shuffle=True) for support_data in support_data_list]
        support_loader_list = [cycle(support_loader) for support_loader in support_loader_list]

        query_data_list = [td[1] for td in task_data_list]
        query_loader_list = [DataLoader(query_data, batch_size=self.args.auxi_batch_size, shuffle=True) for query_data in query_data_list]
        query_loader_list = [cycle(query_loader) for query_loader in query_loader_list]
        return support_loader_list, query_loader_list

    def meta_train(self, support_loader_list, query_loader_list, path, res_path):
        # 在meta train阶段，损失函数参数可训练，模型参数也需要梯度（用于inner loop）
        enable_grad(self.error_weighting)
        enable_grad(self.model)

        ew_optim = self._select_optimizer(self.error_weighting, self.meta_lr, optim_type=self.args.meta_optim_type)
        ew_scheduler = Scheduler(ew_optim, self.args, self.args.warmup_steps)

        epoch_time = time.time()
        meta_step = 0
        for step in range(self.args.warmup_steps):
            meta_step = step + 1
            verbose = (meta_step % 100 == 0)
            task_losses = []
            task_loss_rec, task_loss_auxi = [], []

            meta_lr_cur = ew_scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_lr', meta_lr_cur, meta_step)

            self.model.train()
            self.error_weighting.train()

            # 遍历所有任务，累积meta loss
            for task_id, (support_loader, query_loader) in enumerate(zip(support_loader_list, query_loader_list)):
                meta_loss, meta_loss_rec, meta_loss_auxi = self.inner_loop(task_id, support_loader, query_loader)
                task_losses.append(meta_loss)
                task_loss_rec.append(meta_loss_rec)
                task_loss_auxi.append(meta_loss_auxi)

                self.writer.add_scalar(f'{self.pred_len}/meta_train/task_{task_id+1}_meta_loss', meta_loss.item(), meta_step)
                self.writer.add_scalar(f'{self.pred_len}/meta_train/task_{task_id+1}_meta_loss_rec', meta_loss_rec.item(), meta_step)
                self.writer.add_scalar(f'{self.pred_len}/meta_train/task_{task_id+1}_meta_loss_auxi', meta_loss_auxi.item(), meta_step)

                if verbose:
                    print(f"\ttask: {task_id + 1}/{self.num_tasks} | meta loss: {meta_loss.item():.7f}, "
                          f"meta loss_rec: {meta_loss_rec.item():.7f}, meta loss_auxi: {meta_loss_auxi.item():.7f}")

            # 统一进行损失函数参数的更新
            ew_optim.zero_grad()
            avg_meta_loss = torch.stack(task_losses).mean()
            avg_meta_loss.backward()
            ew_optim.step()

            avg_meta_loss_val = avg_meta_loss.item()
            avg_meta_loss_rec = torch.stack(task_loss_rec).mean().item()
            avg_meta_loss_auxi = torch.stack(task_loss_auxi).mean().item()

            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_loss', avg_meta_loss_val, meta_step)
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_loss_rec', avg_meta_loss_rec, meta_step)
            self.writer.add_scalar(f'{self.pred_len}/meta_train/meta_loss_auxi', avg_meta_loss_auxi, meta_step)
            if self.args.rec_lambda:
                log_weight(self.writer, get_weights(self.error_weighting, 'rec_weights'), f'{self.pred_len}/rec_weights', meta_step)
            if self.args.auxi_lambda:
                log_weight(self.writer, get_weights(self.error_weighting, 'auxi_weights'), f'{self.pred_len}/auxi_weights', meta_step)

            if verbose:
                print(f"Step: {meta_step} cost time: {time.time() - epoch_time:.2f}s")
                print(f"Step: {meta_step} | Avg Meta Loss: {avg_meta_loss_val:.7f}, "
                      f"Avg Meta Loss Rec: {avg_meta_loss_rec:.7f}, Avg Meta Loss Auxi: {avg_meta_loss_auxi:.7f}")
                epoch_time = time.time()

            if self.args.lradj in ['TST']:
                ew_scheduler.step(verbose=verbose)
            else:
                if verbose:
                    ew_scheduler.step(avg_meta_loss_val, meta_step // 100)

        best_ew_path = os.path.join(path, 'error_weighting.pth')
        torch.save(self.error_weighting, best_ew_path)

    def meta_test(self, train_loader, vali_data, vali_loader, criterion, path):
        if self.model_per_task and self.num_tasks > 1:
            del self.task_models[1:]

        disable_grad(self.error_weighting)
        enable_grad(self.model)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer(self.model, self.lr)
        scheduler = Scheduler(model_optim, self.args, train_steps)

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []
            train_loss_tmp, train_loss_feq = [], []
            train_loss_rec, train_loss_auxi = [], []

            lr_cur = scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/meta_test/lr', lr_cur, self.epoch)

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.model.train()
                self.error_weighting.eval()

                self.step += 1
                iter_count += 1

                model_optim.zero_grad()
                outputs, batch_y, _ = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss, loss_rec, loss_auxi = self.error_weighting(outputs, batch_y)
                loss.backward()
                model_optim.step()

                with torch.no_grad():
                    loss_tmp = criterion(outputs, batch_y)
                    loss_feq = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
                    loss_feq = loss_feq.cpu()
                    loss_feq = (loss_feq.abs()**2).mean() if self.args.auxi_loss == 'MSE' else loss_feq.abs().mean()

                train_loss.append(loss.item())
                train_loss_tmp.append(loss_tmp.item()); train_loss_feq.append(loss_feq.item())
                train_loss_rec.append(loss_rec.item()); train_loss_auxi.append(loss_auxi.item())

                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss', loss.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss_tmp', loss_tmp.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss_feq', loss_feq.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss_rec', loss_rec.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/meta_test_iter/loss_auxi', loss_auxi.item(), self.step)

                if (i + 1) % 100 == 0:
                    print(f"\tMeta Test - iters: {i + 1}, epoch: {self.epoch} | loss: {loss.item():.7f}, loss tmp: {loss_tmp.item():.7f}, "
                          f"loss rec: {loss_rec.item():.7f}, loss feq: {loss_feq.item():.7f}, loss auxi: {loss_auxi.item():.7f}")
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f'\tspeed: {speed:.4f}s/iter; cost time: {cost_time:.4f}s; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_tmp = np.average(train_loss_tmp); train_loss_feq = np.average(train_loss_feq)
            train_loss_rec = np.average(train_loss_rec); train_loss_auxi = np.average(train_loss_auxi)
            valid_loss, valid_loss_tmp, valid_loss_feq, valid_loss_rec, valid_loss_auxi = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss_tmp', train_loss_tmp, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss_feq', train_loss_feq, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss_rec', train_loss_rec, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/meta_test/loss_auxi', train_loss_auxi, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', valid_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_tmp', valid_loss_tmp, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_feq', valid_loss_feq, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_rec', valid_loss_rec, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss_auxi', valid_loss_auxi, self.epoch)

            print(f"Epoch: {self.epoch} | Train Loss: {train_loss:.7f}, Tmp: {train_loss_tmp:.7f}, Feq: {train_loss_feq:.7f}, Rec: {train_loss_rec:.7f}, Auxi: {train_loss_auxi:.7f} | Valid Loss: {valid_loss:.7f}, Tmp: {valid_loss_tmp:.7f}, Feq: {valid_loss_feq:.7f}, Rec: {valid_loss_rec:.7f}, Auxi: {valid_loss_auxi:.7f}")
            early_stopping(valid_loss_tmp, self.model, path)
            if early_stopping.early_stop:
                print("Meta Test Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(valid_loss_tmp, self.epoch)

    def train(self, setting, prof=None):
        train_data, train_loader = self._get_data(flag='train')
        support_loader_list, query_loader_list = self.initialize_meta_tasks(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        criterion = self._select_criterion()

        # ============ Meta Train 阶段：只训练损失函数 ============
        print("\n>>>>>>>Meta Training Phase\n")
        self.meta_train(support_loader_list, query_loader_list, path, res_path)
        print("\n>>>>>>>Meta Training Phase completed\n")

        # ============ ML3 Meta Test 阶段：重新初始化模型，使用学习到的损失函数训练 ============
        print("\n>>>>>>>Meta Test Phase\n")
        self.meta_test(train_loader, vali_data, vali_loader, criterion, path)
        print("\n>>>>>>>Meta Test Phase completed\n")

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        best_ew_path = os.path.join(path, 'error_weighting.pth')
        self.error_weighting = torch.load(best_ew_path)

        return self.model

    def test(self, setting, prof=None, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            ckpt_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
            self.error_weighting = torch.load(os.path.join(ckpt_dir, 'error_weighting.pth'))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        self.error_weighting.eval()
        # metric_collector = create_metric_collector(device=self.device)
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

        # m = metric_collector.compute()
        # mae, mse, rmse, mape, mspe, mre = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"], m["mre"]
        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        with torch.no_grad():
            self.error_weighting.to(preds.device)
            loss, loss_tmp, loss_feq, loss_rec, loss_auxi = self.error_weighting(preds, trues)
        print('{}\t| mse:{}, mae:{}, loss:{}, loss feq:{}, loss rec:{}, loss auxi:{}'.format(self.pred_len, mse, mae, loss, loss_feq, loss_rec, loss_auxi))

        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/loss', loss, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/loss_feq', loss_feq, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/loss_rec', loss_rec, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/loss_auxi', loss_auxi, self.epoch)
        self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}, loss:{}, loss feq:{}, loss rec:{}, loss auxi:{}'.format(mse, mae, loss, loss_feq, loss_rec, loss_auxi))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, loss, loss_feq, loss_rec, loss_auxi, rmse, mape, mspe, mre]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())

        if not test or not os.path.exists(os.path.join(res_path, 'config.yaml')):
            print('save configs')
            args_dict = vars(self.args)
            with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
                yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return
