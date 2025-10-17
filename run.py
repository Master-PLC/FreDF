import argparse
import os
import random
import sys

import numpy as np
import setproctitle
import torch
import torch.profiler as profiler
from exp import EXP_DICT
from utils.print_args import print_args
from utils.tools import EvalAction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--fix_seed', type=int, default=2023, help='random seed')
    parser.add_argument('--rerun', type=int, help='rerun', default=0)
    parser.add_argument('--verbose', type=int, help='verbose', default=0)

    # save
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./results/', help='location of results')
    parser.add_argument('--test_results', type=str, default='./test_results/', help='location of test results')
    parser.add_argument('--log_path', type=str, default='./result_long_term_forecast.txt', help='log path')
    parser.add_argument('--output_pred', action='store_true', help='output true and pred', default=False)
    parser.add_argument('--output_vis', action='store_true', help='output visual figures', default=False)

    # data loader
    parser.add_argument('--data_id', type=str, default='ETTm1', help='dataset name')
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: [M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--add_noise', action='store_true', help='add noise')
    parser.add_argument('--noise_amp', type=float, default=1, help='noise ampitude')
    parser.add_argument('--noise_freq_percentage', type=float, default=0.05, help='noise frequency percentage')
    parser.add_argument('--noise_seed', type=int, default=2023, help='noise seed')
    parser.add_argument('--noise_type', type=str, default='sin', help='noise type, options: [sin, normal]')
    parser.add_argument('--cutoff_freq_percentage', type=float, default=0.06, help='cutoff frequency')
    parser.add_argument('--data_percentage', type=float, default=1., help='percentage of training data')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--reconstruction_type', type=str, default="imputation", help='type of reconstruction')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')

    # optimization
    parser.add_argument('--optim_type', type=str, default='adam', help='optimizer type')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--auxi_batch_size', type=int, default=1024, help='batch size of test input data')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', action=EvalAction, default='type1', help='adjust learning rate')
    parser.add_argument('--step_size', type=int, default=1, help='step size for learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--mode', type=int, default=0, help='mode for learning rate decay')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--pct_start', type=float, default=0.2, help='Warmup ratio for the learning rate scheduler')

    # FreDF
    parser.add_argument('--rec_lambda', type=float, default=0., help='weight of reconstruction function')
    parser.add_argument('--auxi_lambda', type=float, default=1, help='weight of auxilary function')
    parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft]')
    parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase]')
    parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
    parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')
    parser.add_argument('--offload', type=int, default=0)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--thread', type=int, default=1)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # Meta
    parser.add_argument("--pretrain_model_path", default=None, type=str)
    parser.add_argument('--meta_lr', type=float, default=0.0005, help='meta learning rate')
    parser.add_argument('--inner_lr', type=float, default=0.0005, help='inner learning rate')
    parser.add_argument('--meta_inner_steps', type=int, default=1, help='meta inner steps')
    parser.add_argument('--num_tasks', type=int, default=5, help='number of tasks')
    parser.add_argument('--overlap_ratio', type=float, default=0.15, help='overlap ratio between tasks')
    parser.add_argument('--meta_optim_type', type=str, default='sgd', help='optimizer type')
    parser.add_argument('--max_norm', type=float, default=1.0, help='max norm for gradient clipping')
    parser.add_argument('--first_order', type=int, default=1, help='first order approximation; True 1 False 0')
    parser.add_argument('--model_per_task', type=int, default=0, help='separate model for each task; True 1 False 0')
    parser.add_argument('--meta_type', type=str, default='all', help='meta learning type')
    parser.add_argument('--weighting_type', type=str, default='softmax', help='type of weighting for auxi loss, options: [softmax, minmax]')

    args = parser.parse_args()

    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    torch.set_num_threads(args.thread)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name in EXP_DICT:
        Exp = EXP_DICT[args.task_name]

    # setproctitle.setproctitle(args.task_name)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_ax{}_rl{}_axl{}_mf{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.auxi_lambda,
                args.rec_lambda,
                args.auxi_loss,
                args.module_first,
                args.des,
                ii
            )

            if not args.rerun and os.path.exists(os.path.join(args.results, setting, "metrics.npy")):
                print(f">>>>>>>setting {setting} already run, skip")
                sys.exit(0)

            exp = Exp(args)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_ax{}_rl{}_axl{}_mf{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.auxi_lambda,
            args.rec_lambda,
            args.auxi_loss,
            args.module_first,
            args.des,
            ii
        )

        # if not args.rerun and os.path.exists(os.path.join(args.results, setting, "metrics.npy")):
        #     print(f">>>>>>>setting {setting} already run, skip")
        #     sys.exit(0)

        exp = Exp(args)  # set experiments

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
