import os
import pickle
import random
import re
import shutil
import time
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return {"result": result, "time": end_time - start_time}
    return wrapper


def load_npy(path):
    return np.load(path)


def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_yaml_as_df(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    df = pd.json_normalize(data)
    return df


def exist_metric(exp_dir):
    try:
        
        res_dir = os.path.join(exp_dir, 'results')
        setting = os.listdir(res_dir)[0]
        setting_dir = os.path.join(res_dir, setting)
        metric_path = os.path.join(setting_dir, 'metrics.npy')
        if os.path.exists(metric_path):
            return True, setting_dir
        else:
            return False, None
    except Exception as e:
        return False, None


def exist_pred(exp_dir):
    try:
        
        res_dir = os.path.join(exp_dir, 'results')
        setting = os.listdir(res_dir)[0]
        setting_dir = os.path.join(res_dir, setting)
        metric_path = os.path.join(setting_dir, 'pred.npy')
        if os.path.exists(metric_path):
            return True, setting_dir
        else:
            return False, None
    except Exception as e:
        return False, None


def exist_stf_metric(exp_dir):
    try:
        res_dir = os.path.join(exp_dir, 'results')
        metric_dir = os.path.join(res_dir, "m4_results")
        metric_path = os.path.join(metric_dir, 'metrics.pkl')
        if os.path.exists(metric_path):
            settings = os.listdir(res_dir)
            setting = [s for s in settings if 'Hourly' in s][0]
            setting_dir = os.path.join(res_dir, setting)
            return True, metric_dir, setting_dir
        else:
            return False, None, None
    except Exception as e:
        return False, None, None


def inverse_stf_metrics(metrics, names):
    new_metrics = {}
    for key, value in metrics.items():
        if key not in names:
            continue
        for k, v in value.items():
            if k in new_metrics:
                new_metrics[k][key] = v
            else:
                new_metrics[k] = {key: v}
    return new_metrics


def keep_split(exp, special_words=[]):
    pattern = '|'.join(map(re.escape, special_words)) + '|_'
    parts = re.findall(f'({pattern})|([^_]+)', exp)
    result = [part[0] or part[1] for part in parts if any(part) and (part[0] or part[1]) != '_']
    output = []
    for part in result:
        try: part = eval(part)
        except: pass
        output.append(part)
    return output


def is_full_group(x):
    if str(x['data_id'].iloc[0]).startswith('PEMS'):
        return set(x['pred_len']) == {12, 24, 36, 48}
    else:
        return set(x['pred_len']) == {96, 192, 336, 720}


def load_metric_from_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    for line in lines[::-1]:
        if line.strip() == '':
            continue

        if 'mse' in line and 'mae' in line:
            parts = line.strip().split(', ')
            metrics = {}
            for part in parts:
                key_value = part.split(':')
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    try:
                        value = float(key_value[1].strip())
                    except ValueError:
                        value = key_value[1].strip()
                    metrics[key] = value
            return metrics
    return None