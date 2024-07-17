# configs/config_file.py
import torch
class Config:
    epochs = 100
    eval_steps = 100
    callback_path = 'callback'
    precision = torch.float32
    accumulated_steps = 1
    eval_period_step = 100
    eval_period_epoch = 1
    saving_period_step = 500
    saving_period_epoch = 1
    log_figure_period_step = 100
    log_figure_period_epoch = 1
    grad_init_scale = 65536.0
    detect_anomaly = False
    recompute_metrics = False
