from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os, sys
import time
import random
import numpy as np
import os
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

sys.path.append("../..")
import models.llm as llm
import torch
import numpy as np
import experiments.utils as utils
from sklearn.preprocessing import minmax_scale
import models.llm as llm

data_path = os.path.join(
    os.path.dirname(__file__), "../../data/gbpcad_one_hour_202311210827.csv"
)
data_name = "gpbcad"
torch.manual_seed(42)
num_lags = 5
lr = 0.001
epochs = 10
label_len = 48
pred_len = 96
batch_size = 16

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

train_data, train_loader = utils.split_data("gbpcad", "train", num_lags, batch_size )
test_data, test_loader = utils.split_data("gbpcad", "test", num_lags, batch_size)
vali_data, vali_loader = utils.split_data("gbpcad", "validate", num_lags, batch_size)

model = llm.Model().float()

trained_parameters = []

for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)
model_optimizer = torch.optim.Adam(trained_parameters, lr=lr)
criterion = nn.MSELoss()
mae_metric = nn.L1Loss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=20, eta_min=1e-8)

train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optimizer, scheduler)

for epoch in range(epochs):
    iter_count = 0
    train_loss = []
    model.train()

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
        iter_count += 1
        model_optim.zero_grad()

        batch_x = batch_x.float().to(accelerator.device)
        batch_y = batch_y.float().to(accelerator.device)
        batch_x_mark = batch_x_mark.float().to(accelerator.device)
        batch_y_mark = batch_y_mark.float().to(accelerator.device)
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float().to(
                accelerator.device)
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        outputs = outputs[:, -pred_len:, :]
        batch_y = batch_y[:, -pred_len:, :]
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())
        accelerator.print(f"Epoch: {epoch}, Iteration: {iter_count}, Loss: {loss.item()}")
        accelerator.backward(loss)
        model_optim.step()

