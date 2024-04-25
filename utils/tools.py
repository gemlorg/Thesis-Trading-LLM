import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm

def cg_metric(pred, true):
    pred = np.sign(pred)
    return np.sum(pred == true) / len(pred)

def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    cg_loss = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            outputs = model(batch_x)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)
            cg_loss.append(cg_metric(outputs, batch_y))
            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    cg_loss = np.average(cg_loss)

    model.train()
    return total_loss, cg_loss

def generate_pathname(args, ii):
    return "{}_{}_{}_nl{}_{}".format(
        args.model_id,
        args.model,
        args.data,
        args.num_lags,
        ii,
    )
