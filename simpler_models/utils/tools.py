import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

def cg_metric(pred, true, last_vals):
    pred_diff = torch.sign(pred - last_vals)
    true_diff = torch.sign(true - last_vals)
    return (torch.sum(pred_diff == true_diff) / len(pred_diff)).item()

def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     2 else args.learning_rate * (0.75 ** ((epoch - 2) // 2))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.90 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))

def get_last_vals(args, inputs):
    if args.model == "ResNet":
        return inputs[:, 0, 0, 0]
    elif args.model == "CNN":
        return inputs[:, 0, 0]
    else:
        return inputs[:, 0]

def vali(args, accelerator, model, vali_data, vali_loader, criterion):
    total_loss = []
    cg_loss = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            outputs = model(batch_x)

            pred = torch.flatten(outputs.detach())
            true = batch_y.detach()
            last_vals = get_last_vals(args, batch_x.detach())

            loss = criterion(pred, true)
            cg_loss.append(cg_metric(pred, batch_y, last_vals))
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

def split_into_lists(t, chunk_len = 3):
    while len(t) % chunk_len != 0:
        np.append(t, [0.])
    nt = []
    for i in range(0, len(t), chunk_len):
        nt.append([[t[i], t[i + 1], t[i + 2]]])
    return nt
