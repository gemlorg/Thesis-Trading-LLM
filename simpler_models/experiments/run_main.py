import sys
sys.path.append("..")

from utils.config_parser import get_args
from utils.tools import generate_pathname, vali, cg_metric, adjust_learning_rate, get_last_vals
import torch
from torch import nn, optim
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import json
import csv
from sklearn.metrics import accuracy_score
from models.linear_regression import LinearRegression
from models.cnn import CNN
from models.mlp import MLP
from models.resNet import ResNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from accelerate import Accelerator
from torch.optim import lr_scheduler

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args = get_args()

accelerator = Accelerator()

train_data, train_loader = data_provider(args, "train")
vali_data, vali_loader = data_provider(args, "val")
test_data, test_loader = data_provider(args, "test")

in_dim = len(train_data[0][0])
out_dim = 1

is_sklearn_model = False

if args.model == "LR":
    model = LinearRegression(in_dim, out_dim).float()
elif args.model == "CNN":
    model = CNN(
        input_channels=in_dim,
        output_channels=out_dim,
        conv_layers=1,
        conv_out_channels=[128],
        conv_kernel_sizes=[5],
        dense_layers=2,
        dense_units=[256, 128],
        seq_length=len(train_data[0][0][0]),
        activation_fn=nn.functional.sigmoid,
    )
elif args.model == "MLP":
    model = MLP([in_dim, 64, 64, out_dim])
elif args.model == "ResNet":
    model = ResNet(dense_units=128)
elif args.model == "RF":
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        criterion=args.criterion,
        random_state=fix_seed,
    )
    is_sklearn_model = True
elif args.model == "SVM":
    model = SVC(kernel = args.kernel, gamma = args.gamma, C = args.C, random_state = fix_seed)
    is_sklearn_model = True

setting = generate_pathname(args, 0)

path = os.path.join(
    "results/", setting + "-" + args.model_comment
)

if not os.path.exists(path):
    os.makedirs(path)
with open(path + '/' + 'args', 'w+') as f:
    json.dump(args.__dict__, f, indent=2)

if is_sklearn_model:
    model.fit(train_data.data_x, train_data.data_y)

    y_test_pred = model.predict(test_data.data_x)
    accuracy = accuracy_score(y_test_pred, test_data.data_y)

    print("Accuracy:", accuracy)
else:
    res_header = ["Epoch", "LearningRate", "TrainLoss", "ValiLoss", "TestLoss", "TrainCG", "TestCG", "ValiCG"]

    csvres = open(path + '/results.csv', 'w+')
    reswriter = csv.writer(csvres)
    reswriter.writerow(res_header)

    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    optimizer = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == "COS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=1e-8
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )

    criterion = nn.MSELoss()

    train_data, train_loader, vali_loader, test_loader, model, optimizer, scheduler = (
        accelerator.prepare(
            train_data, train_loader, vali_loader, test_loader, model, optimizer, scheduler
        )
    )

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        train_cg_loss = []

        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)

            outputs = model(batch_x)

            loss = criterion(torch.flatten(outputs), batch_y)

            train_loss.append(loss.item())

            outputs = torch.flatten(outputs.detach())
            batch_y = batch_y.detach()
            last_vals = get_last_vals(args, batch_x.detach())
            train_cg_loss.append(cg_metric(outputs, batch_y, last_vals))

            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print(
                    "\tspeed: {:.4f}s/iter; left time: {:.4f}s, CG: ".format(
                        speed, left_time) + str(np.mean(train_cg_loss))
                )
                iter_count = 0
                time_now = time.time()

            accelerator.backward(loss)
            optimizer.step()

            if args.lradj == "TST":
                adjust_learning_rate(
                    accelerator, optimizer, scheduler, epoch + 1, args, printout=False
                )
                scheduler.step()

        accelerator.print(
            "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
        )
        train_loss = np.average(train_loss)
        vali_loss, vali_cg_loss = vali(
            args, accelerator, model, vali_data, vali_loader, criterion
        )
        test_loss, test_cg_loss = vali(
            args, accelerator, model, test_data, test_loader, criterion
        )
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} CG_train: {4} CG_test: {5} CG_vali: {6} ".format(
                epoch + 1, train_loss, vali_loss, test_loss, np.mean(train_cg_loss), test_cg_loss, vali_cg_loss
            )
        )
#        ["Epoch", "LearningRate", "TrainLoss", "ValiLoss", "TestLoss", "TrainCG", "TestCG", "ValiCG"]
        reswriter.writerow([epoch+1, optimizer.param_groups[0]["lr"],
                           train_loss, vali_loss, test_loss, np.mean(train_cg_loss), test_cg_loss, vali_cg_loss])
        csvres.flush()

        if args.lradj != "TST":
            if args.lradj == "COS":
                scheduler.step()
                accelerator.print(
                    "lr = {:.10f}".format(optimizer.param_groups[0]["lr"])
                )
            else:
                if epoch == 0:
                    args.learning_rate = optimizer.param_groups[0]["lr"]
                    accelerator.print(
                        "lr = {:.10f}".format(
                            optimizer.param_groups[0]["lr"])
                    )
                adjust_learning_rate(
                    accelerator, optimizer, scheduler, epoch + 1, args, printout=True
                )
        else:
            accelerator.print(
                "Updating learning rate to {}".format(
                    scheduler.get_last_lr()[0])
            )

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        csvres.close()

