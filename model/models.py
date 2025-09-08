import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import json
from collections import defaultdict
import subprocess
import numpy as np
import matplotlib.pyplot
from IPython import display
import time
import re


@torch.no_grad()
def compute_cnn_output_lengths(model, input_lengths):
    if torch.is_tensor(input_lengths):
        return input_lengths.to(dtype=torch.long, device=SETTING["device"])
    return torch.tensor(input_lengths, dtype=torch.long, device=SETTING["device"])


class ResBlock(nn.Module):
    def __init__(self, num_cnn_layers, cnn_filters, cnn_kernel_size, use_resnet=True):
        super(ResBlock, self).__init__()
        self.use_resnet = use_resnet
        layers = []
        for _ in range(num_cnn_layers):
            layers.append(nn.Conv1d(cnn_filters, cnn_filters, cnn_kernel_size, padding=cnn_kernel_size // 2))
            layers.append(nn.BatchNorm1d(cnn_filters))
            layers.append(nn.PReLU())
        self.res_block = nn.Sequential(*layers)

    def forward(self, x):
        res = self.res_block(x)
        if self.use_resnet:
            return x + res
        return res


class ASRModel(nn.Module):
    def __init__(self, ip_channel, num_classes, num_res_blocks=3, num_cnn_layers=1, cnn_filters=50,
                 cnn_kernel_size=15, num_rnn_layers=2, rnn_dim=170, num_dense_layers=1,
                 dense_dim=300, use_birnn=True, use_resnet=True, rnn_type="lstm", rnn_dropout=0.6):
        super(ASRModel, self).__init__()

        # Initial Conv layer
        self.init_conv = nn.Sequential(
            nn.Conv1d(ip_channel, cnn_filters, cnn_kernel_size, padding=cnn_kernel_size // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.PReLU()
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_cnn_layers, cnn_filters, cnn_kernel_size, use_resnet) for _ in range(num_res_blocks)]
        )

        # RNN layers
        rnn_module = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        rnn_input_dim = cnn_filters
        self.rnns = nn.ModuleList()
        for _ in range(num_rnn_layers):
            if use_birnn:
                self.rnns.append(rnn_module(rnn_input_dim, rnn_dim, batch_first=True, dropout=rnn_dropout, bidirectional=True))
                rnn_input_dim = rnn_dim * 2
            else:
                self.rnns.append(rnn_module(rnn_input_dim, rnn_dim, batch_first=True, dropout=rnn_dropout))
                rnn_input_dim = rnn_dim

        # Layer Norm
        self.layer_norm = nn.LayerNorm(rnn_input_dim)

        # Dense layers
        dense_layers = []
        dense_in_dim = rnn_input_dim
        for _ in range(num_dense_layers):
            dense_layers.append(nn.Linear(dense_in_dim, dense_dim))
            dense_layers.append(nn.ReLU())
            dense_in_dim = dense_dim
        self.dense_layers = nn.Sequential(*dense_layers)

        # Output layer
        self.out_layer = nn.Linear(dense_in_dim, num_classes)

        # Config
        self.config = {
            "ip_channel": ip_channel,
            "num_classes": num_classes,
            "num_res_blocks": num_res_blocks,
            "num_cnn_layers": num_cnn_layers,
            "cnn_filters": cnn_filters,
            "cnn_kernel_size": cnn_kernel_size,
            "num_rnn_layers": num_rnn_layers,
            "rnn_dim": rnn_dim,
            "num_dense_layers": num_dense_layers,
            "dense_dim": dense_dim,
            "use_birnn": use_birnn,
            "use_resnet": use_resnet,
            "rnn_type": rnn_type,
            "rnn_dropout": rnn_dropout
        }


    def forward(self, x):
        x = x.transpose(1, 2) # (B, T, C) -> (B, C, T)
        x = self.init_conv(x)
        x = self.res_blocks(x)

        x = x.transpose(1, 2) # -> (B, T, C)
        for rnn in self.rnns:
            x, _ = rnn(x)

        x = self.layer_norm(x)
        x = self.dense_layers(x)
        x = self.out_layer(x)
        x = F.log_softmax(x, dim=-1)  # log_softmax for CTC Loss
        return x  # (B, T, C)

def save_checkpoint(model, optimizer, filename='checkpoint.pth.tar', config_filename='model_config.json'):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    print("=> Saving checkpoint")
    torch.save(checkpoint, filename)

    # Save model configuration
    with open(config_filename, 'w') as f:
        json.dump(model.config, f)
    print("=> Saving model configuration")

def load_checkpoint(checkpoint, model, optimizer):
  print("=> Loading checkpoint")
  model.load_state_dict(checkpoint["state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer"])

def train_fn(train_loader, model, optimizer, loss_fn):
    model.train()
    total_losses = []

    inner_loop = tqdm(train_loader, desc='Batch', leave=False, position=0)

    for x, y, input_lengths, target_lengths in inner_loop:
      # Target lengths should be the longest tensor in the batch (for labels)
      # Check for NaNs/Infs before placing it to device
      if torch.isnan(x).any() or torch.isinf(x).any():
        print("NaN or Inf in input!")
        raise ValueError("Invalid model input")

      # place data to device
      x, y = x.to(SETTING["device"]), y.to(SETTING["device"])
      input_lengths = compute_cnn_output_lengths(model, input_lengths).to(SETTING["device"])
      target_lengths = target_lengths.to(SETTING["device"])

      log_probs = model(x) # (B, T, C)
      T = log_probs.size(1) # TRUE time length
      input_lengths = torch.clamp(input_lengths, max=T)  # also trim the batch
      log_probs = log_probs.transpose(0, 1) # (T, B, C) for CTCLoss

      # Feasibility check: no silent zero-loss
      bad = (target_lengths > input_lengths).nonzero(as_tuple=False).flatten()
      if bad.numel() > 0:
            raise ValueError(
                f"CTC invalid: target_lengths > input_lengths for items {bad.tolist()} "
                f"(max target={int(target_lengths.max())}, max input={int(input_lengths.max())}, T={T})"
            )

      y_concat = torch.cat([y[i][:target_lengths[i].item()] for i in range(y.size(0))])
      # calculate loss
      loss = loss_fn(log_probs, y_concat, input_lengths, target_lengths)

      # Check for NaNs/Infs among loss
      if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
        print("NaN or Inf in loss!")
        raise ValueError("Invalid loss")

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
      optimizer.step()

      total_losses.append(loss.item())

    return sum(total_losses) / len(total_losses)

def eval_fn(dev_loader, model, loss_fn):
    model.eval()
    total_losses = []

    with torch.no_grad():
        for x, y, input_lengths, target_lengths in dev_loader:
            # Check for NaNs/Infs in input
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("NaN or Inf in input during evaluation!")
                continue  # Skip this batch instead of raising

            x, y = x.to(SETTING['device']), y.to(SETTING['device'])
            input_lengths = compute_cnn_output_lengths(model, input_lengths).to(SETTING['device'])
            target_lengths = target_lengths.to(SETTING['device'])

            log_probs = model(x)  # (B, T, C)
            T = log_probs.size(1)  # true time steps
            input_lengths = torch.clamp(input_lengths, max=T)  # same
            log_probs = log_probs.transpose(0, 1)  # (T, B, C)

            # sanity check
            if (target_lengths > input_lengths).any():
                raise ValueError("CTC invalid in eval: target length > input length")

            y_concat = torch.cat([y[i, :target_lengths[i].item()] for i in range(y.size(0))])
            loss = loss_fn(log_probs, y_concat, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN or Inf in loss during evaluation!")
                continue

            total_losses.append(loss.item())

    return sum(total_losses) / len(total_losses) if total_losses else float('inf')

SETTING = {
    "seed": 43,
    "learning_rate": 1e-3,  # made much bigger
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "batch_size": 64,
    "weight_decay": 1e-4,
    "num_epochs": 50,
    "num_workers": 2,
    "pin_memory": True,
    "load_model": True,
    "load_model_file": "/content/drive/MyDrive/ResNetCTC.path.tar",
    "patience": 10,
#    "feat_dir": directory of features
#    "label_dir": direct1ory of labels
}

torch.manual_seed(SETTING["seed"])