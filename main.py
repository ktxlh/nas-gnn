# Train loop (NGE Algorithm 1)
"""
nor := normal cell
red := reduction cell
"""
import json
import os
from argparse import ArgumentParser
from os.path import isdir, join

import torch
import torch.nn as nn
from tqdm import trange

from data import cifar10_dataloaders
from model import TrainArch, ValidArch
from nge import NGE
from train import get_final_performance, update_w, update_x_theta
from utils import count_parameters_in_MB, plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    torch.manual_seed(220589)
    torch.cuda.manual_seed(220589)
    CIFAR_CLASSES = 10
    
    max_epochs = 10
    num_nodes = 7  # |V|
    init_channels = 16
    layers = 8
    batch_size = 64
    criterion = nn.CrossEntropyLoss()
    
    parser = ArgumentParser()
    parser.add_argument('--name', default='None')
    parser.add_argument('--small', action="store_true", help="use len=10 mini subsets. output will postfix -small")
    parser.add_argument('--weighted', action="store_true", help="weighted GCN")
    # parser.add_argument('--attn', action="store_true", help="self-attentive NGE")
    args = parser.parse_args()
    
    tag = args.name + ('-small' if args.small else '')
    out_dir = join(os.getcwd(), 'output', tag)
    if not isdir(out_dir):
        os.mkdir(out_dir)
    
    # Train
    arch_train = TrainArch(init_channels, CIFAR_CLASSES, layers, criterion).to(device)
    print("param size = {:.4f}MB".format(count_parameters_in_MB(arch_train)))
    
    nge_normal = NGE(num_nodes, args.weighted).to(device)
    nge_reduce = NGE(num_nodes, args.weighted).to(device)
    
    dl_train, dl_valid = cifar10_dataloaders(batch_size)
    
    w_optimizer = torch.optim.SGD(arch_train.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    x_theta_optimizer = torch.optim.Adam([
        {'params': nge_normal.parameters()},
        {'params': nge_reduce.parameters()},
    ], lr=6e-4, weight_decay=1e-3)
    
    w_losses, xt_losses = [], []
    for _ in trange(max_epochs, desc='train loop'):
        losses = update_w(dl_train, arch_train, nge_normal, nge_reduce, w_optimizer, args.small)
        w_losses.extend(losses)
        xt_losses.extend([-1 for i in range(len(losses))])
        
        losses = update_x_theta(dl_valid, arch_train, nge_normal, nge_reduce, x_theta_optimizer, args.small)
        xt_losses.extend(losses)
        w_losses.extend([-1 for i in range(len(losses))])
    
    
    plot(tag, 'train', {'w_loss': w_losses, 'xt_loss': xt_losses}, scatter=True)
    torch.save(arch_train.state_dict(), join(out_dir, 'arch_train.pt'))
    torch.save(nge_normal.state_dict(), join(out_dir, 'nge_normal.pt'))
    torch.save(nge_reduce.state_dict(), join(out_dir, 'nge_reduce.pt'))
    
    # Valid
    arch_valid = ValidArch(init_channels, CIFAR_CLASSES, layers, criterion).to(device)
    valid_optimizer = torch.optim.SGD(arch_valid.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    print("param size = {:.4f}MB".format(count_parameters_in_MB(arch_valid)))
    
    for _ in trange(max_epochs, desc='valid loop'):
        valid_losses = update_w(dl_train, arch_valid, nge_normal, nge_reduce, valid_optimizer, args.small)
    plot(tag, 'valid', {'loss': valid_losses}, scatter=False)
    
    acc_1, acc_5 = get_final_performance(dl_valid, arch_valid, nge_normal, nge_reduce, args.small)
    torch.save(arch_valid.state_dict(), join(out_dir, 'arch_valid.pt'))
    with open(join(out_dir, 'logs.json'), 'w') as f:
        json.dump({
            'Top 1 acc': acc_1.item(),
            'Top 5 acc': acc_5.item(),
            'w_losses': w_losses,
            'xt_losses': xt_losses,
            'valid_losses': valid_losses,
            }, f, indent=True)


if __name__ == '__main__':
    main()
