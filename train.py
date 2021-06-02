import torch
from tqdm import tqdm
from utils import accuracy
from numpy.random import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_w(dl_train, arch, nge_normal, nge_reduce, optimizer, small=False):
    # Line 6-8 of NGE Algo 1
    arch.train()
    nge_normal.eval()
    nge_reduce.eval()
    losses = []
    
    with torch.no_grad():
        p_normal = nge_normal()  # num_nodes x num_nodes x num_ops
        p_reduce = nge_reduce()
    
    for input, target in tqdm(dl_train, 'dl_train'):
        if small and random() > 0.1: continue
        optimizer.zero_grad()
        logits = arch(input.cuda(), p_normal, p_reduce)
        loss = arch._loss(logits, target.cuda())
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
    
    return losses

def update_x_theta(dl_valid, arch_train, nge_normal, nge_reduce, optimizer, small=False):
    # Line 9-14 of NGE Algo 1
    arch_train.eval()
    nge_normal.train()
    nge_reduce.train()
    losses = []
    
    for input, target in tqdm(dl_valid, 'dl_valid'):
        if small and random() > 0.1: continue
        optimizer.zero_grad()
        p_normal = nge_normal()
        p_reduce = nge_reduce()
        logits = arch_train(input.cuda(), p_normal, p_reduce)
        loss = arch_train._loss(logits, target.cuda())
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        
    return losses

def get_final_performance(dl_valid, arch_valid, nge_normal, nge_reduce, small=False):
    # Line 16 of NGE Algo 1
    nge_normal.eval()
    nge_reduce.eval()
    arch_valid.eval()
    with torch.no_grad():
        p_normal = nge_normal()
        p_reduce = nge_reduce()
        total_loss = 0
        total_acc_1 = 0
        total_acc_5 = 0
        num_items = 0
        
        for input, target in tqdm(dl_valid, 'dl_valid'):
            if small and random() > 0.1: continue
            logits = arch_valid(input.cuda(), p_normal, p_reduce)
            loss = arch_valid._loss(logits, target.cuda())
            acc = accuracy(logits.cpu(), target)
            total_acc_1 += acc[0] * len(input)
            total_acc_5 += acc[1] * len(input)
            total_loss += loss.cpu().item() * len(input)
            num_items += len(input)
            
        print("Total Accuracy Top-1: {:.4f}".format(total_acc_1 / num_items))
        print("Total Accuracy Top-5: {:.4f}".format(total_acc_5 / num_items))
    return total_acc_1 / num_items, total_acc_5 / num_items