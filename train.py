import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score

import time
import json

from load import accuracy, load_multi_data
from models import FusionGAT3


seed = 72
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, l1_loss = model(features, adj_list_tt)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + l1_loss*lambda_l1
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test]) + l1_loss*lambda_l1
    acc_test = accuracy(output[idx_test], labels[idx_test])
    y_prob = torch.exp(output[:, 1]).detach().numpy()
    auc_score = roc_auc_score(labels[idx_test].detach().numpy(), y_prob[idx_test])
    print('Epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.3f}'.format(loss_train.data.item()),
          'acc_train: {:.3f}'.format(acc_train.data.item()),
          'loss_test: {:.3f}'.format(loss_test.data.item()),
          'acc_test: {:.3f}'.format(acc_test.data.item()),
          "AUC = {:.2f}".format(auc_score),
          'time: {:.3f}s'.format(time.time() - t))

    return loss_test.data.item(), acc_test, auc_score

def compute_test(model):
    model.eval()
    output, l1_loss = model(features, adj_list_tt)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test]) + l1_loss*lambda_l1
    acc_test = accuracy(output[idx_test], labels[idx_test])
    y_prob = torch.exp(output[:, 1]).detach().numpy()
    auc_score = roc_auc_score(labels[idx_test].detach().numpy(), y_prob[idx_test])
    # compute false alarm rate and f1 score
    y_pred = np.array([0 if p < 0.5 else 1 for p in y_prob])
    false_alarm_rate = np.mean(y_pred[idx_test] == 1)
    f1score = f1_score(labels[idx_test].detach().numpy(), y_pred[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()),
          "AUC = {:.2f}".format(auc_score),
          "False Alarm Rate = {:.4f}".format(false_alarm_rate),
          "F1 Score = {:.4f}".format(f1score))
    return loss_test.data.item(), acc_test.data.item(), auc_score, false_alarm_rate, f1score, y_pred


#parameter setting
cuda = False
epochs = 500
lr = 0.0005
weight_decay = 5e-5
dropout = 0.3
hidden_dims1 = 64
hidden_dims2 = 64
fusion1_dim = 4
nb_head = 8
alpha = 0.2 #Alpha for the leaky_relu
lambda_l1s = [0.0001, 0.001, 0.01]
patience = 50

# Load data
sv_path = ""
folder = "../data/"
att_name = "att_pp_main.csv"
edge_list_name = ["edge1.csv", "edge2.csv", "edge3.csv"]
adj_list, features, labels, idx_train, idx_test = load_multi_data(folder, att_name, edge_list_name)

adj_list_tensor = torch.stack(adj_list)
features, adj_list_tt, labels = Variable(features), adj_list_tensor, Variable(labels)

count = 0
for lambda_l1 in lambda_l1s:                      
    count += 1
    # Model and optimizer
    model = FusionGAT3(nfeat=features.shape[1], 
                nhid1=hidden_dims1, 
                nhid2=hidden_dims2,
                fusion1_dim = fusion1_dim,
                nclass=int(labels.max()) + 1, 
                dropout=dropout, 
                alpha=alpha,
                adj_list= adj_list_tt,
                nheads = nb_head)

    optimizer = optim.Adam(model.parameters(), 
                        lr=lr, 
                        weight_decay=weight_decay)

    if cuda:
        model.cuda()
        features = features.cuda()
        adj_list_tt = adj_list_tt.cuda()
        labels = labels.cuda()
        idx_train = torch.tensor(idx_train).cuda()
        idx_test = torch.tensor(idx_test).cuda()

    
    # Train model
    t_total = time.time()
    bad_counter = 0
    best_auc = -1
    best_epoch = 0
    for epoch in range(epochs):
        loss, acc, auc = train(epoch)

        torch.save(model.state_dict(), sv_path + '{}.pkl'.format(epoch))
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

    files = glob.glob(sv_path +'*.pkl')
    for file in files:
        filename = file.split('/')[1]
        epoch_nb = int(filename.split('.')[0])
        if epoch_nb != best_epoch:
            os.remove(file)
            
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(sv_path +'{}.pkl'.format(best_epoch)))

    # Testing
    test_loss, test_acc, auc, far, f1, y_pred = compute_test(model)
    np.save(f"y_pred{count}.npy", y_pred)
    hyper_para = {}
    hyper_para["lr"] = lr
    hyper_para["weight_decay"] = weight_decay
    hyper_para["dropout"] = dropout
    hyper_para["hidden_dim1"] = hidden_dims1
    hyper_para["hidden_dim2"] = hidden_dims2
    hyper_para["fusion1_dim"] = fusion1_dim
    hyper_para["nb_heads"] = nb_head
    hyper_para["lambda"] = lambda_l1
    hyper_para["loss"] = test_loss
    hyper_para["accuracy"] = test_acc
    hyper_para["auc"] = auc
    hyper_para["false_alarm_rate"] = far
    hyper_para["f1_score"] = f1
    with open(sv_path +"hyperpara.json", "a+") as fp:
        fp.write('\n')
        json.dump(hyper_para, fp)