import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *
from layer import SSCLMD
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, f1_score, auc

def train_model( data_s, data_f, train_loader, test_loader, args):
    # feats_dim_list = [i.shape[1] for i in feats]

    model = SSCLMD(in_dim = args.dimensions, hid_dim= args.hidden1, out_dim = args.hidden2, decoder1=args.decoder1)
    # model.load_state_dict(torch.load("dataset1/MDA_model_decLDA.pth"))
    # model.load_state_dict(torch.load("dataset1/MDA_LMI_model_decLDA.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    m = torch.nn.Sigmoid() # -1,1
    loss_fct = torch.nn.BCEWithLogitsLoss()  #
    loss_node = torch.nn.BCELoss()


    loss_history = []
    if args.cuda:  # GPU
        model.to("cuda")
        data_s.to("cuda")
        data_f.to("cuda")
        # lbl.to("cuda")
        # feats = [feat.to("cuda") for feat in feats]

    # Train model
    t_total = time.time()
    print('Start Training...')

    for epoch in range(args.epochs):
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        lbl_1 = torch.ones(997 * 2)  # dataset1: 997, dataset2: 1071
        lbl_2 = torch.zeros(997 * 2)
        lbl = torch.cat((lbl_1, lbl_2)).cuda()
        # all_loss = 0

        for i, (label, inp) in enumerate(train_loader):

            if args.cuda:
                label = label.cuda()
            # print("inp_d: ",inp)
            # print("label: ", len(label))
            model.train()
            optimizer.zero_grad()

            output, log = model(data_s, data_f, inp)
            log = torch.squeeze(m(log))
            loss_class = loss_node(log, label.float())  #

            loss_constra = loss_fct(output, lbl)
            loss_train = loss_class + args.loss_ratio1 * loss_constra
            # print("loss: ",args.loss_ratio1, args.loss_ratio2)
            # print("att: ",att)

            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + log.flatten().tolist()

            if i % 100 == 0:  #
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))
        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # average_loss = all_loss / len(train_loader)
        # loss_history.append(average_loss.item())
        print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(roc_train),
                  'time: {:.4f}s'.format(time.time() - t))

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test = test(model, test_loader, data_s, data_f, args)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))

    # test(model, test_loader, data_s, data_f, args)

def test(model, loader, data_s, data_f, args):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCEWithLogitsLoss()
    loss_node = torch.nn.BCELoss()

    lbl_1 = torch.ones(997 * 2)
    lbl_2 = torch.zeros(997 * 2)
    lbl = torch.cat((lbl_1, lbl_2)).cuda()

    inp_id0 = []
    inp_id1 = []

    model.eval()
    y_pred = []
    y_label = []

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            inp_id0.append(inp[0])
            inp_id1.append(inp[1])

            if args.cuda:
                label = label.cuda()

            output, log = model(data_s, data_f, inp)
            log = torch.squeeze(m(log))
            loss_class = loss_node(log, label.float())
            loss_constra = loss_fct(output, lbl)

            loss = loss_class + args.loss_ratio1 * loss_constra

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + log.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss

