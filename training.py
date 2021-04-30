import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import torch.nn.functional as F


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
#         output = F.softmax(output)
        loss = F.cross_entropy(output, data.y.long().to(device))
#         loss = loss_fn(output, data.y.view(-1, 1).long().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    predict_labels = []
    predict_scores = []
    correct_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            ys = F.softmax(output, 1).to('cpu').data.numpy() 
            predict_labels = predict_labels + list(map(lambda x: np.argmax(x), ys))
            predict_scores = predict_scores + list(map(lambda x: x[1], ys))
#             print(total_labels)
#             print(total_scores)
#             total_preds = torch.cat((total_preds, output.cpu()), 0)
            correct_labels = torch.cat((correct_labels, data.y.cpu()), 0)
    return np.array(predict_labels),np.array(predict_scores),correct_labels.numpy()

#datasets = [['human','celegans'][int(sys.argv[1])]]
#modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
datasets = ['human','celegans']
modeling_list = [GINConvNet, GATNet, GAT_GCN, GCNNet]
negative_ratios = ['1','3','5']
for modeling in modeling_list:
    print(modeling)
    model_st = modeling.__name__

    cuda_name = "cuda:0"
    #if len(sys.argv)>3:
        #cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
    print('cuda_name:', cuda_name)

    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 1000

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    # Main program: iterate over different datasets
    for dataset in datasets:
        for negative_ratio in negative_ratios:
            print('\nrunning on ', model_st + '_' + dataset )
            processed_data_file_train = 'data/processed/' + negative_ratio + '_' + dataset + '_train.pt'
            processed_data_file_test = 'data/processed/' + negative_ratio + '_' + dataset + '_test.pt'
            if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
                print('please run create_data.py to prepare data in pytorch format!')
            else:
                train_data = TestbedDataset(root='data', dataset=negative_ratio+'_'+dataset+'_train')
                test_data = TestbedDataset(root='data', dataset=negative_ratio+'_'+dataset+'_test')

                # make data PyTorch mini-batch processing ready
                train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
                #加载预训练词向量
                embed_matrix = pd.read_csv('./data/embed/'+negative_ratio+'_'+dataset+'_'+'embedding_matrix.csv')
                embed_matrix = embed_matrix.values
                # training the model
                device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
                model = modeling(embed_matrix=embed_matrix).to(device)
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                best_auc = 0
                best_acc = 0
                best_epoch = -1
                model_file_name = './pretrained/model_' + negative_ratio+model_st + '_' + dataset +  '.model'
                result_file_name = './result/result_' + negative_ratio+model_st + '_' + dataset +  '.csv'
                for epoch in range(NUM_EPOCHS):
                    train(model, device, train_loader, optimizer, epoch+1)
                    G,P,T = predicting(model, device, test_loader)
        #             for a,b,c in zip(G, P, T):
        #                 print(a, b, c)
                    #print(G)
                    #print(P)
                    #print(T)
                    ret = [auc(T,P),acc(T,G),recall(T,G)]
                    if ret[0]>best_auc:
                        torch.save(model.state_dict(), model_file_name)
                        with open(result_file_name,'w') as f:
                            f.write(','.join(map(str,ret)))
                        best_epoch = epoch+1
                        best_auc = ret[0]
                        best_acc = ret[1]
                        best_recall = ret[2]
                        print('rmse improved at epoch ', best_epoch, '; best_auc,best_acc, best_recall:', best_auc,best_acc, best_recall,dataset)
                    else:
                        print(ret[1],'No improvement since epoch ', best_epoch, ':best_auc,best_acc, best_recal:', best_auc,best_acc, best_recall,model_st,dataset)

