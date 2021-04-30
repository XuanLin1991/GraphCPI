import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
from keras.preprocessing import text, sequence


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]

    return x  

def split_ngrams(seq):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    a, b, c = zip(*[iter(seq)]*3), zip(*[iter(seq[1:])]*3), zip(*[iter(seq[2:])]*3)
    str_ngrams = ""
    for ngrams in [a,b,c]:
        x = ""
        for ngram in ngrams:
            x +="".join(ngram) + " "
        str_ngrams += x + " "
    return str_ngrams


#词向量
def w2v_pad(df_train,df_test,col, maxlen_,victor_size):
    
    #keras API把数据映射成向量并填充或截断
    tokenizer = text.Tokenizer(num_words=10000, lower=False,filters="")
    tokenizer.fit_on_texts(list(df_train[col].apply(split_ngrams).values) + list(df_test[col].apply(split_ngrams).values))
    train_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_train[col].apply(split_ngrams).values), maxlen=maxlen_)
    test_ = sequence.pad_sequences(tokenizer.texts_to_sequences(df_test[col].apply(split_ngrams).values), maxlen=maxlen_)

    word_index = tokenizer.word_index
    nb_words = len(word_index)
    print(nb_words)
    print(nb_words)
    # 对词向量做处理,把它做出字典形式
    w2v_model = {}
    #with open("./data/embed/protVec_100d_3grams.csv",encoding='utf8') as f:
    with open("data/embed/protVec_100d_3grams.csv",encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            w2v_model[word] = coefs
    print("add w2v finished....")  


    count=0
    embedding_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_vector=w2v_model[word] if word in w2v_model else None
        if embedding_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec
            
    return train_, test_, embedding_matrix

    del protVec_model
    print(embedding_matrix.shape)
    return data_, embedding_matrix

def list_to_dataframe(list_data, data):
    for i in range(len(list_data)):
        data[i] = str(list_data[i])
        data[i] = data[i].lstrip('[').rstrip(']')
        
        
        
# from DeepDTA data
all_prots = []
datasets = ['human','celegans']
negative_ratios = ['1','3','5']
for dataset in datasets:
    for negative_ratio in negative_ratios:
        print('convert data from DeepDTA for ', dataset, negative_ratio)
        fpath = './data/' + dataset + '/' + negative_ratio + '_'
        data = pd.read_csv(fpath + 'data.csv')
        ligands = data['drug']
        proteins = data['target']
        affinity = data['interaction']
        drugs = []
        prots = []
        for d in ligands:
            drugs.append(d)
        for t in proteins:
            prots.append(t)
        affinity = np.asarray(affinity)
        train, test = train_test_split(data,test_size=1/6, random_state=42)
        train_, test_, embedding_matrix = w2v_pad(train,test,'target', 1000,100)
        np.save(fpath+dataset+"_train.npy", train_)
        np.save(fpath+dataset+"_test.npy", test_)
        del train_, test_
         #把embedding matrix保存为csv文件
        pro_embedding_matrix = pd.DataFrame(embedding_matrix)
        pro_embedding_matrix.to_csv('./data/embed/'+negative_ratio+'_'+dataset+'_'+'embedding_matrix.csv', index=False)
        train.to_csv(fpath + dataset + "_train.csv", index=False)
        test.to_csv(fpath + dataset + "_test.csv", index=False)
        drugs = set(drugs)
        prots = set(prots)
        print('\ndataset:', dataset)
        print('train_fold:', len(train))
        print('test_fold:', len(test))
        print('len(set(drugs)),len(set(prots)):', len(set(drugs)),len(set(prots)))
        all_prots += list(set(prots))

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:i for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

# compound_iso_smiles = []
# for dt_name in ['human','celegans']:
#     df = pd.read_csv('data/' + dt_name + '/5_data.csv')
#     compound_iso_smiles += list( df['drug'] )
# compound_iso_smiles = set(compound_iso_smiles)
# smile_graph = {}
# for smile in compound_iso_smiles:
#     g = smile_to_graph(smile)
#     smile_graph[smile] = g

datasets = ['human', 'celegans']
# convert to PyTorch data format
for dataset in datasets:
    for negative_ratio in negative_ratios:
        print(dataset + negative_ratio)
        compound_iso_smiles = []
        df = pd.read_csv('data/' + dataset + '/'+negative_ratio +'_data.csv')
        compound_iso_smiles += list( df['drug'] )
        compound_iso_smiles = set(compound_iso_smiles)
        smile_graph = {}
        for smile in compound_iso_smiles:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        processed_data_file_train = 'data/processed/' + negative_ratio+ dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + negative_ratio + dataset + '_test.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            df = pd.read_csv('data/' + dataset + '/' + negative_ratio + '_' + dataset + '_train.csv')
            train_drugs, train_prots, train_Y = list(df['drug']),list(df['target']),list(df['interaction'])
            XT = [seq_cat(t) for t in train_prots]
            train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)
            df = pd.read_csv('data/' + dataset + '/' + negative_ratio + '_' + dataset + '_test.csv')
            test_drugs, test_prots, test_Y = list(df['drug']),list(df['target']),list(df['interaction'])
            XT = [seq_cat(t) for t in test_prots]
            test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y)
            path = './data/'+ dataset +'/' + negative_ratio+'_'+dataset
            # make data PyTorch Geometric ready
            print('preparing ', dataset + '_train.pt in pytorch format!')
            train_data = TestbedDataset(root='data', dataset=negative_ratio+'_'+dataset+'_train', xd=train_drugs, xt=train_prots, filename= path + '_train',y=train_Y,smile_graph=smile_graph)
            print('preparing ', dataset + '_test.pt in pytorch format!')
            test_data = TestbedDataset(root='data', dataset=negative_ratio+'_'+dataset+'_test', xd=test_drugs, xt=test_prots, filename= path + '_test',y=test_Y,smile_graph=smile_graph)
            print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')        
        else:
            print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')       
