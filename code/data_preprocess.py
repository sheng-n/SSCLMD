from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import torch
import scipy.sparse as sp

"This code uses one-fold cross-validation as an example "
class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.label = triple[:, 2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return self.label[index], (self.entity1[index], self.entity2[index])


def load_data(args, test_ratio=0.2):
    """Read data from path, convert data into loader, return features and adjacency"""
    # read data
    print('Loading {0} seed{1} dataset...'.format(args.in_file, args.seed))  #
    positive = np.loadtxt(args.in_file, dtype=np.int64)
    # print("postive:",positive.shape)

    # sample postive
    link_size = int(positive.shape[0])
    # print("link_size: ",link_size)
    np.random.seed(args.seed)
    np.random.shuffle(positive)
    positive = positive[:link_size]
    # print("postive:",positive.shape)

    # sample negative
    negative_all = np.loadtxt(args.neg_sample, dtype=np.int64)
    # print("negative:", negative_all.shape)#
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:positive.shape[0]])
    print("positve examples: %d, negative examples: %d." % (positive.shape[0], negative.shape[0]))


    test_size = int(test_ratio * positive.shape[0])
    # print("positive_data:",test_size)

    positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)], axis=1)
    negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)], axis=1)
    # print("positive_negative: ",positive.shape, negative.shape)
    # extract features
    train_data = np.vstack((positive[: -test_size], negative[: -test_size]))
    test_data = np.vstack((positive[-test_size:], negative[-test_size:]))

    # construct adjacency
    train_positive = positive[: -test_size]
    # print("train_positive: ",train_positive)
    if args.task_type == 'LDA':

        l_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),

                            shape=(386, 316), dtype=np.float32)
        # print("adj: ",np.sum(l_d),l_d)
        l_d = l_d.toarray()
        m_d = np.loadtxt("dataset1/mi_dis_association_new2.txt")
        l_m = np.loadtxt("dataset1/lnc_mi_interaction_new2.txt")

        a = 0.9
        b = 0.6
        l_l_t = np.loadtxt("dataset1/one_hot_lnc_sim_"+ str(a) + str(b) +".txt")
        m_m_t = np.loadtxt("dataset1/one_hot_mi_sim_"+ str(a) + str(b) + ".txt")
        d_d_t = np.loadtxt("dataset1/one_hot_dis_sim_" + str(a) + str(b) +".txt")


        l_l_f = np.loadtxt("dataset1/lnc_att_graph.txt")  # 230, 230
        m_m_f = np.loadtxt("dataset1/mi_att_graph.txt")   # 436, 436
        d_d_f = np.loadtxt("dataset1/dis_att_graph.txt")  # 405, 405


    if args.task_type == 'MDA':
        m_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                            shape=(295, 316), dtype=np.float32)
        # print("adj: ",np.sum(l_d),l_d)
        m_d = m_d.toarray()
        l_d = np.loadtxt("dataset1/lnc_dis_association_new2.txt")
        l_m = np.loadtxt("dataset1/lnc_mi_interaction_new2.txt")

        a = 0.9
        b = 0.5
        l_l_t = np.loadtxt("dataset1/one_hot_lnc_sim_"+ str(a) + str(b) +".txt")
        m_m_t = np.loadtxt("dataset1/one_hot_mi_sim_"+ str(a) + str(b) + ".txt")
        d_d_t = np.loadtxt("dataset1/one_hot_dis_sim_" + str(a) + str(b) +".txt")

        l_l_f = np.loadtxt("dataset1/lnc_att_graph.txt")  # 230, 230
        m_m_f = np.loadtxt("dataset1/mi_att_graph.txt")   # 436, 436
        d_d_f = np.loadtxt("dataset1/dis_att_graph.txt")  # 405, 405

    if args.task_type == 'LMI':
        l_m = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                            shape=(386, 295), dtype=np.float32)

        # print("adj: ",np.sum(l_d),l_d)
        l_m = l_m.toarray()
        l_d = np.loadtxt("dataset1/lnc_dis_association_new2.txt")
        m_d = np.loadtxt("dataset1/mi_dis_association_new2.txt")

        a = 0.5
        b = 0.5
        l_l_t = np.loadtxt("dataset1/one_hot_lnc_sim_"+ str(a) + str(b) +".txt")  # 230, 230
        m_m_t = np.loadtxt("dataset1/one_hot_mi_sim_"+ str(a) + str(b) + ".txt")  # 436, 436
        d_d_t = np.loadtxt("dataset1/one_hot_dis_sim_" + str(a) + str(b) +".txt")  # 405, 405

        l_l_f = np.loadtxt("dataset1/lnc_att_graph.txt")  # 230, 230
        m_m_f = np.loadtxt("dataset1/mi_att_graph.txt")   # 436, 436
        d_d_f = np.loadtxt("dataset1/dis_att_graph.txt")  # 405, 405

    print(l_d.shape, m_d.shape, l_m.shape, l_l_t.shape, d_d_t.shape, m_m_t.shape,l_l_f.shape,m_m_f.shape,d_d_f.shape)

    # Constructing topology and attribute graph adjacency matrices
    s_adj = construct_structure_graph(l_d, m_d, l_m, l_l_t, m_m_t, d_d_t)
    f_adj = construct_feature_graph(l_l_f, m_m_f, d_d_f)
    # print("adj: ", s_adj.shape, f_adj.shape, node_feature.shape)

    s_adj = laplacian_norm(s_adj)
    f_adj = laplacian_norm(f_adj)

    # Generating feature matrix
    np.random.seed(args.seed)
    features = np.random.normal(loc=0, scale=1, size=(s_adj.shape[0], args.dimensions))
    node_feature = row_normalize(features)

    # adversarial nodes
    np.random.seed(args.seed)
    id = np.arange(node_feature.shape[0])
    id = np.random.permutation(id)
    shuf_feature = node_feature[id]
    # print("adj: ", s_adj.shape, f_adj.shape, node_feature.shape)

    # build data loader
    params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.workers, 'drop_last': True}

    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)
    print(train_loader)

    test_set = Data_class(test_data)
    test_loader = DataLoader(test_set, **params)
    print(test_loader)

    # construct edges
    edges_s = s_adj.nonzero()
    edge_index_s = torch.tensor(np.vstack((edges_s[0], edges_s[1])), dtype=torch.long)

    edges_f = f_adj.nonzero()
    edge_index_f = torch.tensor(np.vstack((edges_f[0], edges_f[1])), dtype=torch.long)

    x = torch.tensor(node_feature, dtype = torch.float)   # original feature
    shuf_feature = torch.tensor(shuf_feature, dtype=torch.float)  # shuffle feature

    data_s = Data(x = x, edge_index = edge_index_s)
    data_f = Data(x = shuf_feature, edge_index = edge_index_f)
    print("data_s: ", data_s)

    print('Loading finished!')
    return data_s, data_f, train_loader, test_loader  # topology graph, feature graph，train dataset, test dataset
