import numpy as np
import scipy.sparse as sp
import torch

# positive samples in the test set to zero
def Preproces_Data(A, test_id):
    copy_A = A / 1
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

# construct topology graph
def construct_structure_graph(lncRNA_disease,  miRNA_disease, lncRNA_miRNA, lncRNA_sim, miRNA_sim, disease_sim ):
    lnc_dis_sim = np.hstack((lncRNA_sim, lncRNA_disease, lncRNA_miRNA))
    # print(lnc_dis_sim.shape)

    dis_lnc_sim = np.hstack((lncRNA_disease.T, disease_sim, miRNA_disease.T))
    # print(dis_lnc_sim.shape)

    mi_lnc_dis = np.hstack((lncRNA_miRNA.T, miRNA_disease, miRNA_sim))
    # print(mi_lnc_dis.shape)

    matrix_A = np.vstack((lnc_dis_sim,dis_lnc_sim,mi_lnc_dis))
    # print(matrix_A.shape)
    return matrix_A

# construct feature graph
def construct_feature_graph(lncRNA, miRNA, disease):
    lnc_shape = lncRNA.shape[0]
    mi_shape = miRNA.shape[0]
    dis_shape = disease.shape[0]
    lnc_dis_mi = np.hstack((lncRNA, np.zeros((lnc_shape,dis_shape)), np.zeros((lnc_shape,mi_shape))))
    dis_lnc_mi = np.hstack((np.zeros((dis_shape,lnc_shape)), disease, np.zeros((dis_shape,mi_shape))))
    mi_lnc_dis = np.hstack((np.zeros((mi_shape,lnc_shape)), np.zeros((mi_shape,dis_shape)), miRNA))
    matrix_A = np.vstack((lnc_dis_mi, dis_lnc_mi, mi_lnc_dis))
    return matrix_A

def laplacian_norm(adj):
    adj += np.eye(adj.shape[0])   # add self-loop
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)

    return norm_A

def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

