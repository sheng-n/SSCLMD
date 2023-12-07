import numpy as np
import copy
import pandas as pd

'''Calculate the intra-edges in the topology graph '''

# positive samples in the test set to zero
def Preproces_Data(A, test_id):
    copy_A = A / 1
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

# GIPK
def calculate_kernel_bandwidth(A):
    IP_0 = 0
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))
        # print(IP)
        IP_0 += IP
    lambd = 1/((1/A.shape[0]) * IP_0)
    return lambd

def calculate_GaussianKernel_sim(A):
    kernel_bandwidth = calculate_kernel_bandwidth(A)
    gauss_kernel_sim = np.zeros((A.shape[0],A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel
            # print("gau",gauss_kernel_sim)
    return gauss_kernel_sim

# threshold function
def label_preprocess(sim_matrix, b):
    new_sim_matrix = np.zeros(shape=sim_matrix.shape)
    # print(lnc_sim_matrix.shape)
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i][j] >= b:
                new_sim_matrix[i][j] = 1

    return new_sim_matrix

def fuse_similarity(sim1, sim2, alpha, theta):
    fused_sim = alpha * sim1 + (1-alpha) * sim2
    fused_sim = label_preprocess(fused_sim, theta)
    return fused_sim

if __name__ == '__main__':

    a = 0.9  # similarity integration coefficient 𝛼  LDA/MDA=0.9; LMI=0.5
    b = 0.6  # intra-graph similarity threshold 𝜃. LDA=0.6; MDA/LMI=0.5

    'dataset1'
    lnc_dis = np.loadtxt("dataset1/lnc_dis_association_new2.txt")  # 240,405,2687
    mi_dis = np.loadtxt("dataset1/mi_dis_association_new2.txt")  # 495,405,13559
    lnc_mi = np.loadtxt("dataset1/lnc_mi_interaction_new2.txt")  # 240,495,1002
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")  # 405,405
    print(lnc_dis.shape,mi_dis.shape,lnc_mi.shape,dis_sem_sim.shape)

    'dataset2'
    # lnc_dis = np.loadtxt("dataset2/lnc_dis_association_new.txt")  # 386,316
    # mi_dis = np.loadtxt("dataset2/mi_dis_association_new.txt")  # 295,316
    # lnc_mi = np.loadtxt("dataset2/lnc_mi_interaction_new.txt")  # 386,295
    # dis_sem_sim = np.loadtxt("dataset2/dis_sem_sim.txt")  # 316,316
    # print(lnc_dis.shape,mi_dis.shape,lnc_mi.shape,dis_sem_sim.shape)

    "lncRNA GIPK"
    lnc_gau_1 = calculate_GaussianKernel_sim(lnc_dis)  # based lncRNA-disease association
    lnc_gau_2 = calculate_GaussianKernel_sim(lnc_mi)   # based lncRNA-miRNA interaction
    lnc_sim = fuse_similarity(lnc_gau_1, lnc_gau_2, a, b)
    np.savetxt("dataset1/one_hot_lnc_sim_" + str(a) + str(b) + ".txt", lnc_sim)
    # print(lnc_gau_1.shape,lnc_gau_2.shape, lnc_sim.shape)
    #
    "miRNA GIPK"
    mi_gau_1 = calculate_GaussianKernel_sim(mi_dis)     # based miRNA-disease association
    mi_gau_2 = calculate_GaussianKernel_sim(lnc_mi.T)   # based lncRNA-miRNA interaction
    mi_sim = fuse_similarity(mi_gau_1, mi_gau_2, a, b)

    np.savetxt("dataset1/one_hot_mi_sim_"+str(a) + str(b) +".txt", mi_sim)
    # print(mi_gau_1.shape,mi_gau_2.shape,mi_sim.shape)

    "disease GIPK"
    dis_gau_1 = calculate_GaussianKernel_sim(lnc_dis.T)  # based lncRNA-disease association
    dis_gau_2 = calculate_GaussianKernel_sim(mi_dis.T)   # based miRNA-disease association
    dis_sim = fuse_similarity(dis_gau_1, dis_gau_2, a, b)
    np.savetxt("dataset1/one_hot_dis_sim_" + str(a) + str(b) + ".txt", dis_sim)









