import pandas as pd
import numpy as np

"This code include: k-mer feature calculation and construct kNN graph"

def k_mer(seq):
    def get_1mer(seq):
        A_count = seq.count("A")
        T_count = seq.count("T")
        C_count = seq.count("C")
        G_count = seq.count("G")
        return [A_count/len(seq), T_count/len(seq), C_count/len(seq), G_count/len(seq)]

    def get_2mer(seq):
        res_dict = {}
        for x in "ATCG":
            for y in "ATCG":
                k = x + y
                res_dict[k] = 0
                # print(k)
        # print(res_dict)
        i = 0
        while i + 2 < len(seq):
            k = seq[i:i + 2]
            i = i + 1
            res_dict[k] = res_dict[k] + 1

        return [x/len(seq) for x in list(res_dict.values())]

    def get_3mer(seq):
        res_dict = {}
        for x in "ATCG":
            for y in "ATCG":
                for z in "ATCG":
                    k = x + y + z
                    res_dict[k] = 0
        i = 0
        while i + 3 < len(seq):
            k = seq[i:i + 3]
            i = i + 1
            res_dict[k] = res_dict[k] + 1
        return [x/len(seq) for x in list(res_dict.values())]

    def get_4mer(seq):
        res_dict = {}
        for x in "ATCG":
            for y in "ATCG":
                for z in "ATCG":
                    for p in "ATCG":
                        k = x + y + z + p
                        res_dict[k] = 0
        i = 0
        while i + 4 < len(seq):
            k = seq[i:i + 4]
            i = i + 1
            res_dict[k] = res_dict[k] + 1
        return [x/len(seq) for x in list(res_dict.values())]

    # return get_1mer(seq) + get_2mer(seq) + get_3mer(seq) + get_4mer(seq)
    return get_1mer(seq) + get_2mer(seq) + get_3mer(seq)

def lncRNA_mer():
    "read data"
    df = pd.read_excel('dataset1/lncRNA_sequences2.xlsx', usecols=['lncRNA_name', 'sequence'])
    # df = pd.read_excel('dataset2/lncRNA_sequences.xlsx', usecols=['lncRNA_name', 'sequence'])
    df['sequence'] = df['sequence'].str.replace('U', 'T')
    lncRNA_dict = dict(zip(df['lncRNA_name'], df['sequence']))

    result = []
    for name, seq in lncRNA_dict.items():
        RNA_mer = k_mer(seq)
        result.append(RNA_mer)

    print(len(result))
    np.savetxt("dataset1/lncRNA_mer_feature.txt",result)
    # np.savetxt("dataset2/lncRNA_mer_feature.txt",result)

def miRNA_mer():
    "read data"
    df = pd.read_excel('dataset1/miRNA_sequences2.xlsx', usecols=['miRNA_name', 'Sequence'])
    # df = pd.read_excel('dataset2/miRNA_sequences.xlsx', usecols=['miRNA_name', 'Sequence'])
    df['Sequence'] = df['Sequence'].str.replace('U', 'T')
    miRNA_dict = dict(zip(df['miRNA_name'], df['Sequence']))

    result = []
    for name, seq in miRNA_dict.items():
        RNA_mer = k_mer(seq)
        result.append(RNA_mer)

    print(len(result))
    np.savetxt("dataset1/miRNA_mer_feature.txt",result)
    # np.savetxt("dataset2/miRNA_mer_feature.txt",result)

def cosine_similarity(features):
    sim_matrix = np.zeros((len(features), len(features)))  # Initialize similarity matrix

    for i in range(len(features)):
        for j in range(i, len(features)):
            v1 = features[i]
            v2 = features[j]

            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    # Output similarity matrix
    print(sim_matrix,sim_matrix.shape)
    return sim_matrix

def construct_knn_graph(sim_matrix):
    num_nodes = sim_matrix.shape[0]
    knn_matrix = np.zeros((num_nodes, num_nodes))

    top_k = 22   # top-k nodes
    for i in range(num_nodes):
        sorted_indices = np.argsort(-sim_matrix[i, :])[:top_k]

        for j in sorted_indices:
            knn_matrix[i][j] = 1
            knn_matrix[j][i] = 1
    # print(knn_matrix.shape, np.sum(knn_matrix))

    knn_matrix = knn_matrix - np.diag(np.diag(knn_matrix))
    print(knn_matrix.shape, np.sum(knn_matrix))

    return knn_matrix


if __name__ == '__main__':

    "k-mer feature"
    # lncRNA_mer()
    miRNA_mer()

    "construct knn graph"
    # lncRNA attribute graph
    # lnc_features = np.loadtxt('dataset1/lncRNA_mer_feature.txt')
    # lnc_seq_sim = cosine_similarity(lnc_features)
    # lnc_att_graph = construct_knn_graph(lnc_seq_sim)
    # np.savetxt("dataset1/lnc_att_graph.txt", lnc_att_graph, fmt="%d")

    # miRNA attribute graph
    # mi_features = np.loadtxt('dataset1/miRNA_mer_feature.txt')
    # mi_seq_sim = cosine_similarity(mi_features)
    # mi_att_graph = construct_knn_graph(mi_seq_sim)
    # np.savetxt("dataset1/mi_att_graph.txt",mi_att_graph,fmt="%d")

    # disease attribute graph
    # dis_sim_matrix = np.loadtxt("dataset1/dis_sem_sim.txt")
    # dis_att_graph = construct_knn_graph(dis_sim_matrix)
    # np.savetxt("dataset1/dis_att_graph.txt",dis_att_graph,fmt="%d")

