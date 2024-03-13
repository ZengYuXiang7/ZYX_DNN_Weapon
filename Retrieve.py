# coding : utf-8
# Author : yuxiang Zeng

class LSH:
    def __init__(self, k, d, nbits):
        self.d = d
        self.k = k
        self.nbits = nbits
        self.index = faiss.IndexLSH(d, nbits)
        self.first = False

    def search_topk_embeds(self, data_base, h_query):
        if not self.first:
            self.index.train(data_base)
            self.index.add(data_base)
            self.first = True
        topk_distence, topk_indices = self.index.search(h_query, self.k)
        return topk_distence, topk_indices


if __name__ == '__main__':
    import numpy as np
    import faiss  # 导入faiss

    # 配置
    d = 64  # 向量维度
    nb = 10000  # 数据库大小
    nq = 100  # 查询向量的数量
    np.random.seed(1234)  # 保证可复现的结果

    # 生成一些随机样本向量
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    lsh = LSH(k=3, d=d, nbits=d)
    topk_distence, topk_indices = lsh.search_topk_embeds(xb, xq)

    for i in range(len(topk_distence)):
        print(f"{topk_indices[i]} {topk_distence[i]}")
