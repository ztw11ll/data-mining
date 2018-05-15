# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

def _hill_climb(x_t, X, W=None, h=0.1, eps=1e-7):
    error = 99.
    prob = 0.
    x_l1 = np.copy(x_t)
    
    radius_new = 0.
    radius_old = 0.
    radius_twiceold = 0.
    iters = 0.
    while True:
        radius_thriceold = radius_twiceold
        radius_twiceold = radius_old
        radius_old = radius_new
        x_l0 = np.copy(x_l1)
        x_l1, density = _step(x_l0, X, W=W, h=h)
        error = density - prob
        prob = density
        radius_new = np.linalg.norm(x_l1-x_l0) #求范数
        radius = radius_thriceold + radius_twiceold + radius_old + radius_new
        iters += 1
        if iters>3 and error < eps:
            break
    return [x_l1, prob, radius]

def _step(x_l0, X, W=None, h=0.1):
    n = X.shape[0]#行，周围点的个数
    d = X.shape[1]#列，维度
    superweight = 0. #周围点对x的影响总和
    x_l1 = np.zeros((1,d))
    if W is None:
        W = np.ones((n,1))
    else:
        W = W
    for j in range(n):
        kernel = kernelize(x_l0, X[j], h, d) 
        kernel = kernel * W[j]/(h**d)  #计算贡献率
        superweight = superweight + kernel
        x_l1 = x_l1 + (kernel * X[j])
    x_l1 = x_l1/superweight
    density = superweight/np.sum(W) #核密度值
    return [x_l1, density]
    
def kernelize(x, y, h, degree): #高斯核函数
    kernel = np.exp(-(np.linalg.norm(x-y)/h)**2./2.)/((2.*np.pi)**(degree/2))
    return kernel

def DENCLUE(X,eps,min_density,y=None):
    if not  eps > 0.0:
        raise ValueError("收敛容限应大于0")
    n_samples = X.shape[0] #样本数量
    n_features = X.shape[1] #样本维度
    density_attractors = np.zeros((n_samples,n_features)) #密度吸引子集合
    radii = np.zeros((n_samples,1)) #半径
    density = np.zeros((n_samples,1)) #核密度
        
    #初始化所有标签为噪声-1
    labels =-np.ones(X.shape[0])
    h = np.std(X)/5.5
    sample_weight = np.ones((  n_samples,1))
    
    #爬山寻找密度吸引子
    for i in range(n_samples):
        density_attractors[i], density[i], radii[i] = _hill_climb(X[i], X, W=sample_weight,h= h, eps= eps)

    cluster_info = {}
    num_clusters = 0
    cluster_info[num_clusters]={'instances': [0],'centroid': np.atleast_2d(density_attractors[0])}
    g_clusters = nx.Graph()#生成聚类图
    for j1 in range(n_samples):
        g_clusters.add_node(j1,attractor=density_attractors[j1], radius=radii[j1],density=density[j1])
   
    #扩充聚类集合
    for j1 in range(n_samples):
        for j2 in (x for x in range(n_samples) if x != j1):
            if g_clusters.has_edge(j1,j2):
                continue
            diff = np.linalg.norm(g_clusters.node[j1]['attractor']-g_clusters.node[j2]['attractor'])
            if diff <= (g_clusters.node[j1]['radius']+g_clusters.node[j1]['radius']):
                g_clusters.add_edge(j1, j2)
                     
    #转换为集群
    clusters = list(nx.connected_component_subgraphs(g_clusters))
    num_clusters = 0
        
    #连接所有集群
    for clust in clusters:
            
        #获得吸引子的最大密度和位置
        max_instance = max(clust, key=lambda x: clust.node[x]['density'])
        max_density = clust.node[max_instance]['density']
        max_centroid = clust.node[max_instance]['attractor']
                
        c_size = len(clust.nodes())
        
        #集合信息
        cluster_info[num_clusters] = {'instances': clust.nodes(),
                    'size': c_size,'pury':float(1-abs(c_size-ids[num_clusters])/c_size),
                    'centroid': max_centroid,
                    'density': max_density
                    }
            
            #如果簇密度不高于最小值，则实例被分类为噪声。
        if max_density >= min_density:
            labels[clust.nodes()]=num_clusters #族的编号           
        print(cluster_info[num_clusters])
        num_clusters += 1
    clust_info_ = cluster_info
    print(labels)
    return  clust_info_
result=[]
result2=[]
with open('iris.txt','r') as f:  #读取文件
    for line in f:
        line=list(map(str,line.split(','))) #逗号分割
        result2.append(line[4]) 
        result.append(list(map(float,line[:4]))) #去除字母
        
D=np.array(result) #原数据
ids =[0,0,0]#统计已知三类标签数量
ids[0]=result2.count(result2[0])
ids[1]=result2.count(result2[3])
ids[2]=result2.count(result2[6])


DENCLUE(D,0.0001,0.03)

