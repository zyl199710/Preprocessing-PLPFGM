#!/usr/bin/env python
# coding: utf-8

# In[4]:

import itertools
import networkx as nx
import numpy as np
import pandas as pd

import random
# import cugraph as cnx


# # 构建边的字典

# In[6]:
def sample_with_RWR_and_three_degree_infuence(K_node,net):
    #重启随机游走算法（RWR）
    # 基于重启随机游走一次采样
    N = 800
    re_p = 0.2
    RWR_node = [K_node]
    switch_node = K_node
    print(float(1 - re_p))
    print(len(list(net.neighbors(switch_node))))
    while (True):
        p = float(1 - re_p) / (len(list(net.neighbors(switch_node))) + float("1e-8"))
        choice_list = list(net.neighbors(switch_node))
        choice_list.append(K_node)
        choice_probability = [p] * len(list(net.neighbors(switch_node)))
        choice_probability.append(re_p)
        # 随机数
        random_number = random.uniform(0, 1)
        for i in range(0, len(choice_list)):
            if random_number < sum(choice_probability[0:i + 1]):
                switch_node = choice_list[i]
                break
        if switch_node not in RWR_node:
            RWR_node.append(switch_node)
        if len(RWR_node) >= N:
            break
        # 基于三度影响力二次采样
    nodes = list(set(net.nodes)-set(RWR_node))
    net.remove_nodes_from(nodes)

    # for node_1 in net.neighbors(K_node):
    #     if node_1 in RWR_node:
    #         item = K_node, node_1
    #         edges.append(item)
    #     for node_2 in net.neighbors(node_1):
    #         if node_2 in RWR_node:
    #             item = node_1, node_2
    #             edges.append(item)
    #         for node_3 in net.neighbors(node_2):
    #             if node_3 in RWR_node:
    #                 item = node_2, node_3
    #                 edges.append(item)
    # edges = list(set(edges))
   # sub_G.add_edges_from(edges)
    return net


def preprocess(filename, pagerank_alpha, max_depth, start_vertices):
    net = nx.DiGraph()
    net1 = nx.Graph()
    edge_net = nx.Graph()
    edges = []
    id = 0
    '''
    construct origin net
    '''
    f = open(filename, 'r')

    num_trust = 0
    num_untrust = 0

    for line in f:
        line = str(line)
        line = line.strip()
        if not line:
            break

        tokens = line.split()  # split a line to a list of tokens

        if '#' in tokens:  # pass the comment lines
            continue
            '''
            if tokens[0] not in net.nodes:
                net.add_node(tokens[0],edge_of_node=[id])
            else:
                edge_of_nodes=list(net.nodes[tokens[0]]['edge_of_node'])
                edge_of_nodes.append(id)
                net.add_node(tokens[0],edge_of_node=edge_of_nodes)
            if tokens[1] not in net.nodes:
                net.add_node(tokens[1],edge_of_node=[id])
            else:
                edge_of_nodes=list(net.nodes[tokens[1]]['edge_of_node'])
                edge_of_nodes.append(id)
                net.add_node(tokens[1],edge_of_node=edge_of_nodes)
            '''
        net.add_edge(int(tokens[0]), int(tokens[1]), label=tokens[2])
        net1.add_edge(tokens[0], tokens[1])
    #    if tokens[2]=='1':
    #        num_trust+=1
    #    elif tokens[2]=='-1':
    #        num_untrust+=1
    f.close()
    '''
    final_edge=0

    remove_list=[]
    if num_trust>num_untrust:
        gap=num_trust-num_untrust
        for edge in net.edges:
            if gap==0:break
            if net[edge[0]][edge[1]]['label']=='1':
                remove_list.append(edge)
                gap-=1


    elif num_trust<num_untrust:
        gap=num_untrust-num_trust
        for edge in net.edges:
            if gap==0:break
            if net[edge[0]][edge[1]]['label']=='-1':
                remove_list.append(edge)
                gap-=1



    for edge in remove_list:
        net.remove_edge(edge[0],edge[1])
        if net1.has_edge(edge[0],edge[1]):net1.remove_edge(edge[0],edge[1])

    conn=max(nx.connected_components(net1),key=len)
    net=net.subgraph(conn)      #get the max connected component    
    '''
  #  indices,_,_=nx.gnp_random_graph(G=net,max_depth=max_depth,start_vertices=start_vertices)
  #  indices=indices.to_pandas()
  #  indices=set(indices)
    #print(len(list(net.neighbors(5))))
    net=sample_with_RWR_and_three_degree_infuence(5, net)
    print(nx.number_of_nodes(net))
    print(nx.number_of_edges(net))
    pagerank = nx.pagerank(net, alpha=pagerank_alpha)
    edge_net = nx.Graph()
    features = []
    labels = []
    d = {}
    orgedges = []
    for edge in net.edges:
        start = edge[0]
        end = edge[1]
        net[edge[0]][edge[1]]['id'] = id
        PR1 = pagerank[start]
        PR2 = pagerank[end]
        comm_neigh = len(set(net.neighbors(start)) & set(net.neighbors(end)))
        feature = [net.in_degree[start], net.in_degree[end], net.out_degree[start], net.out_degree[end],
                   net.degree[start], net.degree[end], comm_neigh, PR1, PR2]
        features.append(feature)
        tuple = edge[0], edge[1]
        edge_net.add_node(id, start_node=edge[0], end_node=edge[1], label=net[edge[0]][edge[1]]['label'])
        d[tuple] = id
        id += 1
        if net[edge[0]][edge[1]]['label'] == '1':
            num_trust += 1
            labels.append(1)
        elif net[edge[0]][edge[1]]['label'] == '-1':
            num_untrust += 1
            labels.append(0)
    filename = filename.split('.')
    #np.save('dataset/Epinion/feature-epinions.npy', features)
    print('finish feature')
    #   for edge in
    #   matrix=nx.adjacency_matrix(edge_net,dtype=np.bool)
    #    matrix=matrix.todense()
    #  np.save('Epinion/edge-epinion.npy',matrix)

    #  edges = []
    # ll = []
    # for edge in net.edges:
    #     tuple = edge[0], edge[1]
    #     ll.append(tuple)

    for edge in net.edges:
        l1 = list(net.neighbors(edge[0]))
        l2 = list(net.neighbors(edge[1]))
        for l11 in l1:
            if l11 != edge[1]:
                tuple1 = l11,edge[1]
                tuple2 = edge[1],l11
                if tuple1 in d:
                    tt = d[tuple1], d[edge]

                    edges.append(tt)
                if tuple2 in d:
                    tt = d[tuple2], d[edge]

                    edges.append(tt)
        for l22 in l2:
            if l22 != edge[0]:
                tuple1 = l11,edge[0]
                tuple2 = edge[0],l11
                if tuple1 in d:
                    tt = d[tuple1], d[edge]

                    edges.append(tt)
                if tuple2 in d:
                    tt = d[tuple2], d[edge]

                    edges.append(tt)
    # for i in range(id):
    #     for j in range(i + 1, id):
    #         l1 = set(ll[i])
    #         l2 = set(ll[j])
    #         l = l1.intersection(l2)
    #         if len(l) == 1:
    #             tuple = i, j
    #             if tuple not in edges:
    #                 edges.append(tuple)

    #np.save('dataset/Epinion/edge-epinions.npy', edges)
    #np.save('dataset/Epinion/label-epinions.npy', labels)
    print(num_trust, num_untrust)
    print(len(edges))
    print(len(labels))

    return edges, labels, features, net
    '''
    nx.number_connected_components(net1)

    print(len(net.nodes))
    print(len(edges))
    '''

    '''
    for edge in edges:
        start=edge_net.nodes[id]['start']
        end=edge_net.nodes[id]['end']
        #print(start,' ',end)
        node1=net.nodes[start]
        node2=net.nodes[end]
        for edge1 in list(node1['edge_of_node']):
            edge_net.add_edge(id,edge1)
        for edge2 in list(node2['edge_of_node']):
            edge_net.add_edge(id,edge2)
        id+=1

    id=0
    features=[]
    page_rank=nx.pagerank(net,alpha=pagerank_alpha)
    for edge in edges:
        start=edge_net.nodes[id]['start']
        end=edge_net.nodes[id]['end']
        d_in1=net.in_degree[start]
        d_in2=net.in_degree[end]
        d_out1=net.out_degree[start]
        d_out2=net.out_degree[end]
        d_tot1=d_in1+d_out1
        d_tot2=d_in2+d_out2
        comm_ne=len(set(nx.neighbors(net,start))&set(nx.neighbors(net,end)))
        PR1=page_rank[start]
        PR2=page_rank[end]
        features.append([d_in1,d_in2,d_out1,d_out2,d_tot1,d_tot2,comm_ne,PR1,PR2])
        id+=1

    print(features[:1])

    np.save('feature.npy',features)

    np.save('adjacency.npy',np.array(nx.adjacency_matrix(edge_net).todense()))
    '''
def xiaoqi():
    edges = np.load('dataset/epinion/edge-epinions.npy')
    G = nx.Graph()
    G.add_edges_from(edges)
    ll = []
    ll = list(G.edges)
    print(len(ll))
    np.save('dataset/epinion/edge-epinions1.npy', ll)

# In[7]:
def crf_data():
    edges = np.load('dataset/epinion/edge-epinions.npy')
    labels = np.load('dataset/epinion/label-epinions.npy')
    features = np.load('dataset/epinion/feature-epinions.npy')
    print(len(labels))
    max1 = 0
    min1 = 0
    feature1 =[]
    feature2 = []
    feature3 = []
    feature4 = []
    feature5 = []
    feature6 = []
    feature7 = []
    label = []
    for i in range(len(features)):

        feature1.append(features[i][0])
        feature2.append(features[i][1])
        feature3.append(features[i][2])
        feature4.append(features[i][3])
        feature5.append(features[i][4])
        feature6.append(features[i][5])
        feature7.append(features[i][6] )
        label.append(labels[i])
    f1 = pd.cut(feature1,bins = 21,labels = list(range(1,22)))
    f2 = pd.cut(feature2, bins=21, labels=list(range(1, 22)))
    f3 = pd.cut(feature3,bins = 21,labels = list(range(1,22)))
    f4 = pd.cut(feature4,bins = 21,labels = list(range(1,22)))
    f5 = pd.cut(feature5, bins=21, labels=list(range(1, 22)))
    f6 = pd.cut(feature6, bins=21, labels=list(range(1, 22)))
    f7 = pd.cut(feature7, bins=21, labels=list(range(1, 22)))
    train_mask = np.load('dataset/epinion/train_mask-epinions.npy')
    print(len(train_mask))
    file = open('output/epinion/feature.txt',"w")
    for i in range(len(labels)):
        if train_mask[i] == True:
            line0 = '+'+str(label[i])
        if train_mask[i] == False:
            line0 = '?' + str(label[i])
        line1 = 'first_a_p_'+str(f1[i])+':1'
        line2 = 'second_a_p_'+str(f2[i])+':1'
        line3 = 'p_ratio_' + str(f3[i])+':1'
        line4 = 'first_ratio_' + str(f4[i])+':1'
        line5 = 'second_ratio_' + str(f5[i])+':1'
        line6 = 'first_year_span_' + str(f6[i])+':1'
        line7 = 'pr_ratio_' + str(f7[i])+':1'

        lines = line0+' '+line1+' '+line2+' '+line3+' '+line4+' '+line5+' '+line6+' '+line7+'\n'
        file.writelines(lines)
    file.close()
    i = 0
    lines = []
    file = open('output/epinion/edge.txt', "w")
    d_label1 = dict()
    positive = []
    for i in range(len(labels)):
        if labels[i] == 1:
            positive.append(i)
    G= nx.Graph()
    G.add_edges_from(edges)
    for i in range(len(positive)):
        for j in range(i + 1, len(positive)):


            if train_mask[positive[i]] == True and train_mask[positive[j]] == True:

                if G.has_edge(positive[i],positive[j]):
                    line = '#edge' + ' ' + str(positive[i]) + ' ' + str(positive[j]) + ' ' + 'same-advisor' + '\n'
                    lines.append(line)


    for line in lines:
        file.writelines(line)
    file.close()



edges, labels, features, net = preprocess(filename='dataset/epinion/soc-sign-epinions.txt', pagerank_alpha=0.85, max_depth=10000, start_vertices=[1, 2, 4, 5, 6])
#xiaoqi()
crf_data()
#getshifufeature(edges, labels, features, net)
G1 = nx.Graph()
G1.add_edges_from(edges)

# In[8]:



