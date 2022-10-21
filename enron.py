import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
file = open('dataset/enron/weighted_links.csv','r')

file1 = open('dataset/enron/enron_manager_subordinate_relationships.txt','r')



def readgraph(file, file1):
    egoe = []
    edges = []
    G = nx.Graph()
    edge1 = []
    edge2 = []
    colors = []
    num1 = 0
    num2 = 0
    email = []
    d_w = {}
    d_w_a = {}
    d_sen = {}
    for lines in file1.readlines():
        lines = lines.strip('\n')
        line = lines.split(',')
        if line[4] not in egoe:
            egoe.append(line[4])
        if line[5] not in egoe:
            egoe.append(line[5])
        tuple1 = line[4],line[5]
        tuple2 = line[5],line[4]
       # print(tuple1)
        if tuple1 not in edge2:
            edge2.append(tuple1)
        if tuple1 not in edge1:
            edge1.append(tuple1)
        if tuple2 not in edge1:
            edge1.append(tuple2)
    #print(len(egoe))
    num = 0
    for lines in file.readlines():
        if lines.__contains__('@') == 1:
            lines = lines.strip('\n')
            line = lines.split(',')
            if line[2].isdigit():
                d_w_a[line[0], line[1]] = int(line[2])
            num = num + 1
   # print (d_w_a)
            #print(num)
    file.close()
    file = open('dataset/enron/weighted_links.csv', 'r')
    for lines in file.readlines():
        if lines.__contains__('@') == 1:
            lines = lines.strip('\n')
            line = lines.split(',')
            if line[0] not in email:
                email.append(line[0])
            if line[1] not in email:
                email.append(line[1])
            tuple = line[0], line[1]

            if line[0] in egoe or line[1] in egoe:
                tuple1 = line[0], line[1]
                tuple2 = line[1], line[0]
                i = 0
                j = 0
                if tuple1 in d_w_a:
                    i = d_w_a[tuple1]
                if tuple2 in d_w_a:
                    j = d_w_a[tuple2]
                if i+j > 100 or (tuple1 in edge1 and i + j > 0):
                    if line[0] not in d_sen:
                        d_sen[line[0]] = []
                    d_sen[line[0]].append(line[1])
                    #print(tuple1)
                    if tuple1 == ('daren.farmer@enron.com', 'yvette.connevey@enron.com'):
                        print('TT')
                    d_w[tuple1] = int(line[2])
                    if ('lauragammell@hotmail.com', 'Gerald.Nemec@enron.com') in d_w:
                        print('T')
                    if tuple1 not in edges and tuple2 not in edges:
                        if tuple1 in edge1:
                            colors.append('r')
                            num1 = num1 + 1
                            if tuple1 not in edge2:
                                tuple1 = tuple2
                        else:
                            colors.append('b')
                            num2 = num2 + 1
                        edges.append(tuple1)




                '''
                if int(line[2]) > 15 or ((line[0],line[1]) in edge1 and int(line[2]) > 5):
                    tuple = line[0], line[1]
                    if tuple in edge1:
                        num = num + 1
                        if tuple not in edge2:    #将上下级关系顺序一致
                            tuple = line[1], line[0]

                            if tuple not in edge2:
                                print('no')

                          #  print('1')
                        else:
                            tuple = line[0], line[1]
                            if tuple not in edge2:
                                print('no')
                           # print('2')
                    else:
                        tuple = line[0], line[1]
                    if tuple not in edges:
                        if tuple in edge2:
                            colors.append('r')
                            
                        else:
                            colors.append('b')
                            
                        edges.append(tuple)
                        '''
    G.add_edges_from(edges)
    d_n_adj = dict()
    A = np.array(nx.adjacency_matrix(G).todense())
    i = 0
    for node in G.nodes:
        d_n_adj[node] = A[i]
        i = i+1
    print(num1, num2, num)
    '''
    num = 0
    for e in egoe:
        if e in email:
            num = num + 1
    print('总数：',len(egoe),'邮件包含数：',num)
    '''
    num = 0
    for edge in edge2:
        if edge in edges:
            num = num + 1
    print(num)

    num = 0
    for edge in edge1:
        if edge not in edge2:
            if edge in edges:
                num = num + 1
    print(num)
    return G, edges, colors, d_w, d_sen, d_n_adj



def getfeature_CRF(edges, d_w, d_sen, colors):
    feature = []
    feature1 =[]
    feature2 = []
    feature3 = []
    feature4 = []
    feature5 = []
    feature6 = []
    feature7 = []
    feature8 = []
    feature9 = []
    feature10 = []

    features = []
    sen1 = 0
    sen2 = 0
    sen3 = 0
    sen4 = 0
    rec1 = 0
    rec2 = 0
    for edge in edges:
        sen1 = 0
        sen2 = 0
        sen3 = 0
        sen4 = 0
        rec1 = 0
        rec2 = 0
        for tuple in d_w:
            if edge[0] == tuple[0]:   #1发的所有邮件
                if edge[1] == tuple[1]:
                    sen1 = d_w[tuple]
                else:
                    sen2 = sen2 + d_w[tuple]
            if edge[1] == tuple[0]:
                if edge[0] == tuple[1]:
                    sen3 = d_w[tuple]
                else:
                    sen4 = sen4 + d_w[tuple]
            if edge[0] == tuple[1]:
                if edge[1] != tuple[0]:
                    rec1 = rec1 + d_w[tuple]
            if edge[1] == tuple[1]:
                if edge[0] != tuple[0]:
                    rec2 = rec2 + d_w[tuple]
        num = 0
        for e in d_sen:
            if edge[0] in d_sen[e]:
                if edge[1] in d_sen[e]:
                    tuple1 = e, edge[0]
                    tuple2 = e, edge[1]
                    num = num + d_w[tuple1]+ d_w[tuple2]
        feature1.append(sen1 + sen2)
        feature2.append(sen3 + sen4)
        feature3.append(n_pr[edge[0]])
        feature4.append(n_pr[edge[1]])
        feature5.append(sen1)  #都是1？
        feature6.append(sen3)  #doushi 0 ???
        #feature.append(sen2)  #???
        #feature.append(sen4)
        feature7.append(rec1)
        feature8.append(rec2)
        feature9.append(num)
        feature10.append(n_pr[edge[0]]-n_pr[edge[1]])
  #  print(feature5)
  #  print(feature6)
  #  print(feature7)

    f1 = pd.qcut(feature1, 4,labels = list(range(0,4)))
    f2 = pd.qcut(feature2, 4,labels = list(range(0,4)))
    f3 = pd.qcut(feature3, 4,labels = list(range(0,4)))
    f4 = pd.qcut(feature4, 4,labels = list(range(0,4)))
    f5 = pd.qcut(feature5, 4,labels = list(range(0,4)))
    #f5 = pd.qcut(feature5, bins=10 )
    f6 = pd.qcut(feature6, 4,labels = list(range(0,4)))
    f7 = pd.qcut(feature7, [0,0.4,0.6,0.8,1],labels = list(range(0,4)))
    f8 = pd.qcut(feature8, [0,0.3,0.5,0.7,0.9,1],labels = list(range(0,5)))
    f9 = pd.qcut(feature9, [0,0.6,0.8,1],labels = list(range(0,3)))
    f10 = pd.qcut(feature10, 4, labels = list(range(0,4)))
    print(f5)
    print(f6)
    print(f7)
    train_mask = np.load('dataset/enron/train_mask-enron.npy') ##define the trainset
    file = open('output/enron/enron-feature.txt', "w")
    for i in range(len(edges)):
        if train_mask[i] == True:
            if colors[i] == 'r':
                line0 = '+' + '1'
            if colors[i] == 'b':
                line0 = '+' + '0'
        if train_mask[i] == False:
            if colors[i] == 'r':
                line0 = '?' + '1'
            if colors[i] == 'b':
                line0 = '?' + '0'
        line1 = 'att1_'+str(f1[i])+':1'
        line2 = 'att2_'+str(f2[i])+':1'
        line3 = 'att3_' + str(f3[i])+':1'
        line4 = 'att4_' + str(f4[i])+':1'
        line5 = 'att5_' + str(f5[i])+':1'
        line6 = 'att6_' + str(f6[i])+':1'
        line7 = 'att7_' + str(f7[i])+':1'
        line8 = 'att8_' + str(f8[i]) + ':1'
        line9 = 'att9_' + str(f9[i]) + ':1'
        line10 = 'att10_' + str(f10[i]) + ':1'

        lines = line0+' '+line1+' '+line2+' '+line5+' '+line6+' '+line7+' '+line8+' '+line9+'\n'
        file.writelines(lines)

def getCRFedge(edges, d_sen, colors):
    i = 0
    lines = []
    ll = []
    file = open('output/enron/enron-edge.txt',"w")
    d_label1 = dict()
    train_mask = np.load('dataset/enron/train_mask-enron.npy')
    d = {}
    num = 0
    for edge in edges:
        d[edge] = num
        num = num + 1
    num = 0
    for edge in edges:
        for e in d_sen:
            sen = 0
            for tuple in d_w:
                if e == tuple[0]:  # 1发的所有邮件
                    sen = sen + d_w[tuple]
            if edge[0] in d_sen[e]:
                if edge[1] in d_sen[e]:
                    if sen >= 10:
                        tuple1 = e, edge[0]
                        tuple2 = e, edge[1]
                        if tuple1 in d and tuple2 in d:
                            tt = d[tuple1],d[tuple2]

                            if tt not in ll and tuple1 != tuple2:
                                ll.append(tt)
                                line = '#edge' + ' ' + str(d[tuple1]) + ' ' + str(d[tuple2]) + ' ' + '111' + '\n'
                                lines.append(line)
    for i in range(len(edges)):
        if colors[i] == 'r':
            d_label1[edges[i]] = 1
    print(d_label1)
    for i in range(len(d_label1)):
        for j in range(i+1, len(d_label1)):

            tuple1 = list(d_label1.keys())[i]
            tuple2 = list(d_label1.keys())[j]
            if train_mask[d[tuple1]]==True and train_mask[d[tuple2]]==True:

                if tuple1[1] == tuple2[1]:
                    tt = d[tuple1], d[tuple2]

                    if tt not in ll:
                        ll.append(tt)
                        line = '#edge'+' '+str(d[tuple1]) + ' ' + str(d[tuple2]) +' '+'222'+'\n'
                        lines.append(line)

                if tuple1[0] == tuple2[0]:
                    tt = d[tuple1], d[tuple2]

                    if tt not in ll:
                        ll.append(tt)
                        line = '#edge' + ' ' + str(d[tuple1]) + ' ' + str(d[tuple2])+' '+ '333'+'\n'
                        lines.append(line)
    for line in lines:
        file.writelines(line)


G, edges, colors, d_w, d_sen, d_n_adj= readgraph(file, file1)

n_pr = nx.pagerank(G)

getfeature_CRF(edges, d_w, d_sen, colors)
getCRFedge(edges, d_sen, colors)

def paint(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, with_labels = False, edge_color=colors, node_size = 50, width = 2 )  # pos=nx.spring_layout(G)
    plt.axis('off')
    plt.show()

#paint(G)