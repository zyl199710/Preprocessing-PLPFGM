import itertools
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
def author_paper(file):
    tuple = []
    G = nx.Graph()
    d_a_a_p = {}
    edges = []
    d_a_p = dict()
    d_p_a = dict()
    for lines in file.readlines():
        line = lines.split('\t')
        if int(line[0]) not in d_p_a:
            d_p_a[int(line[0])] = []
        if int(line[1]) not in d_p_a[int(line[0])]:
            d_p_a[int(line[0])].append(int(line[1]))

        if int(line[1]) not in d_a_p:
            d_a_p[int(line[1])] = []
        if int(line[0]) not in d_a_p[int(line[1])]:
            d_a_p[int(line[1])].append(int(line[0]))
    #file3 = open('dataset\Coauthor\ author_paper_mini.txt',"w")

    for i in d_a_p:
        for j in d_a_p[i]:
            tuple = i,j
            edges.append(tuple)
            #file3.write(i+'\t'+j+'\n')

    file = open('dataset/coauthor/paper_year.txt')
    d_p_y = dict()
    d_a_y = dict()
    for lines in file.readlines():
        line1 = lines.split('\n')
        line = line1[0].split('\t')
        d_p_y[int(line[0])]= int(line[1])
    for i in d_a_p:
        l = []
        for j in d_a_p[i]:
            l.append(d_p_y[j])
        l.sort()
        d_a_y[i] = l[0]

    for i in d_p_a:
        for j1 in d_p_a[i]:
            for j2 in d_p_a[i]:
                i1, i2 = min(j1, j2), max(j1, j2)
                if d_a_y[i1] < d_a_y[i2]:
                    i1, i2 = i2, i1
                tuple = i1, i2
                if tuple not in d_a_a_p:
                    d_a_a_p[tuple] = []
                if i not in d_a_a_p[tuple]:
                    d_a_a_p[tuple].append(i)
    #print(d_a_a_p)
    return d_a_p,d_p_a,d_a_a_p,d_a_y

def number_node(file1,file2,file3,file4,file5,d_p_a,d_a_p):
    num = 1
    d = dict()
    d_label = dict()
    edges = []
    nodes = []
    G = nx.Graph()
    '''
    for lines in file1.readlines():
        lines = lines.strip('\n')
        line = lines.split('\t')
        tuple = int(line[0]), int(line[1])

        tuple = min(tuple[0], tuple[1]), max(tuple[0], tuple[1])
        d_label[tuple] = 0
        if tuple not in edges:
            d[tuple] = num
            edges.append(tuple)  # 作者-
            num = num + 1
'''
    for lines in file2.readlines():
        lines = lines.strip('\n')
        line = lines.split(' ')
        tuple = int(line[0]),int(line[1])

        d_label[tuple] = 1
        if tuple not in edges:
            d[tuple] = num
            i1, i2 = tuple
            tuple1 = i2, i1
            edges.append(tuple1)
            edges.append(tuple)  # 作者-
            num = num + 1

    for lines in file3.readlines():
        lines = lines.strip('\n')
        line = lines.split(' ')
        tuple = int(line[0]), int(line[1])

        d_label[tuple] = 1
        if tuple not in edges:
            d[tuple] = num
            i1, i2 = tuple
            tuple1 = i2, i1
            edges.append(tuple1)
            edges.append(tuple)  # 作者-
            num = num + 1

    for lines in file4.readlines():
        lines = lines.strip('\n')
        line = lines.split(' ')
        tuple = int(line[0]), int(line[1])


        d_label[tuple] = 1
        if tuple not in edges:
            d[tuple] = num
            i1, i2 = tuple
            tuple1 = i2, i1
            edges.append(tuple1)
            edges.append(tuple)  # 作者-
            num = num + 1

    for lines in file5.readlines():
        lines = lines.strip('\n')
        line = lines.split(' ')
        tuple = int(line[0]), int(line[1])

        d_label[tuple] = 1
        if tuple not in edges:
            d[tuple] = num
            i1, i2 =tuple
            tuple1 = i2, i1
            edges.append(tuple1)
            edges.append(tuple)  # 作者-
            num = num + 1

    G.add_edges_from(edges)
    sub_G = max(nx.connected_components(G), key=len)
    small_components = sorted(nx.connected_components(G), key=len)[:-1]  #前三极大图
    G.remove_nodes_from(itertools.chain.from_iterable(small_components))
    print(len(G.edges))
    for p in d_p_a:
        l = set(d_p_a[p]).intersection(G.nodes)
        #print(l)
        if len(l) >= 1:
            for i in range(len(d_p_a[p])):
                for j in range(i+1,len(d_p_a[p])):
                    i1,i2 = min(int(d_p_a[p][i]),int(d_p_a[p][j])), max(int(d_p_a[p][i]),int(d_p_a[p][j]))
                    tuple = i1 , i2
                    #print(tuple)
                    if tuple not in edges:
                        edges.append(tuple)
                        d[tuple] = num
                        i1, i2 = tuple
                        tuple1 = i2, i1
                        edges.append(tuple1)
                        num = num+1
                        d_label[tuple] = 0

    #print(d,len(d))
    G.add_edges_from(edges)


    '''
    nx.draw_networkx(G, node_size=50, with_labels = False)  # pos=nx.spring_layout(G)
    plt.axis('off')
    plt.show()
'''
#   print(d)
#   print(d_label)
    return G,d_label,d

def getfeature(d_a_p,d,d_a_y,n_pr,d_label):
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
    for tuple in d:
        i = tuple[0]
        j = tuple[1]
        feature1.append(len(d_a_p[i]))
        feature2.append(len(d_a_p[j]))
        feature3.append(len(d_a_p[i]) / len(d_a_p[j]))
        l1 = d_a_p[tuple[0]]
        l2 = d_a_p[tuple[1]]
        p_cp = len(set(l1).intersection(l2))
        feature4.append(p_cp / len(d_a_p[i]))
        feature5.append(p_cp / len(d_a_p[j]))
        feature6.append(d_a_y[i] - d_a_y[j])
        feature7.append(n_pr[i] / n_pr[j] )
        label.append(d_label[tuple])
    f1 = pd.cut(feature1,bins = 21,labels = list(range(1,22)))
    f2 = pd.cut(feature2, bins=21, labels=list(range(1, 22)))
    f3 = pd.cut(feature3,bins = 21,labels = list(range(1,22)))
    f4 = pd.cut(feature4,bins = 21,labels = list(range(1,22)))
    f5 = pd.cut(feature5, bins=21, labels=list(range(1, 22)))
    f6 = pd.cut(feature6, bins=21, labels=list(range(1, 22)))
    f7 = pd.cut(feature7, bins=21, labels=list(range(1, 22)))
    train_mask = np.load('dataset/coauthor/train_mask.npy')
    file = open('output/coauthor/feature.txt',"w")
    for i in range(len(d)):
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

def getedge(d,d_label):
    i = 0
    lines = []
    file = open('output/coauthor/edge.txt',"w")
    d_label1 = dict()
    train_mask = np.load('dataset/coauthor/train_mask.npy')
    for tuple in d_label:
        if d_label[tuple] == 1:
            d_label1[tuple] = 1
    for i in range(len(d_label1)):
        for j in range(i+1, len(d_label1)):

            tuple1 = list(d_label1.keys())[i]
            tuple2 = list(d_label1.keys())[j]
            if train_mask[d[tuple1]-1]==True and train_mask[d[tuple2]-1]==True:

                if tuple1[1] == tuple2[1]:
                    line = '#edge'+' '+str(d[tuple1]-1) + ' ' + str(d[tuple2]-1) +' '+'same-advisor'+'\n'
                    lines.append(line)

                if tuple1[0] == tuple2[0]:
                    line = '#edge' + ' ' + str(d[tuple1] - 1) + ' ' + str(d[tuple2] - 1)+' '+ 'same-student'+'\n'
                    lines.append(line)
    for line in lines:
        file.writelines(line)



file1 = open('dataset/coauthor/colleague.txt')
file2 = open('dataset/coauthor/phd.ans')
file3 = open('dataset/coauthor/MathGenealogy_50896.ans')
file4 = open('dataset/coauthor/teacher.ans')
file5 = open('dataset/coauthor/ai.ans')
file6 = open('dataset/coauthor/paper_author.txt')

d_a_p, d_p_a, d_a_a_p, d_a_y = author_paper(file6)

G, d_label, d = number_node(file1, file2, file3, file4, file5, d_p_a, d_a_p)
print('dic_finished')
n_pr = nx.pagerank(G)
getfeature(d_a_p,d,d_a_y,n_pr,d_label)
getedge(d,d_label)


