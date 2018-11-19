import os
# import sys
import time

# import matplotlib.pyplot as plt
# from matplotlib import collections as mc
import networkx as nx
from copy import copy, deepcopy
# import pydot
# from networkx.drawing.nx_pydot import graphviz_layout
from forceatlas2 import ForceAtlas2
import numpy as np
from node2vec import Node2Vec
from attri2vec import Attri2Vec
import graph
import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# import node2vec
import nltk
from nltk.cluster import KMeansClusterer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
import logging
from color import color, shape
# from zoompan import ZoomPan
import matplotlib.patches as patches
# from draggablerectangle import DraggableRectangle
# import itertools

NODE_COLOR = 'lightseagreen'
NODE_SIZE = 150
NODE_SHAPE = 'o'

logging.basicConfig(
    filename="system.log",
    level=logging.WARNING,
    format="%(asctime)s:%(levelname)s:%(message)s"
)


class Attri2vecTest:
    dataFile = ''
    attriFile = ''
    classFile = ''
    graph = ''
    attriGraph = ''
    nodeGraph = ''
    attriNode = []
    attriEdges = []
    attriLines = []
    nodeEdges = []
    nodeLines = []
    nodeLabels = []
    model = ''
    cluster = ''
    nodePosition = ''
    nodeXs = []
    nodeYs = []
    clusterColors = color
    saNodeColors = NODE_COLOR
    nodeColors = []
    saNodeSizes = NODE_SIZE
    nodeSizes = []
    saNodeShapes = NODE_SHAPE
    nodeShapes = []
    nodeFactor = 1.0
    # edgeColor = ''
    selectedNode = []
    isShowLabel = False
    isShowAttriNode = False
    isShowEdge = False

    embeddingMethodList = ['node2vec', 'struc2vec', 'attri2vec', 'HSCA', 'TADW']
    embeddingMethod = 'node2vec'
    P = 1.0
    Q = 1.0
    R = 1.0
    D = 16
    clusteringMethodList = ['K-means', 'EM', 'GAA']
    clusteringMethod = 'K-means'
    clusteringNum = 6
    reductionMethodList = ['t-SNE', 'NONE', 'Isomap', 'LLE', 'MLLE', 'HLLE', 'Spectral', 'LTSA', 'MDS']
    reductionMethod = 't-SNE'
    colorMethodList = ['Unitary', 'Attribute', 'Cluster', 'Class']
    colorMethod = 'Unitary'
    colorCombox = ''
    sizeMethodList = ['Unitary', 'Degree', 'Class']
    sizeMethod = 'Unitary'
    shapeMethodList = ['Unitary', 'Value', 'Class']
    shapeMethod = 'Unitary'

    ax = ''
    cur_xlim = ''
    cur_ylim = ''
    oax = ''
    overviewRec = ''
    overViewWidget = ''

    def __init__(self, dir=[]):
        if dir == []:
            self.scalabilityTest()
        else:
            self.openfolder(dir)
            self.batch()

    def getHeaderAndSeperator(self, file):
        hs = [False, ' ']
        with open(file, 'rb') as inf:
            header = inf.readline().decode()
            # print('header', header)
            if len(header.split(',')) > 1:
                hs[1] = ','
            elif len(header.split('\t')) > 1:
                hs[1] = '\t'
            elif len(header.split(' ')) > 1:
                hs[1] = ' '
            else:
                return []
            # print('hs', hs)
            header = header.split(hs[1])
            # print('header', header)
            if header[0] == 'Source':
                hs[0] = True
            inf.close()
            return hs

    def readEdgeFile(self, edgeFile):
        self.dataFile = edgeFile
        with open(self.dataFile, 'rb') as inf:
            hinf = self.getHeaderAndSeperator(self.dataFile)
            if len(hinf) == 0:
                return
            if hinf[0]:
                next(inf, '')  # skip a line
            e = nx.read_edgelist(inf, delimiter=hinf[1], nodetype=str, encoding="utf-8")
            self.nodeGraph = nx.Graph()
            self.nodeGraph.add_edges_from(e.edges())
        # print('Nodes', len(self.nodeGraph.nodes()), 'Edges', len(self.nodeGraph.edges()))
        self.graph = self.nodeGraph

    def snapReader(self, dir):
        edgeFile = dir + '/edgelist'
        vertex2attriFile = dir + '/vertex2aid'
        classFile = dir + '/Class_info'
        if not os.path.isfile(edgeFile):
            return False
        print(time.ctime())
        beginTime = time.time()
        print('0: Open SNAP Begin')
        self.readEdgeFile(edgeFile)

        # todo ground true class file
        self.classFile = classFile
        with open(self.classFile, 'rb') as inf:
            next(inf, '')  # skip a line
            na = [[] for x in self.nodeGraph.nodes()]
            na = dict(zip(self.nodeGraph.nodes(), na))
            # print('na', na)
            cls = 0
            for line in inf:
                line = line.decode()
                # print('line', line)
                if line.startswith('Circle'):
                    cls = int(line[line.index('#') + 1:])
                else:
                    nc = line.split('	')
                    # print('nc', nc)
                    for x in nc:
                        if x == '\n':
                            continue
                        if x in list(na.keys()):
                            na[x].append(cls)
                        else:
                            print('class file node not in edge file', x)
                            self.nodeGraph.add_node(x)
                            na[x] = [cls]
                    # [na[x].append(cls) for x in nc if x != '\n']
            # print('na', na)
            nx.set_node_attributes(self.nodeGraph, 'class', na)
            # get_attris = nx.get_node_attributes(self.nodeGraph, 'class')
            # print('get_attris', get_attris)

        self.attriFile = vertex2attriFile
        with open(self.attriFile, 'rb') as inf:
            hinf = self.getHeaderAndSeperator(self.attriFile)
            if len(hinf) == 0:
                return
            if hinf[0]:
                next(inf, '')  # skip a line
            e = nx.read_edgelist(inf, delimiter=hinf[1], create_using=nx.DiGraph(), nodetype=str,
                                 encoding="utf-8")
            na = [[] for x in self.nodeGraph.nodes()]
            na = dict(zip(self.nodeGraph.nodes(), na))
            # print('na', list(na.keys()))
            for x in e.edges():
                if x[0] in list(na.keys()):
                    na[x[0]].append(x[1])
                else:
                    print('attribute file node not in edge file', x)
                    self.nodeGraph.add_node(x[0])
                    na[x[0]] = [x[1]]

            # [na[x[0]].append(x[1]) for x in e.edges()]
            # print('na', na)
            nx.set_node_attributes(self.nodeGraph, 'value', na)
            # get_attris = nx.get_node_attributes(self.nodeGraph, 'value')
            # print('get_attris', get_attris)
            ea = [(x[0], 'attri-' + x[1]) for x in e.edges()]
            [self.attriNode.append(x[1]) for x in ea if x[1] not in self.attriNode]
            self.attriGraph = deepcopy(self.nodeGraph)
            self.attriGraph.add_edges_from(ea)
            self.graph = self.attriGraph

        print('Nodes', len(self.nodeGraph.nodes()), 'Edges', len(self.nodeGraph.edges()))
        print('Nodes + AttriNode', len(self.attriGraph.nodes()), 'Edges + AttriEdge',
              len(self.attriGraph.edges()))
        print('Time of Open', time.time() - beginTime)
        return True

    def linqsReader(self, dir):
        edgeFile = ''
        vertex2attriFile = ''
        for filename in os.listdir(dir):
            if filename.endswith(".cites"):
                edgeFile = filename
            elif filename.endswith(".content"):
                vertex2attriFile = filename

        if edgeFile == '':
            return False

        edgeFile = dir + '/' + edgeFile
        vertex2attriFile = dir + '/' + vertex2attriFile

        # print(edgeFile, vertex2attriFile)
        print(time.ctime())
        beginTime = time.time()
        print('0: Open LINQS Begin')
        self.readEdgeFile(edgeFile)
        self.attriFile = vertex2attriFile
        with open(self.attriFile, 'rb') as inf:
            hinf = self.getHeaderAndSeperator(self.attriFile)
            # print('hinf', hinf)
            if len(hinf) == 0:
                return
            if hinf[0]:
                next(inf, '')  # skip a line
            ea = []
            nal = []
            na = [[] for x in self.nodeGraph.nodes()]
            na = dict(zip(self.nodeGraph.nodes(), na))
            nc = dict.fromkeys(self.nodeGraph.nodes())
            for line in inf:
                # print('line', line)
                el = line.decode().rstrip('\n').split(hinf[1])
                # el = line.split(hinf[1])
                eleSize = len(el)
                # print('el', el[eleSize-1])
                if nal == []:
                    [nal.append('attri-' + str(i)) for i in range(1, eleSize - 1)]
                nc[el[0]] = el[eleSize - 1]
                for i in range(1, eleSize - 1):
                    if el[i] != '0':
                        # print('el[0]', na[el[0]])
                        ea.append((el[0], 'attri-' + str(i)))
                        if el[0] in list(na.keys()):
                            na[el[0]].append(i)
                        else:
                            print('class file node not in edge file', el[0])
                            self.nodeGraph.add_node(el[0])
                            na[el[0]] = [i]
                        # na[el[0]].append(i)
                # [(ea.append((el[0], 'attri-' + str(i))), na[el[0]].append('attri-' + str(i)))
                # for i in range(1, eleSize-2) if el[i] != '0']
            # print('nal', nal)
            # print('nc', nc)
            nx.set_node_attributes(self.nodeGraph, 'value', na)
            nx.set_node_attributes(self.nodeGraph, 'class', nc)
            self.attriGraph = deepcopy(self.nodeGraph)
            self.attriNode = nal
            # get_attris = nx.get_node_attributes(self.nodeGraph, 'class')
            # print('get_attris', get_attris)
            self.attriGraph.add_edges_from(ea)
            # self.attriGraph.remove_nodes_from(isolatedNodes)
            self.graph = self.attriGraph

        print('Nodes', len(self.nodeGraph.nodes()), 'Edges', len(self.nodeGraph.edges()))
        print('Nodes + AttriNode', len(self.attriGraph.nodes()), 'Edges + AttriEdge',
              len(self.attriGraph.edges()))
        print('Time of Open', time.time() - beginTime)
        return True

    def tadwReader(self, dir):
        edgeFile = dir + '/graph.txt'
        vertex2attriFile = dir + '/feature.txt'
        classFile = dir + '/group.txt'
        print(time.ctime())
        beginTime = time.time()
        print('0: Open TADW Begin')
        self.readEdgeFile(edgeFile)

        # todo ground true class file
        self.classFile = classFile
        with open(self.classFile, 'rb') as inf:
            hinf = self.getHeaderAndSeperator(self.classFile)
            # print('hinf', hinf)
            if len(hinf) == 0:
                return
            if hinf[0]:
                next(inf, '')  # skip a line
            na = [[] for x in self.nodeGraph.nodes()]
            na = dict(zip(self.nodeGraph.nodes(), na))
            # print('na', na)
            for line in inf:
                nc = line.decode().rstrip('\r').split(hinf[1])
                if nc[0] in list(na.keys()):
                    na[nc[0]].append(int(nc[1]))
                else:
                    print('class file node not in edge file', nc[0])
                    self.nodeGraph.add_node(nc[0])
                    na[nc[0]] = [int(nc[1])]
            # print('na-class', na)
            nx.set_node_attributes(self.nodeGraph, 'class', na)
            # get_attris = nx.get_node_attributes(self.nodeGraph, 'class')
            # print('get_attris', get_attris)

        self.attriFile = vertex2attriFile
        with open(self.attriFile, 'rb') as inf:
            hinf = self.getHeaderAndSeperator(self.attriFile)
            # print('hinf', hinf)
            if len(hinf) == 0:
                return
            if hinf[0]:
                next(inf, '')  # skip a line
            ea = []
            nal = []
            na = [[] for x in self.nodeGraph.nodes()]
            na = dict(zip(self.nodeGraph.nodes(), na))
            for num, line in enumerate(inf, 0):
                nc = line.decode().rstrip('\r').split(hinf[1])
                # print('nc', nc)
                eleSize = len(nc)
                # print('eleSize', eleSize)
                if nal == []:
                    [nal.append('attri-' + str(i)) for i in range(0, eleSize)]
                for i in range(0, eleSize):
                    if nc[i] != '0':
                        ea.append((str(num), 'attri-' + str(i)))
                        na[str(num)].append(str(i))
            # print('na-attri', na)
            # print('ea', ea)
            nx.set_node_attributes(self.nodeGraph, 'value', na)
            # get_attris = nx.get_node_attributes(self.nodeGraph, 'value')
            # print('get_attris', get_attris)

            self.attriGraph = deepcopy(self.nodeGraph)
            self.attriNode = nal
            self.attriGraph.add_edges_from(ea)
            self.graph = self.attriGraph

        print('Nodes', len(self.nodeGraph.nodes()), 'Edges', len(self.nodeGraph.edges()))
        print('Nodes + AttriNode', len(self.attriGraph.nodes()), 'Edges + AttriEdge',
              len(self.attriGraph.edges()))
        print('Time of Open', time.time() - beginTime)
        return True

    def openfolder(self, dir):
        logging.warning('Dataset {}'.format(dir))
        if dir == '':
            return

        if not self.snapReader(dir):
            if not self.linqsReader(dir):
                if not self.tadwReader(dir):
                    return

        self.plot()

    def embedding(self):
        if self.embeddingMethod == 'node2vec':
            self.node2vec()
        elif self.embeddingMethod == 'struc2vec':
            self.struc2vec()
        elif self.embeddingMethod == 'attri2vec':
            self.attri2vec()
        elif self.embeddingMethod == 'HSCA':
            self.hsca()
            self.plot()
            return

        self.reducting()
        self.clustering()  # includes plot
        # self.plot_subplot()

    def batch(self):
        # [2.0, 3.0, 4.0, 5.0, 6.0]
        # [0.05, 0.1, 0.15, 0.2, 0.3]
        # [2.0, 3.0, 4.0, 5.0, 6.0]
        for i in [1.0]:  # range(10, 0, -1) *.1  0.25, 0.5, 1.0,
            for j in [1.0]:  # , 0.5, 1.0, 2.0, 4.0
                for k in [0.01]:  # 0.25, 0.5,
                    for d in [8]:  # 16, 32, 128
                        self.P = i
                        self.Q = j
                        self.R = k
                        self.D = d
                        self.struc2vec()
                        self.svm()
                        # self.reducting()
                        # self.clustering()
                        # self.figure.savefig('./results/p'+str(i*.1)+'-q'+str(j*.1)+'-r'+str(k*.1)+'.png')

    def scalabilityTest(self):
        ap = 3  # attribute num per node
        for a_num in range(1, 2):
            # ar = 100*a_num  # total attribute number
            ar = 10*pow(10, a_num)
            for n_power in range(1, 7):
                self.nodeGraph = nx.erdos_renyi_graph(10*pow(10, n_power), 0.05)
                na = [np.random.randint(ar, size=(1, ap)) for x in self.nodeGraph.nodes()]
                print('na', na)
                na = dict(zip(self.nodeGraph.nodes(), na))
                print('na', na)
                nx.set_node_attributes(self.nodeGraph, 'value', na)
                ea = []
                for key, value in na.items():
                    for x in value.ravel():
                        ea.append((key, 'attri-' + str(x)))
                # print('ea', ea)
                self.attriNode.clear()
                [self.attriNode.append(x[1]) for x in ea if x[1] not in self.attriNode]
                self.attriGraph = deepcopy(self.nodeGraph)
                self.attriGraph.add_edges_from(ea)
                self.graph = self.attriGraph

                beginTime = time.time()
                self.attri2vec()
                logging.warning('- scalarTest ar={} ap={} n={} t={}'.format(ar, ap, 10*pow(10, n_power),
                                                                            time.time() - beginTime))

    def node2vec(self):
        if self.nodeGraph == '':
            return
        # self.graph = self.nodeGraph
        beginTime = time.time()
        print('1: node2vec Begin')
        # Precompute probabilities and generate walks
        # g = self.graph
        # g = deepcopy(self.graph)
        # if self.attriNode:
        #    g.remove_nodes_from(self.attriNode)
        node2vec = Node2Vec(self.nodeGraph, dimensions=16, walk_length=30, num_walks=200, p=self.P, q=self.Q, workers=4)
        print('Time of Node2Vec', time.time() - beginTime)
        beginTime = time.time()
        # Embed
        # Any keywords acceptable by gensim.Word2Vec can be passed,
        # `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)
        print('Time of FIT', time.time() - beginTime)

    def struc2vec(self):
        if self.nodeGraph == '':
            return
        beginTime = time.time()
        print('1: struc2vec Begin')
        strucGraph = graph.from_networkx(self.nodeGraph)
        strucGraph = struc2vec.Graph(strucGraph, 'undirected', 4, untilLayer=None)
        if True:
            strucGraph.preprocess_neighbors_with_bfs_compact()
        else:
            strucGraph.preprocess_neighbors_with_bfs()
        if True:
            strucGraph.create_vectors()
            strucGraph.calc_distances(compactDegree=True)
        else:
            strucGraph.calc_distances_all_vertices(compactDegree=True)
        strucGraph.create_distances_network()
        strucGraph.preprocess_parameters_random_walk()
        strucGraph.simulate_walks(10, 80)

        walks = LineSentence('random_walks.txt')
        self.model = Word2Vec(walks, size=self.D, window=10, min_count=0, hs=1, sg=1, workers=4, iter=5)
        print('Time of Struc2Vec', time.time() - beginTime)

    def attri2vec(self):
        if self.attriGraph == '':
            return
        self.graph = self.attriGraph
        beginTime = time.time()
        print('1: attri2vec Begin')
        # logging.warning('- attri2vec p={} q={} r={} d={}'.format(self.P, self.Q, self.R, self.D))
        # Precompute probabilities and generate walks
        attri2vec = Attri2Vec(self.graph, dimensions=self.D, walk_length=30, num_walks=200, p=self.P, q=self.Q,
                              r=self.R, workers=4)
        print('Time of Attri2Vec', time.time() - beginTime)
        beginTime = time.time()
        # Embed
        # Any keywords acceptable by gensim.Word2Vec can be passed,
        # `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        self.model = attri2vec.fit(window=10, min_count=1, batch_words=4)
        print('Time of FIT', time.time() - beginTime)

    def geri(self):
        if self.attriGraph == '':
            return
        self.graph = self.attriGraph
        beginTime = time.time()
        print('1: GERI Begin')
        logging.warning('- gemi p={} q={} r={} d={}'.format(self.P, self.Q, self.R, self.D))
        # Precompute probabilities and generate walks
        geri = GERI(self.graph, dimensions=self.D, walk_length=30, num_walks=200, p=self.P, q=self.Q,
                              r=self.R, workers=4)
        print('Time of GERI', time.time() - beginTime)
        beginTime = time.time()
        # Embed
        # Any keywords acceptable by gensim.Word2Vec can be passed,
        # `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        self.model = geri.fit(window=10, min_count=1, batch_words=4)
        print('Time of FIT', time.time() - beginTime)

    def clustering(self):
        if self.model == '':
            return
        # Clustering
        if self.clusteringMethod == 'K-means':
            self.kmeans()
        elif self.clusteringMethod == 'EM':
            self.em()
        elif self.clusteringMethod == 'GAA':
            self.gaa()
        index = self.colorCombox.findText('Cluster', QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.colorCombox.setCurrentIndex(index)
        self.colorMapping()

    def svm(self):
        beginTime = time.time()
        print('3: svm Begin')
        mv = self.model.vocab
        # print('mv', mv)
        for x in self.attriNode:
            if x in mv:
                del mv[x]
            else:
                print('x', x)  # why we lost attri-445 in cora data
        # print('mv', mv)
        model = self.model[mv]
        # print('model', model)
        realClass = nx.get_node_attributes(self.nodeGraph, 'class')
        y = []
        [y.append(realClass[v]) for v in self.model.vocab if v not in self.attriNode]
        # print('y', y)
        for x in range(1, 10):
            # print('train_size', x*.1)
            # logging.warning('- attri2vec p={} q={} r={} d={} t={}'.format(self.P, self.Q, self.R, self.D, x*.1))
            # logging.warning('-- train_size: {}'.format(x*.1))
            x_train, x_test, y_train, y_test = train_test_split(model, y, random_state=1, train_size=x*.1)
            # print('train', x_train, y_train)
            # C: Penalty parameter C of the error term. (default=1.0)
            # decision_function_shape: 'ovo', 'ovr'
            # random_state: The seed of the pseudo random number generator to use when shuffling the data
            for c in range(1, 11):
                logging.warning('- p={} q={} r={} d={} t={} c={}'.format(
                    self.P, self.Q, self.R, self.D, x * .1, c * .1))
                # logging.warning('--- svm-c: {}'.format(c * .1))
                clf = svm.SVC(C=c*.1, kernel='linear', decision_function_shape='ovr', random_state=0)
                # clf = svm.LinearSVC(random_state=0)
                clf.fit(x_train, y_train)  # .ravel()
                # clf.fit(model, y)
                # logging.warning('Class: {}'.format(clusters))
                print('accuracy train', clf.score(x_train, y_train))
                y_preds = clf.predict(x_train)
                print('y_hat_train', y_preds)
                # calculate f1
                mean_f1 = f1_score(y_train, y_preds, average='micro')
                print('mean_f1 train micro', mean_f1)
                logging.warning('--- mean_f1 train micro: {}'.format(mean_f1))
                mean_f1 = f1_score(y_train, y_preds, average='macro')
                print('mean_f1 train macro', mean_f1)
                logging.warning('--- mean_f1 train macro: {}'.format(mean_f1))
                print('accuracy test', clf.score(x_test, y_test))
                y_preds = clf.predict(x_test)
                # print('y_hat_test', y_preds)
                mean_f1 = f1_score(y_test, y_preds, average='micro')
                print('mean_f1 test micro', mean_f1)
                logging.warning('--- mean_f1 test micro: {}'.format(mean_f1))
                mean_f1 = f1_score(y_test, y_preds, average='macro')
                print('mean_f1 test macro', mean_f1)
                logging.warning('--- mean_f1 test macro: {}'.format(mean_f1))

        print('Time of svm', time.time() - beginTime)

    def kmeans(self):
        beginTime = time.time()
        print('3: kmeans Begin')
        model = self.model[self.model.vocab]
        print('model', model)
        kclusterer = KMeansClusterer(self.clusteringNum, distance=nltk.cluster.util.cosine_distance,
                                     repeats=25, avoid_empty_clusters=True)
        clusters = kclusterer.cluster(model, assign_clusters=True)
        print('clusters', clusters)
        self.cluster = dict(zip(list(self.model.vocab), clusters))
        print('self.clusters', self.cluster)

        # realClass = nx.get_node_attributes(self.nodeGraph, 'class')
        # print('realClass', realClass)
        # realClassSet = set(realClass.values())
        # print('realClassSet', realClassSet)
        # clustersSet = set(clusters)
        # print('clustersSet', clustersSet)
        # classMapping = np.zeros((len(clustersSet), len(realClassSet)))
        # for x in self.nodeGraph.nodes():
        #     classMapping[int(self.cluster[x])][int(realClass[x])] += 1
        # print('classMapping', classMapping)
        # orderArray = [x for x in range(len(clustersSet))]
        # allOrder = list(itertools.permutations(orderArray))
        # maxSum = 0
        # maxIdx = []
        # minD = min(len(classMapping), len(classMapping[0]))
        # for x in allOrder:
        #     subMaxSum = 0
        #     for idx, val in enumerate(x):
        #         if idx < minD:
        #             subMaxSum += classMapping[val][idx]
        #     if subMaxSum > maxSum:
        #         maxSum = subMaxSum
        #         maxIdx = x
        # print('maxSum: ', maxSum)
        # logging.warning('maxSum: {}/{}'.format(maxSum, len(self.nodeGraph.nodes())))
        # clusters = [maxIdx[x] for x in clusters]
        # print('clusters', clusters)
        # logging.warning('Class: {}'.format(clusters))
        # logging.warning('Truth: {}'.format(realClass.values()))
        # self.cluster = dict(zip(list(self.model.vocab), clusters))
        # tot = 0
        # for x in self.nodeGraph.nodes():
        #     if self.cluster[x] == int(realClass[x]):
        #         tot += 1
        # print('tot', tot)

        # print('classMapping', classMapping)
        # staticMap = np.zeros((len(classMapping[0]), 3))
        # print('staticMap', staticMap)
        # for i in range(0, len(classMapping[0])):
        #     staticMap[i][0] = classMapping[:, i].argmax()
        #     staticMap[i][1] = i
        #     staticMap[i][2] = np.max(classMapping[:, i])
        # print('staticMap', staticMap)
        # mapTo = np.zeros(len(staticMap))
        # for j in range(0, len(staticMap)):
        #     m = staticMap[:, 2].argmax()
        #     if staticMap[m][0] != staticMap[m][1]:
        #         if mapTo[staticMap[m][1]] == 0:
        #             mapTo[staticMap[m][1]] = staticMap[m][0]
        #         staticMap[m][2] = 0
        #     else:
        #
        # print('mapTo', mapTo)

        print('Time of kmeans', time.time() - beginTime)

        # self.saNodeColors = KMeans(n_clusters=3, random_state=9).fit_predict(embeddings_2d)

    def em(self):
        beginTime = time.time()
        # todo
        print('Time of Gaussian EM', time.time() - beginTime)

    def gaa(self):
        beginTime = time.time()
        # todo
        print('Time of Group Average Agglomerative', time.time() - beginTime)

    def reducting(self):
        print('reductionMethod', self.reductionMethod)
        if self.reductionMethod == 't-SNE':
            self.tsne()
        elif self.reductionMethod == 'NONE':
            return
        elif self.reductionMethod == 'Isomap':
            self.isomap()
        elif self.reductionMethod == 'LLE':
            self.lle()
        elif self.reductionMethod == 'MLLE':
            self.mlle()
        elif self.reductionMethod == 'HLLE':
            self.hlle()
        elif self.reductionMethod == 'Spectral':
            self.spectral()
        elif self.reductionMethod == 'LTSA':
            self.ltsa()
        elif self.reductionMethod == 'MDS':
            self.mds()

        self.nodeXs.clear()
        self.nodeEdges.clear()
        self.attriEdges.clear()
        self.cur_xlim = ''

    def hsca(self):
        crs = open("HSCA_D128_FEATURES_CORA.txt", "r")
        rows = [row.strip().split(',') for row in crs]
        embeddings = np.array(rows)
        embeddings = embeddings.astype(np.float)
        print('hsca embedding', embeddings)
        tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=30)  # random_state=7,
        embeddings_2d = tsne.fit_transform(embeddings)
        print('embedding_2d', embeddings_2d)
        nodePosition = dict(zip(self.attriGraph.nodes(), embeddings_2d))
        print('nodePosition', nodePosition)
        self.nodePosition.update(nodePosition)

    def tsne(self):
        beginTime = time.time()
        print('2: TSNE Begin')
        if self.model == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("These is no model.")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        gn = [x for x in self.model.vocab if len(x) > 0]
        embeddings = np.array([self.model[x] for x in gn])
        # print('embeddings', embeddings)
        tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=30)  # random_state=7,
        embeddings_2d = tsne.fit_transform(embeddings)
        # print('embeddings_2d', embeddings_2d)
        # todo: fix reducting (position) or fix embedding method
        # self.nodePosition = dict(zip(self.model.vocab, embeddings_2d))
        nodePosition = dict(zip(self.model.vocab, embeddings_2d))
        # print('nodePosition', nodePosition)
        self.nodePosition.update(nodePosition)
        # print('nodePosition', self.nodePosition)
        print('Time of TSNE', time.time() - beginTime)

    def isomap(self):
        beginTime = time.time()
        # todo
        print('Time of Isomap', time.time() - beginTime)

    def lle(self):
        beginTime = time.time()
        # todo
        print('Time of Locally linear embedding', time.time() - beginTime)

    def mlle(self):
        beginTime = time.time()
        # todo
        print('Time of Modified Locally Linear Embedding', time.time() - beginTime)

    def hlle(self):
        beginTime = time.time()
        # todo
        print('Time of Hessian Eigenmapping', time.time() - beginTime)

    def spectral(self):
        beginTime = time.time()
        # todo
        print('Time of Spectral Embedding', time.time() - beginTime)

    def ltsa(self):
        beginTime = time.time()
        # todo
        print('Time of Local tangent space alignment', time.time() - beginTime)

    def mds(self):
        beginTime = time.time()
        # todo
        print('Time of Multi-dimensional Scaling', time.time() - beginTime)

    def colorMapping(self):
        needRefresh = False
        if self.colorMethod == 'Unitary':
            self.saNodeColors = NODE_SIZE
            needRefresh = True
        elif self.colorMethod == 'Attribute':
            needRefresh = self.attri2Color()
        elif self.colorMethod == 'Cluster':
            needRefresh = self.cluster2Color()
        elif self.colorMethod == 'Class':
            needRefresh = self.class2Color()
        if needRefresh:
            self.nodeColors.clear()
            self.plot()

    def cluster2Color(self):
        print('4: cluster2Color Begin')
        if self.cluster:
            c2c = [self.clusterColors[v % len(self.clusterColors)] for v in self.cluster.values()]
            # print('c2c', c2c)
            self.saNodeColors = dict(zip(self.cluster.keys(), c2c))
            # print('nodeColors', self.saNodeColors)
            return True
        return False

    def attriMean(self, v):
        if isinstance(v, list):
            # todo multi label color mapping
            if len(v) > 0:
                return v[0]
            else:
                return 0
        else:
            return v

    def attri2Color(self):
        if self.attriGraph == '':
            return False
        self.graph = self.attriGraph
        print('4: attri2Color Begin')
        if nx.get_node_attributes(self.graph, 'value'):
            # todo value is not num. v.isnumeric()
            c2c = [self.clusterColors[int(self.attriMean(v)) % len(self.clusterColors)] for v in
                   nx.get_node_attributes(self.graph, 'value').values()]
            print('c2c', c2c)
            self.saNodeColors = dict(zip(self.graph.nodes(), c2c))
            return True
        # print('nodeColors', self.saNodeColors)
        return False

    def class2Color(self):
        if self.attriGraph == '':
            return False
        self.graph = self.attriGraph
        print('4: class2Color Begin')
        if nx.get_node_attributes(self.graph, 'class'):
            # todo value is not num. v.isnumeric()
            c2c = [self.clusterColors[int(self.attriMean(v)) % len(self.clusterColors)] for v in
                   nx.get_node_attributes(self.graph, 'class').values()]
            print('c2c', c2c)
            self.saNodeColors = dict(zip(self.graph.nodes(), c2c))
            return True
        # print('nodeColors', self.saNodeColors)
        return False

    def sizeMapping(self):
        needRefresh = False
        if self.sizeMethod == 'Unitary':
            self.saNodeSizes = NODE_SIZE
            needRefresh = True
        elif self.sizeMethod == 'Degree':
            needRefresh = self.degree2Size()
        elif self.sizeMethod == 'Class':
            needRefresh = self.class2Size()
        if needRefresh:
            self.nodeSizes.clear()
            self.plot()

    def degree2Size(self):
        d2s = []
        if self.attriGraph:
            d2s = [nx.degree(self.attriGraph)[x] * 10 if nx.degree(self.attriGraph)[x] * 10 > 50 else 50
                   for x in self.attriGraph.nodes()]
        elif self.nodeGraph:
            d2s = [nx.degree(self.graph)[x] * 10 if nx.degree(self.graph)[x] * 10 > 50 else 50
                   for x in self.nodeGraph.nodes()]
        else:
            return False
        self.saNodeSizes = dict(zip(self.nodeGraph.nodes(), d2s))
        # print('nodeSizes', self.saNodeSizes)
        return True

    def class2Size(self):
        if self.attriGraph == '':
            return False
        self.graph = self.attriGraph
        print('4: attri2Color Begin')
        if nx.get_node_attributes(self.graph, 'class'):
            # todo value is not num. v.isnumeric()
            cv = nx.get_node_attributes(self.graph, 'class').values()
            c2s = [len(v) * 10 if len(v) > 0 else 5 for v in cv]
            print('c2s', c2s)
            self.saNodeSizes = dict(zip(self.graph.nodes(), c2s))
            return True
        # print('nodeColors', self.saNodeColors)
        return False

    def shapeMethodChanged(self):
        self.shapeMethod = self.sender().currentText()
        self.shapeMapping()

    def shapeMapping(self):
        needRefresh = False
        if self.shapeMethod == 'Unitary':
            self.saNodeShapes = NODE_SHAPE
            needRefresh = True
        elif self.shapeMethod == 'Value':
            needRefresh = self.attri2Shape()
        elif self.shapeMethod == 'Class':
            needRefresh = self.class2Shape()
        if needRefresh:
            self.nodeShapes.clear()
            self.plot()

    def attri2Shape(self):
        if self.attriGraph == '':
            return False
        self.graph = self.attriGraph
        print('4: attri2Shape Begin')
        if nx.get_node_attributes(self.graph, 'value'):
            # todo value is not num. v.isnumeric()
            vv = nx.get_node_attributes(self.graph, 'value').values()
            v2s = [shape[int(v)] for v in vv]
            # print('v2s', v2s)
            self.saNodeShapes = dict(zip(self.graph.nodes(), v2s))
            return True
        # print('nodeColors', self.saNodeColors)
        return False

    def class2Shape(self):
        if self.attriGraph == '':
            return False
        self.graph = self.attriGraph
        print('4: class2Shape Begin')
        if nx.get_node_attributes(self.graph, 'class'):
            # todo value is not num. v.isnumeric()
            cs = nx.get_node_attributes(self.graph, 'class').values()
            c2s = [shape[int(v)] for v in cs]
            print('c2s', c2s)
            self.saNodeShapes = dict(zip(self.graph.nodes(), c2s))
            return True
        # print('nodeColors', self.saNodeColors)
        return False

    def graphLayout(self):
        if self.graph == '':
            return
        # self.nodePosition = graphviz_layout(self.graph, prog="sfdp")
        if 0:
            self.nodePosition = nx.spring_layout(self.graph, iterations=200)
        else:
            forceatlas2 = ForceAtlas2(
                # Behavior alternatives
                outboundAttractionDistribution=False,  # Dissuade hubs
                linLogMode=False,  # NOT IMPLEMENTED
                adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                edgeWeightInfluence=1.0,

                # Performance
                jitterTolerance=1.0,  # Tolerance
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                multiThreaded=False,  # NOT IMPLEMENTED

                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,

                # Log
                verbose=True)
            self.nodePosition = forceatlas2.forceatlas2_networkx_layout(self.graph, pos=None, iterations=1)

    def curFunction(self, xlim, ylim):
        # print('x,y', xlim, ylim)
        self.cur_xlim = xlim
        self.cur_ylim = ylim
        if self.oax:
            if self.oax.patches:
                self.overviewRec.remove()
            self.overviewRec = patches.Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0], fill=False)
            # print('self.overviewRec', self.overviewRec)
            self.oax.add_patch(self.overviewRec)
            # self.overviewRec = self.oax.bar(xlim[0], ylim[1]-ylim[0], xlim[1]-xlim[0], ylim[0], align='edge', zorder=3)[0]
            # dr = DraggableRectangle(self.overviewRec)
            # dr.connect()
        self.overViewWidget.plot()

    def plot(self):
        return


Attri2vecTest('./datasets/webkb')
#Attri2vecTest()
print('Finish.')
