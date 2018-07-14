# -*- coding: utf-8 -*-
from data import *
from numpy import *
import numpy as np
import random
#from networkx import *
import networkx as nx
import copy
import time
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import linear_model
######################################
#        工序和机器编码              #
# ********************************** #
# 基于工序的编码
def make_process_code(a):
    """a-->workpiece_process [3, 4, 5, 4, 3, 4, 3, 4, 5, 4, 3, 5, 3, 4, 5, 4, 3, 4, 3, 4]"""
    list1 = []
    for i in range(len(a)):
        for j in range(a[i]):
            list1.append(i)
    random.shuffle(list1)
    return list1
#process_code = make_process_code(workpiece_process)  # 基于工序的编码
#print "基于工序的编码:%s" % process_code


def count(a, b, c):
    """列表a中数字b从0到c出现的次数"""
    num = 0
    for i in range(0, c+1):
        if a[i] == b:
            num += 1
    return num


# 基于机器的编码
def make_machine_code(a, b):
    """根据工序的编码以及每道工序可选机器生成机器编码
       a-->workpiece_machine [[[0, 5, 8, 10, 17, 49, 64], [1, 7, 13, 14, 28]]]
       b-->process_code [11, 15, 2, 0, 1, 8, 3, 13, 16, 12, 6, 0, 19, 3, 14, 18,
       15, 7, 2, 1, 2, 8, 7, 2, 16, 7, 5, 15, 10, 13, 17, 9, 4, 2, 14, 11, 7, 10,
       8, 12, 1, 3, 11, 17, 5, 6, 0, 12, 11, 4, 19, 11, 6, 9, 18, 4, 5, 15, 17, 9,
       17, 19, 8, 16, 19, 3, 10, 5, 13, 14, 14, 14, 18, 1, 13, 9, 8]
    """
    list1 = []
    for i in range(len(b)):
        num = count(b, b[i], i)
        c = b[i]
        list1.append(random.choice(a[c][num-1]))
    return list1
#machine_code = make_machine_code(workpiece_machine, process_code)
#print "基于机器的编码：%s" % machine_code


################################################
#                 初始化种群                   #
# ******************************************** #
def make_initial_process_population(a, num):
    """初始化工序编码种群
       a-->workpiece_process
       num-->种群中个体的数量"""
    list1 = []
    for i in range(num):
        list1.append(make_process_code(a))
    return list1
#process_code_population = make_initial_process_population(workpiece_process, 50)
#print "工序种群编码:%s" % process_code_population  # 初始化工序编码种群


def make_initial_machine_population(a, b):
    """初始化机器编码种群
       a-->workpiece_machine[[[0, 5, 8, 10, 17, 49, 64], [1, 7, 13, 14, 28]]]
       b-->process_code_population"""
    list1 = []
    for i in range(len(b)):
        list1.append(make_machine_code(a, b[i]))
    return list1
#machine_code_population = make_initial_machine_population(workpiece_machine, process_code_population)
#print "机器种群编码:%s" % machine_code_population   # 初始化机器编码种群


# 计算模块度
def calculate_modularity(a, b, n, c):
    """a-->process_code
       b-->machine_code
       n-->单元数
       c-->machine_unit机器所属单元"""
    workpiece = [[] for i in range(n)]
    for i in range(len(a)):
        for j in range(n):
            if a[i] == j:
                workpiece[j].append(b[i])
    A = zeros((67, 67))
    for i in range(len(workpiece)):
        for j in range(len(workpiece[i])):
            if j+1 < len(workpiece[i]):
                A[workpiece[i][j]][workpiece[i][j+1]] += 1
    A = mat(A)
    b1 = zeros((67, 50))
    num = 0
    d = 0
    for i in range(n):
        d += c[i]
        while num < d:
            b1[num][i] = 1
            num += 1
    b2 = b1.T
    m = (A.sum(axis=1)).sum(axis=0)[0, 0] / 2
    k = A.sum(axis=1)
    B = A - (k * k.T) * (float(1 / m)) * 0.5
    d = b2 * B * b1
    modularity1 = 1/m*np.trace(d)*0.5
    return modularity1


# 从种群中挑选出模块度最大的50个个体
def choose_best_population(a, n):
    """a-->workpiece_process
       n-->从n个个体中挑选出模块度最大的五十个个体
       """
    process_code_population1 = make_initial_process_population(a, n)
    machine_code_population1 = make_initial_machine_population(workpiece_machine, process_code_population1)
    modularity_value1 = []
    for i in range(len(process_code_population1)):
        modularity_value1.append(calculate_modularity(process_code_population1[i], machine_code_population1[i],
                                                     20, machine_unit))

    list1 = range(n)
    dict1 = dict(zip(list1, modularity_value1))
    return sorted(dict1.items(), key=lambda item: item[1], reverse=True)





#####################################################
#                   优化目标                        #
# ************************************************* #


# 运输距离
def calculate_transport_distance(a, b):
    """如果机器a, b在同一个单元内则距离为零"""
    list1 = []
    for i in range(len(unit_machine)):
        for j in range(len(unit_machine[i])):
            if unit_machine[i][j] == a:
                list1.append(i)
                break
    for i in range(len(unit_machine)):
        for j in range(len(unit_machine[i])):
            if unit_machine[i][j] == b:
                list1.append(i)
                break
    if list1[0] == list1[1]:
        distance = 0
    else:
        distance = unit_distance[(list1[0], list1[1])]
    return distance


# 成本
def calculate_cost(a, b, c):
    """计算每个零部件的成本
       a-->process_code工序编码
       b-->machine_code机器编码
       c-->workpiece_cost零件工序在机器上的加工费用
       """
    # 零部件的工序加工费用
    cost = [[]for i in range(len(a))]  # 50个零部件的加工费用
    element = []
    for l in range(len(a)):
        if a[l] in element:
            continue
        else:
            element.append(a[l])
            cost[l].append(c[a[l]][0][b[l]])
            for i in range(l+1, len(a)):
                if a[i] == a[l]:
                    num = count(a, a[i], i)   # 列表a中a[i]从0到i出现的次数
                    cost[l].append(c[a[l]][num-1][b[i]])
    return cost


# 最大完工时间
def calculate_max_makespan(a, b):
    """计算每个零部件成本
           a-->process_code工序编码
           b-->machine_code机器编码
           c-->workpiece_machine_time每个零部件在机器上的加工时间"""
    matrix1 = np.zeros((67, 50))    # 根据工序编码和机器编码生成每个零部件在每个机器上的加工时间
    for i in range(len(a)):
        try:
            matrix1[(b[i], a[i])] = workpiece_machine_time[(a[i], b[i])]
        except KeyError:
            pass
    matrix2 = np.zeros((67, 50))   # 根据工序编码和机器编码生成每个零部件在每个机器上的完工时间
    distance = 0
    for i in range(len(a)):
        try:
            max_column = matrix2.max(axis=0)  # 每列最大值
            max_row = matrix2.max(axis=1)  # 每行最大值
            for j in range(i-1, -1, -1):     # 求出当前工序和紧前工序加工机器之间的距离
                if a[i] == a[j]:
                    p = b[i]
                    q = b[j]
                    distance = machine_distance(p, q)
                    pass
            matrix2[(b[i], a[i])] = max(max_column[a[i]]+distance, max_row[b[i]]) + workpiece_machine_time[(a[i], b[i])]

        except KeyError:
            pass
    return matrix2.max()
#max_makespan = calculate_max_makespan(process_code, machine_code)  #最大完工时间
#print "最大完工时间：%s" % max_makespan


# 所有零件总成本
def calculate_total_cost(a, b):
    """计算每个零部件的成本
       a-->process_code工序编码
       b-->machine_code机器编码
       """
    # 零部件的工序加工费用
    cost = []  # 20个零部件的加工费用
    for i in range(len(a)):
        cost.append(workpiece_machine_cost[a[i], b[i]])
    # 机器之间的运输费用
    for i in range(len(a)):
        for j in range(i - 1, -1, -1):  # 求出当前工序和紧前工序加工机器之间的距离
            if a[i] == a[j]:
                p = b[i]
                q = b[j]
                cost.append(0.2*machine_distance(p, q))
    return sum(cost)
#total_cost = calculate_total_cost(process_code, machine_code)
#print "总的成本：%s" % total_cost


#####################################################
#                   小世界遗传算法                  #
# ************************************************* #


# 生成小世界网络
def small_world_population(a, b, c):
    """a-->种群中个体数量
       b-->每个节点的相邻节点
       c-->节点重连概率"""
    ER = nx.random_graphs.watts_strogatz_graph(a, b, c)  # 生成包含a个节点、b个相邻节点、以概率0.2连接的随机图
    pos = nx.shell_layout(ER)
    #nx.draw(ER, pos, with_labels=True, node_size=100)
    #print ER.nodes()
    #print ER.neighbors(1)
    #plt.show()
    list_neighbour = [[] for i in range(100)]  # 节点邻居
    for i in range(100):
        list_neighbour[i].extend(ER.neighbors(i))
    return list_neighbour


#list_neighbours = small_world_population(50, 4, 0.3)  # 每个节点的相邻节点
#print list_neighbours


# 计算每个节点和相邻节点的欧式距离
#def calculate_crowding_distance(list1, list2):
#    crowding_distance = pow((list1[0] - list2[0]), 2) + pow((list1[1] - list2[1]), 2)
#    return crowding_distance


def compare(p, q):
    """比较大小"""
    return p <= q


def dominates(P, Q):
    bools = []
    for p, q in zip(P, Q):
        bools.append(compare(p, q))
    return all(bools)


def calculate_two_objective(a, b):
    """  计算每个个体的两个目标值
        a-->process_code工序编码
       b-->machine_code机器编码"""
    two_objective = []
    two_objective.append(calculate_max_makespan(a, b))
    two_objective.append(calculate_total_cost(a, b))
    return two_objective
#two_objective_list = calculate_two_objective(process_code, machine_code)


# 找出每个个体相邻最好的个体
def dominate_sort(a, b, c, p):
    """ a-->list_neighbours #相邻节点
        b-->                #工序编码种群
        c-->                #机器编码种群
        p-->交配的概率如果交配则挑选出最好的个体"""
    objective_list = [[] for i in range(len(a))]
    list_neighbours1 = copy.deepcopy(a)
    best_neighbours = [[]for i in range(len(a))]  # 最好的个体列表
    for i in range(len(a)):
        if random.random() < p:  # 判断是否要挑选出最好的个体
            for j in range(len(a[i])):
                objective_list[i].append(calculate_two_objective(b[a[i][j]], c[a[i][j]]))
        else:
            objective_list[i].append(0)
    for i in range(len(a)):
        if len(objective_list[i]) == 1:
            list_neighbours1[i] = [-1]
        else:
            for j in range(len(a[i])):
                for l in range(len(a[i])):
                    if j == l:
                        continue
                    if dominates(objective_list[i][j], objective_list[i][l]):
                        list_neighbours1[i][l] = -1
    list_neighbours2 = [[] for i in range(len(a))]
    for i in range(len(a)):
        for j in range(len(list_neighbours1[i])):
            if list_neighbours1[i][j] > 0:
                list_neighbours2[i].append(list_neighbours1[i][j])
    for i in range(len(a)):
        if len(list_neighbours2[i]) == 0:
            pass
        else:
            best_neighbours[i].append(random.choice(list_neighbours2[i]))
    return best_neighbours
#best_neighbours = dominate_sort(list_neighbours, process_code_population, machine_code_population, 0.3)
#print best_neighbours


############################################
#                交叉变异                  #
# **************************************** #
def crossover(a, b, c):
    """种群中的个体与最好的相邻节点相互交配
        a-->process_code_population
        b-->best_neighbours
        c-->machine_code_population"""
    P1 = []
    P2 = []
    J2 = range(25, 50)
    Q1 = []
    Q2 = []
    new_process_code_population1 = []
    new_machine_code_population1 = []
    for i in range(len(b)):
        if len(b[i]) == 0:
            new_process_code_population1.append(a[i])
            new_machine_code_population1.append(c[i])
        else:
            P1.append(a[i])                 # 原始种群中选择交配的个体
            P2.append(a[b[i][0]])           # 最好的相邻节点
            Q1.append(c[i])
            Q2.append(c[b[i][0]])
    # 把工件分为两个部分第一个部分为0-9，第二个部分为10-19
    C1 = [[] for i in range(len(P1))]
    C2 = [[] for i in range(len(P1))]
    for i in range(len(P1)):
        for j in range(len(P1[i])):
            if P2[i][j] in J2:
                C1[i].append(P2[i][j])
                C2[i].append(Q2[i][j])

    num = 0
    for i in range(len(P1)):
        for j in range(len(P1[i])):
            if P1[i][j] in J2:
                try:
                    P1[i][j] = C1[i][num]
                    Q1[i][j] = C2[i][num]
                    num += 1
                except IndexError:
                    pass
    for i in range(len(P1)):
        new_process_code_population1.append(P1[i])
        new_machine_code_population1.append(Q1[i])
    new_code_population1 = []
    new_code_population1.append(new_process_code_population1)
    new_code_population1.append(new_machine_code_population1)
    return new_code_population1
#new_code_population = crossover(process_code_population, best_neighbours, machine_code_population)
#new_process_code_population = new_code_population[0]
#new_machine_code_population = new_code_population[1]


# 机器编码基因突变
def mutation(a, b, p):
    """a-->new_process_code_population
       b-->new_machine_code_population
       p 基因突变概率"""
    for i in range(len(a)):
        if random.random()< p:
            list1 = make_machine_code(workpiece_machine, a[i])
            list2 = []
            for j in range(10):
                list2.append(random.randint(0, 70))
            for l in range(len(list2)):
                b[i][l] = list1[l]
    return b
#new_machine_code_population = mutation(new_process_code_population, new_machine_code_population, 0.5)
#print new_machine_code_population


# 挑选出最好的结果
def choose_best_result(a):
    """a-->result"""
    list1 = []
    list2 = []
    for i in range(len(a)):
        for j in range(len(a)):
            if i == j:
                continue
            if dominates(a[i], a[j]):
                list1.append(a[j])
    for i in range(len(a)):
        if a[i] not in list1:
            list2.append(a[i])

    return list2

process_time = []
all_makespan = []
all_cost = []

N = 1
iterations = 1  # 代数
# 运行次数10次
for i in range(N):
    start = time.clock()
    process_code = make_process_code(workpiece_process)
    machine_code = make_machine_code(workpiece_machine, process_code)
    process_code_population = make_initial_process_population(workpiece_process, 100)
    machine_code_population = make_initial_machine_population(workpiece_machine, process_code_population)
    modularity_value = choose_best_population(workpiece_process, 50)
    process_code_population1 = []
    machine_code_population1 = []
    other_process_code_population = make_initial_process_population(workpiece_process, 100)
    other_machine_code_population = make_initial_machine_population(workpiece_machine, other_process_code_population)
    for i in range(50):
        process_code_population1.append(process_code_population[modularity_value[i][0]])
        machine_code_population1.append(machine_code_population[modularity_value[i][0]])
        process_code_population1.append(other_process_code_population[i])
        machine_code_population1.append(other_machine_code_population[i])
    result = []
#print process_code
#print machine_code
# 繁殖代数10， 20， 30， 40， 50， 100， 200， 300， 400， 500， 700， 1000
    for i in range(iterations):
        list_neighbours = small_world_population(100, 2, 0.3)
        best_neighbours = dominate_sort(list_neighbours, process_code_population1, machine_code_population1, 0.3)
        new_code_population = crossover(process_code_population1, best_neighbours, machine_code_population1)
        new_process_code_population = new_code_population[0]
        new_machine_code_population = new_code_population[1]
        new_machine_code_population = mutation(new_process_code_population, new_machine_code_population, 0.5)
        process_code_population1 = new_process_code_population
        machine_code_population1 = new_machine_code_population
    for i in range(len(process_code_population1)):
        two_objective_list = calculate_two_objective(process_code_population1[i], machine_code_population1[i])
        result.append(two_objective_list)
    print result
    best_result = choose_best_result(result)
    print best_result
    end = time.clock()-start
    print "运行时间：%s" % end

    process_time.append(end)
    for i in range(len(best_result)):
        all_makespan.append(best_result[i][0])
        all_cost.append(best_result[i][1])


# 数据分析


# 求平均值
def average_value(a):
    b = sum(a)/(len(a))
    return b
average_process_time = average_value(process_time)
average_all_makespan = average_value(all_makespan)
average_all_cost = average_value(all_cost)
print "平均运行时间：%s" % average_process_time
print "平均完工时间：%s" % average_all_makespan
print "平均总成本：%s" % average_all_cost
process_code_population3 = make_initial_process_population(workpiece_process, 100)
machine_code_population3 = make_initial_machine_population(workpiece_machine, process_code_population3)
modularity = []
result1 = []
for i in range(len(process_code_population3)):
    modularity.append(calculate_modularity(process_code_population3[i], machine_code_population3[i], 20, machine_unit))
for i in range(len(process_code_population3)):
    two_objective_list2 = calculate_two_objective(process_code_population3[i], machine_code_population3[i])
    result1.append(two_objective_list2)


result2 = []
for i in range(len(result1)):
    result2.append(result1[i][1])
plt.plot(modularity, result2, 'ro')
data = array([modularity, result2])
cov(data, bias=1)
#print corrcoef(data)
plt.show()


list3 = [[] for i in range(50)]
G = nx.Graph()
for i in range(len(process_code_population3[0])):
    for j in range(50):
        if process_code_population3[0][i] == j:
            list3[j].append(machine_code_population3[0][i])
print list3
for i in range(len(list3)):
    for j in range(len(list3[i])):
        if j+1 < len(list3[i]):
            G.add_edge(list3[i][j], list3[i][j+1])

nx.draw(G, pos=nx.spectral_layout(G), with_labels=True,node_size=30)
plt.show()
degree = nx.degree_histogram(G)          #返回图中所有节点的度分布序列
x = range(len(degree))                             #生成x轴序列，从1到最大度
y = [z / float(sum(degree)) for z in degree]
#将频次转换为频率，这用到Python的一个小技巧：列表内涵，Python的确很方便：）
X_parameter = []
Y_parameter = []
for single_square_feet, single_price_value in zip(x, y):
    X_parameter.append([float(single_square_feet)])
    Y_parameter.append(float(single_price_value))

# 模型拟合
regr = linear_model.LinearRegression()
regr.fit(X_parameter, Y_parameter)
# 模型结果与得分
print('Coefficients: \n', regr.coef_,)
print("Intercept:\n", regr.intercept_)
# The mean square error
print("Residual sum of squares: %.8f"
      % np.mean((regr.predict(X_parameter) - Y_parameter) ** 2))  # 残差平方和

# 可视化

plt.scatter(X_parameter, Y_parameter, color='black')
plt.plot(X_parameter, regr.predict(X_parameter), color='blue', linewidth=3)

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.xlabel(u'节点度', fontproperties=font)
plt.ylabel(u'频数', fontproperties=font)
plt.loglog(x, y, 'ro')           #在双对数坐标轴上绘制度分布曲线
plt.show()
print nx.average_clustering(G)
#pathlengths =[]
#for v in G.nodes():
#    spl=single_source_shortest_path_length(G,v)
#    print('%s %s' % (v,spl))
#    for p in spl.values():
#        pathlengths.append(p)
#print "平均路径长度：%s" % (sum(pathlengths)/len(pathlengths))