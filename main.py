import math
import random
import sys
from time import perf_counter
from copy import deepcopy
from itertools import chain

import requests
from geopy import distance as geodesic
import scipy.spatial.distance as distance
import tabulate
import numpy as np                 # v 1.19.2
import matplotlib.pyplot as plt    # v 3.3.2
import queue
import csv
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from pulp import *
from unionfind import unionfind



def getCellTowersInArea(latmin, latmax, longmin, longmax):
    apitoken = 'pk.cea749ff23714a31c6e9d2563c7d68d6'
    url_getcells = 'http://www.opencellid.org/cell/getInArea'
    payload_string = f'?key={apitoken}&BBOX={latmin},{longmin},{latmax},{longmax}&format=json'
    count = 1000
    iter = 0

    f = open('celltowers_munich.csv', 'w')
    while count == 1000:
        r = requests.get(url_getcells + payload_string + f'&offset={iter * 1000}')
        response = r.json()
        count = response['count']
        for row in response['cells']:
            lat = row['lat']
            lon = row['lon']
            f.write(f'{lat},{lon}\n')
        print(f'run {iter} with status code {r.status_code}')
        iter += 1
    f.close()


def sample_coords(amount, maxX, maxY):
    return random.sample([(x,y) for x in range(0,maxX) for y in range(0,maxY)], amount)


def read_coords_from_csv(filename, amount, alpha = 2, latindex=0, lonindex=1):
    rows = []
    with open(filename, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append((float(row[latindex]),float(row[lonindex])))
    coords = random.sample(rows, amount)
    dist = []
    for s1 in coords:
        dist_row = []
        for s2 in coords:
            dist_row.append(math.pow(geodesic.distance((s1[latindex], s1[lonindex]), (s2[latindex], s2[lonindex])).m, alpha))
        dist.append(dist_row)
    return dist,coords


def calculate_distances(coords, alpha=2):
    dist = []
    for x1, y1 in coords:
        dist_i = []
        for x2, y2 in coords:
            dist_i.append(math.pow(distance.euclidean((x1, y1), (x2, y2)), alpha))
        dist.append(dist_i)
    return dist


def calculate_sets(dist, isRounded=False):
    amount_stations = len(dist)
    dist = np.array(dist)
    dist = dist[:,1:]
    dist_sorted = dist.argsort()
    dist_final = [[]*(len(dist)) for i in range(amount_stations)]
    for i in range(0, amount_stations):
        for j in range(1, len(dist)):
            if i == 0:
                setij = set()
                for x in dist_sorted[i][0:j]:
                    setij.add(x+1)
                setij.add(0)
                if not isRounded:
                    dist_final[i].append((setij, dist[i][dist_sorted[i][j-1]]))
                else:
                    dist_final[i].append((setij, round(dist[i][dist_sorted[i][j-1]])))
            else:
                setij = set()
                for x in dist_sorted[i][1:j]:
                    setij.add(x+1)
                if len(setij) == 0:
                    continue
                if not isRounded:
                    dist_final[i].append((setij, dist[i][dist_sorted[i][j-1]]))
                else:
                    dist_final[i].append((setij, round(dist[i][dist_sorted[i][j-1]])))
    return dist_final#[*zip(*dist_final)] #transposes set matrix

# add range circles
def draw_graph(coords, edges=[], ranges=[], title='', fname='', cost=0):
    ticks_frequency = 1
    # Plot points
    coords = np.array(coords)
    xs = coords[:,0].flatten()
    ys = coords[:,1].flatten()

    plt.style.use('ggplot')
    if len(title) > 0:
        plt.suptitle(title, fontsize=16, fontweight='bold')
    if len(edges) > 0:
        edges = np.array(edges)
        plt.plot(xs[edges.T], ys[edges.T], linestyle='solid', color='#3E4999', alpha=0.5)

    plt.plot(xs,ys, 'ro', markerfacecolor='#FF8811', markeredgecolor='#FF8811')
    plt.plot(xs[0], ys[0], 'bo', markerfacecolor='#F8333C', markeredgecolor='#F8333C')
    for i,(x,y) in enumerate(zip(xs,ys)):
        plt.annotate(i, xy=(x + 0.02,y - 0.05))
    if cost > 0:
        plt.title(f'Total cost: {round(cost)}', fontsize=7, pad=10, loc='left')
    if len(fname) > 0:
        plt.savefig(fname, format='pdf')
        plt.close()
    else:
        plt.show()


def draw_rays(amount, isRay=False, hasUnitCircle=False, f=0.0):
    plt.style.use('ggplot')
    xs = [math.sin(2 * math.pi * i / amount + f) for i in range(amount)]
    ys = [math.cos(2 * math.pi * i / amount + f) for i in range(amount)]
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    scalar = 2 if isRay else 1
    for i in range(amount):
        plt.plot([0, scalar * xs[i]], [0, scalar * ys[i]], linestyle='solid', color='#3E4999', alpha=0.5)
    if hasUnitCircle:
        plt.gca().add_patch(plt.Circle((0, 0), 1, color='#3E4999', fill=False))
    plt.plot(0, 0, 'bo', markerfacecolor='#F8333C', markeredgecolor='#F8333C')
    plt.plot(xs,ys, 'ro', markerfacecolor='#FF8811', markeredgecolor='#FF8811')

    plt.xticks([])
    plt.yticks([])
    plt.show()


def init_mst(adjmatrix):
    (cost, ranges, edges) = prim(adjmatrix, is_bip=False)
    return cost, ranges, edges


def init_bip(adjmatrix):
    (cost, ranges, edges) = prim(adjmatrix, is_bip=True)
    return cost, ranges, edges


def prim(adjmatrix, is_bip=False):
    q = queue.PriorityQueue()
    dist = np.array(adjmatrix)
    reached_by = [-1 for i in range(len(adjmatrix))]
    ranges = np.array([0 for i in range(len(adjmatrix))])

    for i, x in enumerate(dist[0]):
        q.put((x, (0,i)))
    while not q.empty():
        (prio, (u,v)) = q.get()
        if reached_by[v] > -1:
            continue
        else:
            reached_by[v] = u
            ranges[u] = dist[u][v]
            for i, x in enumerate(dist[v]):
                q.put((x, (v,i)))
            if is_bip:
                for i, x in enumerate(dist[u]):
                    q.put((x-ranges[u], (u,i)))

    edges = []
    for u,v in enumerate(reached_by):
        if v < 0:
            continue
        edges.append([u,v])
    return sum(ranges), ranges, edges


def init_gsv(adjmatrix):
    (cost, ranges, edges) = gsv(adjmatrix)
    return cost, ranges, edges


def gsv(adjmatrix):
    sets = calculate_sets(adjmatrix)
    discovered = set()
    ranges = [0 for i in range(len(adjmatrix))]
    parents = [-1 for i in range(len(adjmatrix))]
    discovered.add(0)
    while len(discovered) < len(adjmatrix):
        min_cost =  sys.maxsize * 2 + 1
        min_power = sys.maxsize * 2 + 1
        newly_discovered = set()
        min_cost_station = -1
        for discovered_station in discovered:
            for (reachable, dist) in sets[discovered_station]:
                undiscovered_stations = reachable.difference(discovered)
                if len(undiscovered_stations) > 0 and min_cost > dist / len(undiscovered_stations):
                    min_cost = dist / len(undiscovered_stations)
                    newly_discovered = undiscovered_stations
                    min_cost_station = discovered_station
                    min_power = dist
        ranges[min_cost_station] = min_power
        discovered = discovered.union(newly_discovered)
        for new_station in newly_discovered:
            parents[new_station] = min_cost_station

    edges = []
    for u,v in enumerate(parents):
        if v < 0:
            continue
        edges.append([u, v])
    return sum(ranges), ranges, edges


def gsv_nonconnected(adjmatrix):
    sets = calculate_sets(adjmatrix)
    discovered = set()
    ranges = [0 for i in range(len(adjmatrix))]
    parents = [-1 for i in range(len(adjmatrix))]
    children = [set() for i in range(len(adjmatrix))]
    while len(discovered) < len(adjmatrix):
        min_cost = sys.maxsize * 2 + 1
        min_power = sys.maxsize * 2 + 1
        newly_discovered = set()
        min_cost_station = -1
        min_set = set()
        for i in range(len(adjmatrix)):
            for (reachable, dist) in sets[i]:
                undiscovered_stations = reachable.difference(discovered)
                if len(undiscovered_stations) > 0 and min_cost > dist / len(undiscovered_stations):
                    min_cost = dist / len(undiscovered_stations)
                    newly_discovered = undiscovered_stations
                    min_cost_station = i
                    min_power = dist
                    min_set = reachable
        ranges[min_cost_station] = min_power
        children[min_cost_station] = min_set
        discovered = discovered.union(newly_discovered)
        for new_station in newly_discovered:
            parents[new_station] = min_cost_station


    #connect graph
    dist_sorted = np.array(adjmatrix).argsort().tolist()
    #print(dist_sorted)
    #reachable = children[0].union(set(0))
    #while len(reachable) < len(adjmatrix):
        #min_station = -1
        #min_dist = sys.maxsize * 2 + 1
        #for i in reachable:
            #if adjmatrix[i][dist_sorted[i][0]] < min_dist:
                #min_station = i
                #min_dist = adjmatrix[i][dist_sorted[i][0]]

    edges = []
    for u,v in enumerate(parents):
        if v < 0:
            continue
        edges.append([u, v])
    return sum(ranges), ranges, edges

def init_ca(adjmatrix):
    (cost, ranges, edges) = ca(adjmatrix)
    return cost,ranges,edges


def ca(adjmatrix):
    ranges = [0 for i in range(0,len(adjmatrix))]
    sets = calculate_sets(adjmatrix)
    G = nx.from_numpy_array(np.array(adjmatrix))
    mst = nx.minimum_spanning_tree(G)

    while True:
        maxCE = 0
        center = -1
        power = -1
        newT = mst
        newEdges = []
        for i in range(0,len(adjmatrix)):
            for (reachable, cost) in sets[i]:
                stations = list(reachable)
                old_values = []
                for station in stations:
                    if i == station:
                        continue
                    old_values.append((station,G[i][station]['weight']))
                    G[i][station]['weight'] = 0

                mst_prime = nx.minimum_spanning_tree(G)
                valCE = (mst.size(weight='weight') - mst_prime.size(weight='weight')) / cost
                if valCE > maxCE:
                    newT = mst_prime
                    maxCE = valCE
                    center = i
                    newEdges = reachable

                for (station, weight) in old_values:
                    G[i][station]['weight'] = weight

        if maxCE > 2:
            print(f'found improvement of value {maxCE}')
            for station in newEdges:
                G[center][station]['weight'] = 0
            mst = newT
        else:
            for station in newEdges:
                if not center == station:
                    G[center][station]['weight'] = 0
            mst = newT
            break

    A = set()
    S = set(list(range(0,len(adjmatrix))))
    edges = []
    q = queue.Queue()
    A.add(0)
    q.put(0)
    while not q.empty():
        parent = q.get()
        diffS = S.difference(A)
        children = nx.neighbors(mst, parent)
        maxRange = 0
        for child in children:
            if child in diffS:
                edges.append([parent,child])
                q.put(child)
                A.add(child)
                maxRange = max(maxRange, adjmatrix[parent][child])
        ranges[parent] = maxRange

    return sum(ranges), ranges, edges


def run_algorithms(alpha, amount, filename='NONE', maxX=0, maxY=0, isVisual=False):
    beforeAdjmatrix = perf_counter()
    if filename != 'NONE':
        (adjmatrix,coords) = read_coords_from_csv(filename, amount, alpha=alpha)
    else:
        coords = sample_coords(amount, maxX, maxY)
        adjmatrix = calculate_distances(coords, alpha)
    beforeMst = perf_counter()
    (costMst,rangesMst,edgesMst) = init_mst(adjmatrix)
    beforeBip = perf_counter()
    (costBip,rangesBip,edgesBip) = init_bip(adjmatrix)
    beforeLP = perf_counter()
    (costLP, edgesLP) = calculateLP(adjmatrix, alpha=alpha, isConnected=False)
    beforeCA = perf_counter()
    if amount <= 100:
        (costCA, rangesCA, edgesCA) = init_ca(adjmatrix)
    else:
        (costCA, rangesCA, edgesCA) = (1,1,1)
    afterCA = perf_counter()

    timeAdjmatrix = beforeMst - beforeAdjmatrix
    timeMst = beforeBip - beforeMst
    timeBip = beforeLP - beforeBip
    timeLP = beforeCA - beforeLP
    timeCA = afterCA - beforeCA
    if isVisual:
        draw_graph(coords, edgesMst, 'MST Approximation', f'images/mst_{amount}_{maxX}_{alpha}.pdf', cost=costMst)
        draw_graph(coords, edgesBip, 'BIP Approximation', f'images/bip_{amount}_{maxX}_{alpha}.pdf', cost=costBip)
        draw_graph(coords, edgesCA, 'CA Approximation', f'images/ca_{amount}_{maxX}_{alpha}.pdf', cost=costCA)
    return timeAdjmatrix, timeBip, timeMst, timeLP, timeCA, costBip, costMst, costLP, costCA


def save_results(amountRuns, alpha, amount, filename='NONE', maxX=0, maxY=0):
    timesAdjmatrix = []
    timesBip = []
    timesMst = []
    timesLP = []
    timesCA = []
    costsBip = []
    costsMst = []
    costsLP = []
    costsCA = []

    for i in range(0, amountRuns):
        print(f'starting {i}th run.')
        timeAdjmatrix, timeBip, timeMst, timeGreedy, timeCA, costBip, costMst, costLP, costCA = run_algorithms(alpha, amount, filename=filename, maxX=maxX, maxY=maxY)
        timesAdjmatrix.append(timeAdjmatrix)
        timesBip.append(timeBip)
        timesMst.append(timeMst)
        timesLP.append(timeGreedy)
        timesCA.append(timeCA)
        costsMst.append(costMst)
        costsBip.append(costBip)
        costsLP.append(costLP)
        costsCA.append(costCA)
    f = open('results_new.csv', 'a')
    print(f'Adjacency Matrix:{np.mean(timesAdjmatrix)}, BIP: {np.mean(timesBip)}, '
          f'MST: {np.mean(timesMst)}, Greedy: {np.mean(timesLP)}, Costs MST: {np.mean(costsMst)}, '
          f'Costs BIP/MST: {np.mean(costsBip) / np.mean(costsMst)}, Costs Greedy/MST: {np.mean(costsLP) / np.mean(costsMst)},'
          f'Costs CA / MST: {np.mean(costsCA) / np.mean(costsMst)}')
    f.write(f'{maxX},{amount},{alpha},{np.mean(timesAdjmatrix):.5f},{np.mean(timesBip):.5f},{np.mean(timesMst):.5f},{np.mean(timesLP):.5f},{np.mean(timesCA):.5f},{np.mean(costsMst) /np.mean(costsLP):.4f},{(np.mean(costsBip) / np.mean(costsLP)):.4f},{round(np.mean(costsLP))},{(np.mean(costsCA) / np.mean(costsLP)):.4f}\n')
    f.close()
    f_var = open('results_var.csv', 'a')
    f_var.write(f'{maxX},{amount},{alpha},{np.var(timesAdjmatrix):.5f},{np.var(timesBip):.5f},{np.var(timesMst):.5f},{np.var(timesLP):.5f},{np.var(timesCA):.5f},{np.var(costsMst)},{(np.var(costsBip)):.4f},{(np.var(costsLP)):.4f},{(np.var(costsCA)):.4f}\n')
    f_var.close()


def calculateLP(adjmatrix, alpha=2, isConnected=False):
    sets = calculate_sets(adjmatrix)
    prob = LpProblem("ConnectedSetCover", LpMinimize)
    contained = [list() for i in range(len(adjmatrix))]
    outgoing = [list() for i in range(len(adjmatrix))]
    names = dict()
    xs = list()
    costs = list()
    set_counter = 0
    for i in range(len(sets)):
        setsI = sets[i]
        for (set, cost) in setsI:
            if i == 0:
                set.add(i)
            else:
                set.discard(i)
                set.discard(0) #TODO needed?
                if len(set) == 0:
                    continue
            if 's' + str(abs(hash(f"{i}-{repr(set)}"))) in names:
                continue
            names['s' + str(abs(hash(f"{i}-{repr(set)}")))] = f"{i}-{repr(set)}"
            x = LpVariable('s' + str(abs(hash(f"{i}-{repr(set)}"))), 0, 1)
            xs.append(x)
            outgoing[i].append(set_counter)
            costs.append(cost)
            for station in set:
                contained[station].append(x)
            set_counter += 1

    for reachedBy in contained:
        prob += lpSum(reachedBy) >= 1  # each station contained in at least 1 set
    if isConnected:  #refactor this mess
        edges = np.array([[LpVariable('s' + str(abs(hash(f'edge {i} {j}'))), 0, 0) for j in range(len(xs))] for i in range(len(xs))])
        print('is connected')
        for i,x in enumerate(xs):
            station = int(names[x.getName()].split('-')[0])
            reachable = eval(names[x.getName()].split('-')[1])
            for endStation in reachable:
                for y in outgoing[endStation]:
                    edges[i][y].bounds(low=0,up=len(adjmatrix))
                    prob += edges[i][y] <= len(adjmatrix) * x
            if station > 0:
                prob += lpSum(edges[i]) - lpSum(edges[:,i]) <= -x
    prob += lpDot(xs, costs)  # objective function
    print(len(xs))
    #prob.writeLP('CSV.lp', max_length=1000)
    solver = CPLEX_CMD(path='/opt/ibm/ILOG/CPLEX_Studio221/cplex/bin/x86-64_linux/cplex', options=['set simplex limits iterations 10000', 'set barrier limits iteration 10000'])
    try:
        status = prob.solve(solver)
    except Exception:
        pass
    print(LpStatus[status])
    print(f'total cost: {value(prob.objective)}')
    edges = []
    for x in xs:
        if value(x) > 0.0:
            print(f'{names[x.getName()]}: {value(x)}')
            parent = int(names[x.getName()].split('-')[0])
            children = eval(names[x.getName()].split('-')[1])
            for child in children:
                edges.append([parent, int(child)])
    return value(prob.objective), edges


def calculateLPNew(adjmatrix, alpha=2):
    sets = calculate_sets(adjmatrix)
    prob = LpProblem("ConnectedSetCover", LpMinimize)
    contained = [list() for i in range(len(adjmatrix))]
    outgoing = [list() for i in range(len(adjmatrix))]
    names = dict()
    xs = list()
    costs = list()
    set_counter = 0
    for i in range(len(sets)):
        setsI = sets[i]
        for (set, cost) in setsI:
            if i == 0:
                set.add(i)
            else:
                set.discard(i)
                set.discard(0) #TODO needed?
                if len(set) == 0:
                    continue
            if 's' + str(hash(f"{i}-{repr(set)}")).replace('-','n') in names: #FIXME sometimes collision occurs when 0 is removed from set; find better way
                continue
            names['s' + str(hash(f"{i}-{repr(set)}")).replace('-','n')] = f"{i}-{repr(set)}"
            x = LpVariable('s' + str(hash(f"{i}-{repr(set)}")).replace('-','n'), 0, 1)
            xs.append(x)
            outgoing[i].append(set_counter)
            costs.append(cost)
            for station in set:
                contained[station].append(x)
            set_counter += 1
    print(len(xs))
    print(len(names))
    for reachedBy in contained:
        prob += lpSum(reachedBy) >= 1  # each station contained in at least 1 set
    prob += lpDot(xs, costs)  # objective function
    prob.writeLP('CSV.lp', max_length=1000)
    solver = CPLEX_CMD(path='/opt/ibm/ILOG/CPLEX_Studio221/cplex/bin/x86-64_linux/cplex', options=['set simplex limits iterations 10000', 'set barrier limits iteration 10000'])

    try:
        status = prob.solve(solver)
    except Exception as e:
        print(e)
        pass
    print(LpStatus[status])
    print(f'total cost: {value(prob.objective)}')
    edges = []
    sets = {}
    for x in xs:
        if value(x) > 0.0:
            print(f'{names[x.getName()]}: {value(x)}')
            parent = int(names[x.getName()].split('-')[0])
            children = eval(names[x.getName()].split('-')[1])
            if parent in sets:
                sets[parent] = sets[parent].union(children)
            else:
                sets[parent] = children
            for child in children:
                edges.append([parent, int(child)])
    iteration = 0
    connectedComponents = findConnectedComponents(sets,len(adjmatrix))
    while len(connectedComponents) > 1:
        for connectedComponent in connectedComponents:
            xPrimes = list()
            for station in connectedComponent:
                xPrime = LpVariable(f'p{iteration}_{station}',0,1)
                prob += lpSum(filter(lambda x: (int(names[x.getName()].split('-')[0]) not in connectedComponent),contained[station])) >= xPrime  #potentially wrong
                xPrimes.append(xPrime)
            prob += lpSum(xPrimes) >= 1

        try:
            status = prob.solve(solver)
        except Exception as e:
            print(e)
            pass
        print(LpStatus[status])
        print(f'total cost: {value(prob.objective)}')
        sets = {}
        for x in xs:
            if value(x) > 0.0:
                print(f'{names[x.getName()]}: {value(x)}')
                parent = int(names[x.getName()].split('-')[0])
                children = eval(names[x.getName()].split('-')[1])
                if parent in sets:
                    sets[parent] = sets[parent].union(children)
                else:
                    sets[parent] = children
                for child in children:
                    edges.append([parent, int(child)])
        connectedComponents = findConnectedComponents(sets,len(adjmatrix))
        iteration += 1
    return value(prob.objective), edges

def findConnectedComponents(sets, amountStations): #FIXME change so only if reached by cumulative value of 1 does component count as connected
    u = unionfind(amountStations)
    discovered = [False for i in range(0, amountStations)]
    for i in range(0,amountStations):
        if not discovered[i]:
            bfs(u, i, sets, discovered)
    print(u.groups())
    return u.groups()


def bfs(u, parent, sets, discovered):
    q = queue.Queue()
    q.put(parent)
    while not q.empty():
        station = q.get()
        discovered[station] = True
        if station in sets:
            children = sets.get(station)
            for child in children:
                u.unite(parent, child)
                if not discovered[child]:
                    q.put(child)
                    discovered[child] = True


if __name__ == '__main__':
    maxX = [10,100,100,100,200,200,200, 200]
    maxY = [10,100,100,100,200,200,200, 200]
    alpha = 4
    amount = [20,20,50,100,20,50,100,200, 300]
    maxX_uniq = [20,50,50,100]
    amount_uniq = [100,100,200,200]
    for x,y,a in zip(maxX_uniq, maxX_uniq, amount_uniq):
        save_results(2, alpha, a, filename='NONE', maxX=x, maxY=y)
    #timeAdjmatrix, timeBip, timeMst, timeGreedy, timeCA, costBip, costMst, costGreedy, costCA = run_algorithms(alpha, 50, maxX=25, maxY=25)
    #draw_rays(6, isRay=True, hasUnitCircle=False)
    #draw_rays(6, isRay=True, hasUnitCircle=False, f=math.pi/4)
    #amount = 200
    #coords = sample_coords(amount, amount, amount)
    #adjmatrix = calculate_distances(coords, alpha)
    #(cost,ranges,edges) = init_bip(adjmatrix)
    #coords = sample_coords(6,6,6)
    #draw_graph(coords)
    #with open('coords.csv', 'w') as file:
        #writer = csv.writer(file, delimiter = ';')
        #writer.writerow(coords)
    #sets = calculate_sets(calculate_distances(coords, alpha=2), isRounded=True)
    #with open('sets.csv', 'w') as file:
        #writer = csv.writer(file, delimiter = ';')
        #writer.writerows(sets)




