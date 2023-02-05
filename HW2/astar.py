import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):
    # Begin your code (Part 4)
    '''
    In order to implement a* algorithm, we need to read the heuristicFile, 
    which contains each vertice's heuristic distance.
    '''
    hfile = open(heuristicFile)
    hreader = csv.DictReader(hfile)
    h = []
    for r in hreader:
        h.append(r)
    # node 1079387396 1737223506 8513026827

    file = open(edgeFile)
    reader = csv.DictReader(file)
    edges = []
    for r in reader:
        edges.append(r)
    # start	end	distance speedLimit
    
    adj = {}
    for i in range(len(edges)):
        cur = edges[i]['start']
        cur = int(cur)
        dest = edges[i]['end']
        dest = int(dest)
        dist = edges[i]['distance']
        dist = float(dist)
        if cur not in adj:
            adj[cur] = []
            adj[cur].append([dest, dist])
        else:
            adj[cur].append([dest, dist])
    '''
    This for loop is to find the heuristic distance of the starting point
    '''
    # find the heuristic distance
    for i in range(len(h)):
        if h[i]['node'] == str(start):
            hDist = float(h[i][str(end)])
    '''
    In a* algorithm, we not only need to concern each edge's distance, but also
    the heuristic distance of the coming node. So I put [vertice, 
    distance, heuristic distance] into the priority queue. In the for loop at 
    64-89, we need to access the current comparing vertice's heuristic 
    distance, and sum up the accumulated distance so far, the edge's distance 
    and the heuristic distance of current vertice, then put these information
    back in the queue. On the other hand, if a vertice is already in the queue,
    do the relaxation steps, which can help us find the shortest path to the
    ending point. Keep doing these steps until we reach to the ending point.
    '''
    pq = []  # priority queue
    pq.append([start, 0, hDist])  # [vertice, dist, total dist]
    visited = []
    visited.append(start)
    compared = []
    compared.append(start)
    parents = {}
    while (len(pq)):
        pq = sorted(pq, key=lambda pq: pq[2])  # sort by total dist
        top = pq.pop(0)
        s = top[0]
        dist = top[1]
        visited.append(s)
        for i in range(len(adj[s])):
            cur = adj[s][i][0]
            curDist = adj[s][i][1]
            for j in range(len(h)):
                if h[j]['node'] == str(cur):
                    hDist = float(h[j][str(end)])
            if cur not in adj:
                continue
            elif cur not in compared:
                totDist = dist + curDist + hDist
                pq.append([cur, dist + curDist, totDist])
                compared.append(cur)
                parents[cur] = s
                if cur == end:
                    visited.append(end)
                    pq.clear()
                    parents[end] = s            
                    break
            elif cur in compared:
                # relaxation
                for k in range(len(pq)):
                    if pq[k][0] == cur and pq[k][1] > dist + curDist:
                        pq[k][1] = dist + curDist
                        pq[k][2] = pq[k][1] + hDist
                        parents[cur] = s
    '''
    As usual, the same
    '''
    path = [end]
    dist = 0
    num_visited = len(visited) - 1 # start doesn't count
    parent = end
    while(parent != start):
        cur = parent
        parent = parents[cur]
        path.append(parent)
        for e in edges:
            s = str(parent)
            dest = str(cur)
            if e['start'] == s and e['end'] == dest:
                dist += float(e['distance'])
    path.reverse()

    return(path, dist, num_visited)
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
