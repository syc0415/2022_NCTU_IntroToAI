import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar_time(start, end):
    # Begin your code (Part 6)
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
    '''
    Above is all the same as astar.
    However, in the folowing part, since we need take 'time' as the comparing
    factor instead of the distance, this time I save the speed limit of each
    edge into adjacency list.
    '''
    adj = {}
    for i in range(len(edges)):
        cur = edges[i]['start']
        cur = int(cur)
        dest = edges[i]['end']
        dest = int(dest)
        dist = edges[i]['distance']
        dist = float(dist)
        speed = edges[i]['speed limit']
        speed = float(speed)
        time = dist / speed
        if cur not in adj:
            adj[cur] = []
            adj[cur].append([dest, time, speed])
        else:
            adj[cur].append([dest, time, speed])
    
    # find the heuristic distance
    for i in range(len(h)):
        if h[i]['node'] == str(start):
            hDist = float(h[i][str(end)])
    '''
    The main idea is still the same as astar while the factor we concerned 
    change from distance to time. In this algorithm, every time we access a 
    vertice, we need to also access its speed limit to calculate the time if we
    take on this edge would cost.
    
    '''
    pq = []  # priority queue
    hTime = hDist / (60 * 1000 / 3600) # set a speed
    pq.append([start, 0, hTime])  # [vertice, time, total time]
    visited = []
    visited.append(start)
    compared = []
    compared.append(start)
    parents = {}
    while (len(pq)):
        pq = sorted(pq, key=lambda pq: pq[2])  # sort by total time
        top = pq.pop(0)
        s = top[0]
        time = top[1]
        visited.append(s)
        for i in range(len(adj[s])):
            cur = adj[s][i][0]
            curTime = adj[s][i][1]
            curSpeed = adj[s][i][2]
            for j in range(len(h)):
                if h[j]['node'] == str(cur):
                    hDist = float(h[j][str(end)])
            hTime = hDist / curSpeed
            if cur not in adj:
                continue
            elif cur not in compared:
                totTime = time + curTime + hTime
                pq.append([cur, time + curTime, totTime])
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
                    if pq[k][0] == cur and pq[k][1] > time + curTime:
                        pq[k][1] = time + curTime
                        pq[k][2] = pq[k][1] + hTime
                        parents[cur] = s
    '''
    Only change the total distance into total time. Remember to do the unit
    conversion.
    '''
    path = [end]
    time = 0
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
                speed = float(e['speed limit']) * 1000 / 3600
                time += float(e['distance']) / speed
    path.reverse()

    return(path, time, num_visited)
    # raise NotImplementedError("To be implemented")
    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
