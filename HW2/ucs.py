import csv
edgeFile = 'edges.csv'


def ucs(start, end):
    # Begin your code (Part 3)
    file = open(edgeFile)
    reader = csv.DictReader(file)
    edges = []
    for r in reader:
        edges.append(r)
    # start	end	distance speedLimit
    '''
    This time, append another element which represents the distance of the 
    adjacency vertices. Other part just stay the same.
    '''
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
    In UCS, I used the concept of priority queue to implement the alogprithm.
    In every while loop, I choose the vertice with the smallest distance to 
    process, and find its neighboring vertices. If that vertice hasn't been 
    visited before, sum up their distance and put the [vertice, distance] into
    pq. Else, I compared new calculated total distance with existing total
    distance, if the current one is smaller than the existing one, update its
    value. This step is also called relaxation. Again, once we reach to the 
    ending point, stop the algorithm and do the next part.
    '''
    pq = [] #priority queue
    pq.append([start, 0]) # [vertice, dist]
    visited = []
    visited.append(start)
    parents = {}
    while (len(pq)):
        pq = sorted(pq, key = lambda pq : pq[1]) # sort by dist
        top = pq.pop(0)
        s = top[0]
        dist = top[1]
        for i in range(len(adj[s])):
            cur = adj[s][i][0]
            curDist = adj[s][i][1]
            if cur not in adj:
                continue
            elif cur not in visited:
                pq.append([cur, dist + curDist])
                visited.append(cur)
                parents[cur] = s
                if cur == end:
                    pq.clear()
                    parents[end] = s
                    break
            elif cur in visited:
                # relaxation
                for j in range(len(pq)):
                    if pq[j][0] == cur and pq[j][1] > dist + curDist:
                        pq[j][1] = dist + curDist
                        parents[cur] = s
    '''
    Well, stay the same again.
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
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
