import csv
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    '''
    First, use csv to read egeFile. I used a list of dictionaries to save the
    information of every edges In the next for loop, I used a dictionary of
    list to create a adjacency list for every vertice. The key of the 
    dictionary is the starting vertice, and the value is a list that stores its
    neighboring vertices.
     '''
    file = open(edgeFile)
    reader = csv.DictReader(file)
    edges = [] #list of dicts [{}, {}...]
    for r in reader:
        edges.append(r)
    # start	end	distance speedLimit

    adj = {} #dict of lists{[], []...}
    for i in range(len(edges)):
        cur = edges[i]['start']
        cur = int(cur)
        dest = edges[i]['end']
        dest = int(dest)
        if cur not in adj:
            adj[cur] = []
            adj[cur].append(dest)
        else:
            adj[cur].append(dest)
    
    '''
    Second, implement BFS algorithm to find the route, which uses a queue to
    determine which vertice is going to process. In every loop, we need to 
    process all of the neighboring vertices. Also, I use a list called 
    'visited' to see if a vertice is visited or not. If it hasn't been visited,
    append the current vertice into to queue and mark it as visited. On the
    other hand, a dictionary called 'parents' is also very important, because
    it tells us which vertice is the current vertice's predecessor. Finally,
    when the algorithm reach the ending point, I clear up the queue and finish 
    the BFS searching.
    '''
    queue = []
    queue.append(start)
    visited = []
    visited.append(start)
    parents = {}
    # parents[start] = -1
    while (len(queue)):
        s = queue.pop(0)
        for i in range(len(adj[s])):
            cur = adj[s][i]
            if cur not in adj:
                visited.append(cur)
                continue
            elif cur not in visited:
                queue.append(cur)
                visited.append(cur)
                parents[cur] = s
                if cur == end:
                    queue.clear()
                    parents[end] = s
                    break
                
    '''
    Last, we need to calculate path, distance and the number of visited 
    vertices. Since I used the length of the list 'visited' to calculate the 
    visited vertices, I need to minus its length by one since the list contains
    the starting point, which should not be counted in the answer. Next, I used
    the dictionary 'parents' to find the path by going to the vertice's
    predecessor one by one, and append them into the list called 'path'.
    At the same time, I sum up their distance to get the total distance.
    After traversing from the ending point to the starting point, I reverse the 
    list 'path' and return the assigned data: path, dist, num_visited.
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
    # End your code (Part 1)

if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
