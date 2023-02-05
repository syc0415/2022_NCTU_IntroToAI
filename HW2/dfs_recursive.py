import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    f = open(edgeFile)
    reader = csv.DictReader(f)
    edges = [] #list of dicts [{}, {}...]
    for r in reader:
        edges.append(r)
    
    nodes = {} #dict of dicts of list{{[], []...}, {}...}
    for i in range(len(edges)):
        cur = edges[i]['start']
        cur = int(cur)
        dest = edges[i]['end']
        dest = int(dest)
        if cur not in nodes:
            nodes[cur] = {}
            nodes[cur]['neighbors'] = [dest] #list
        else:
            nodes[cur]['neighbors'].append(dest)

    stack = []
    queue.append(start)
    visited = []
    visited.append(start)
    nodes[start]['parent'] = -1
    
    while (queue):
        node = queue.pop(0)
        for i in range(len(nodes[node]['neighbors'])):
            cur = nodes[node]['neighbors'][i]
            if cur not in nodes:
                visited.append(cur)
                continue
            elif cur not in visited:
                queue.append(cur)
                visited.append(cur)
                nodes[cur]['parent'] = node
                if cur == end:
                    queue.clear()
                    nodes[end]['parent'] = node
                    break
            
    path = [end]
    dist = 0
    num_visited = len(visited) - 1 # start doesn't count
    parent = end
    while(parent != start):
        cur = parent
        parent = nodes[cur]['parent']
        path.append(parent)        
        for e in edges:
            s = str(parent)
            dest = str(cur)
            if e['start'] == s and e['end'] == dest:
                dist += float(e['distance'])
    path.reverse()
    
    return(path, dist, num_visited) 
    # raise NotImplementedError("To be implemented")
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
