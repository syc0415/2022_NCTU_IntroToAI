import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    '''
    Same concept as explained in BFS.
    '''
    file = open(edgeFile)
    reader = csv.DictReader(file)
    edges = [] #list of dicts [{}, {}...]
    for r in reader:
        edges.append(r)
    # start	end	distance speedLimit
    
    adj = {} #dict of list{[], []...}
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
    Well, DFS and BFS are only slightly different. The main difference is that 
    BFS use queue and DFS use stack to determine which vertice is going to
    process. So, silimarly, I used a list called 'visited', a dictionary called
    'parents' to help me save the information I need. However, as I have 
    mentioned above, I used stack to implement DFS instead of queue, which is 
    used in BFS. During the implementation, if we confronted a vertice which
    hasn't been visited, I append it onto the stack. And the important thing 
    is, this time, I append the list 'visited' out of the for loop instead of 
    marking them in the for loop, which means that I would mark vertices as
    visited only after all of their reachable vertices are prcessed. On the 
    contrary, BFS algorithm marked vertices visited 'once' they are processed, 
    regardless of its neighoring vertices. Now back to DFS, after finding the 
    ending point, clear up the stack and finish the DFS searching.
    '''
    stack = []
    stack.append(start)
    visited = []
    visited.append(start)
    parents = {}
    # parents[start] = -1
    while (len(stack)):
        s = stack[-1]
        stack.pop()
        visited.append(s)
        if s not in visited:
            visited.append(s)
        if s not in adj:
            continue        
        for node in range(len(adj[s])):
            cur = adj[s][node]
            if cur not in visited:
                stack.append(cur)
                parents[cur] = s                
                if cur == end:
                    stack.clear()
                    parents[end] = s
                    break
    
    '''
    Same concept as explained in BFS.
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
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
