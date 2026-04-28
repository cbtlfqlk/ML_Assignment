Points = [[4,4],[0,2],[4,3],[0,0],[5,4],[1,1],[2,0],[5,5]]


"""
means = [Points[0],Points[5]]

def iter(means,Points):
    meansP = [[] for i in means]
    def dist(a,b):
        c = 0
        for i,j in zip(a,b): c+=abs(i-j)
        return c
    #print(means)
    for point in Points:
        meansP[min(range(len(meansP)),key=lambda x:dist(means[x],point))].append(point)
    for i,j in enumerate(meansP):
        means[i] = (sum(map(lambda x:x[0],j))/len(j),sum(map(lambda x:x[1],j))/len(j))

print(means)
iter(means,Points)
print(means)
iter(means,Points)
print(means)

"""

def dbscan(points,MinPts=2,eps=2):
    def dist(a,b): return sum(abs(i-j) for i,j in zip(a,b))
    clusts = [-1 for i in points]
    visited = [0 for i in points]
    clust = 0
    def scan(ind:int, cluster = None):
        nonlocal dist, points,clust, clusts,eps,MinPts
        if visited[ind]: return
        visited[ind] = 1;
        point = points[ind]
        close = [i for i,j in enumerate(points) if dist(j,point)<=eps]
        clust = cluster or (clust:=clust+1)
        if len(close)>=MinPts:
            for i in close: clusts[i] = clust;i==ind or scan(i,clust)
    for i in range(len(points)): scan(i)
    return clusts

print(dbscan(Points,2,2))
            

        
            
