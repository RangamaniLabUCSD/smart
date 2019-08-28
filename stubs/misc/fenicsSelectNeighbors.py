from dolfin import *
vmesh = Mesh('/home/justin/gitrepos/fenics-systems-biology-framework/gamertetmesh.xml')
#vmesh = UnitCubeMesh(5,5,5)
m = BoundaryMesh(vmesh,'exterior')
print('mesh has %d number of vertices' % m.num_vertices())
vm = list(vertices(m))


initVertex = 0

oldNeighbors = {initVertex}
allVertices = {initVertex}
loopidx = 0
while True:
    loopidx += 1

    newNeighbors = set() # init as empty
    for neighbor in oldNeighbors: 
        neighborVertex = list(vertices(m))[neighbor]
        for edge in edges(neighborVertex):
            newVertexIndices = {v.index() for v in vertices(edge)}

            newNeighbors = newNeighbors.union(newVertexIndices)

    print('%d vertices added to set' % len(oldNeighbors))
    oldNeighbors = newNeighbors.difference(allVertices)
    allVertices = allVertices.union(oldNeighbors)

    print('found the %d-ring of neighbors' % loopidx)
    if len(oldNeighbors) == 0: 
        print('breaking loop. no more new neighbors')
        break

    if loopidx >= 100:
        print('probably coded something wrong or this is a giant mesh')
        break

    print('Found %d vertices so far...' % len(allVertices))

print(len(allVertices))
