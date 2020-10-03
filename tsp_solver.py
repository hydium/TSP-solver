import sys, getopt, math
import numpy as np


args = sys.argv[1:]

optlist, args = getopt.getopt(args, 'p:f:')

textfile = args[0]

f = open(textfile, "r")

parsing_nodes = False 

for line in f:
	s_line = line.split()
	if s_line[0] == "EOF":
		break

	if "DIMENSION" in s_line[0]:
		dim = int(s_line[-1])
		coords = np.zeros((2, dim)) #arrays with x, y coordinates of the nodes

	if parsing_nodes:
		index = int(s_line[0]) - 1
		coords[0][index] = float(s_line[1])
		coords[1][index] = float(s_line[2])

	if s_line[0] == "NODE_COORD_SECTION":
		parsing_nodes = True

f.close()

graph = np.zeros((dim, dim))

#calculate edge weights
for i in range(dim):
	graph[i][i] = 0

for i in range(1, dim):
	for j in range(0, i):
		graph[i][j] = math.sqrt((coords[0][i] - coords[0][j])**2 + (coords[1][i] - coords[1][j])**2)
		graph[j][i] = graph[i][j] 


print("graph is done")

#construction of MST
class Node:
	def __init__(self, vertex):
		self.vertex = vertex
		self.parent = None
		self.children = []
		self.priority = 0

nodes = []

root = Node(0)
root.priority = 0
nodes.append(root)

for i in range(1, dim):
	new_node = Node(i)
	new_node.priority = graph[0][i]
	new_node.parent = 0
	root.children.append(i)
	nodes.append(new_node)


for i in range(dim - 2):
	min_priority = math.inf
	min_vertex = None

	for j in range(0, dim):
		priority = nodes[j].priority
		if priority > 0 and priority < min_priority:
			min_priority = priority
			min_vertex = j

	nodes[min_vertex].priority = 0

	for j in range(0, dim):
		node = nodes[j]
		priority = node.priority
		if priority > graph[min_vertex][j]:
			node.priority = graph[min_vertex][j]
			if node.parent is not None:
				nodes[node.parent].children.remove(j)
			node.parent = min_vertex
			nodes[min_vertex].children.append(j)

print("mst is done")

#dfs to obtain TSP tour
discovered = np.zeros(dim)
tour = np.zeros(dim)
S = [] #stack

S.append(0)

index = 0
while len(S) > 0:
	vertex = S.pop()
	if discovered[vertex] == 0:
		discovered[vertex] = 1
		tour[index] = vertex
		index = index + 1
		for child in nodes[vertex].children:
			S.append(child)		


distance = 0
for i in range(dim - 1):
	distance = distance + graph[int(tour[i])][int(tour[i + 1])]

distance = distance + graph[int(tour[dim - 1])][int(tour[0])]

print(distance)

f = open("solution.csv", "w")

for i in range(dim):
	f.write(str(int(tour[i] + 1)) + "\n")


f.close()

#best tour
# f = open("best_tour.txt", "r")

# best_tour = np.zeros(dim)

# index = 0
# for city in f:
# 	best_tour[index] = int(city)
# 	index = index + 1

# distance = 0
# for i in range(dim - 1):
# 	distance = distance + graph[int(best_tour[i]) - 1][int(best_tour[i + 1]) - 1]

# distance = distance + graph[int(best_tour[dim - 1]) - 1][int(best_tour[0]) - 1]

# print(distance)

# f.close()