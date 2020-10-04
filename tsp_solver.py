import sys, getopt, math, random
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

# population size and number of iterations
#change later
p_size = dim
iterations = 3 


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

# dfs to obtain TSP tour


def dfs_non_rec():
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

	return tour


def dfs_rec():
	global iteration, discovered, tour
	discovered = np.zeros(dim)
	tour = np.zeros(dim)
	iteration = 0
	dfs_rec_(0)
	return tour

def dfs_rec_(vertex):
	global iteration
	discovered[vertex] = 1
	tour[iteration] = vertex
	iteration = iteration + 1

	for child in nodes[vertex].children:
		if discovered[child] == 0:
			dfs_rec_(child)


def calc_distance(tour):
	distance = 0
	for i in range(dim - 1):
		distance = distance + graph[int(tour[i])][int(tour[i + 1])]

	distance = distance + graph[int(tour[dim - 1])][int(tour[0])]

	return distance

def crossover_and_mutation(parent1, parent2):
	start = random.randint(0, dim - 1)
	end = random.randint(start + 1, dim)

	child = []

	sublist = parent1[start:end]

	child_index = 0
	parent2_index = 0

	while child_index < start:
		vertex = parent2[parent2_index] 
		
		if vertex not in sublist:
			child.append(vertex)
			child_index = child_index + 1

		parent2_index = parent2_index + 1

	print(child)
	print(sublist)
	child = child + sublist
	child_index = child_index + end - start 

	while child_index < dim:
		vertex = parent2[parent2_index]

		if vertex not in sublist:
			child.append(vertex)
			child_index = child_index + 1

		parent2_index = parent2_index + 1

	#mutation, which is just swapping 2 vertices
	a = random.randint(0, dim - 1)
	b = random.randint(0, dim - 1)
	child[a], child[b] = child[b], child[a]

	return child

def sort_population(population, distances, size):
	sorted_indices = np.argsort(distances)
	sorted_distances = np.sort(distances)

	sorted_population = []

	for i in range(size):
		sorted_population.append(population[sorted_indices[i]])

	if size < len(sorted_distances):
		sorted_distances = sorted_distances[:size]

	return sorted_population, sorted_distances



population = []


branching_nodes = []

for i in range(dim):
	if len(nodes[i].children) > 1:
		branching_nodes.append(i)

number_of_branching_nodes = len(branching_nodes)

#it's simpler to work with even population size
if p_size % 2 == 1:
	p_size = p_size - 1

half_p_size = p_size // 2
quart_p_size = half_p_size // 2 # number of solutions replaces each iteration
parents_remaining = p_size - quart_p_size # number of parents remaining after each iteration
	

if number_of_branching_nodes == 0 or p_size == 0 or iterations == 0:
	tour = dfs_rec()

	f = open("solution.csv", "w")

	for i in range(dim):
		f.write(str(int(tour[i] + 1)) + "\n")

	f.close()

	print(calc_distance(tour))

	exit()


distances = np.zeros(p_size)

# generate population
for i in range(half_p_size):
	rec_tour = dfs_rec()
	non_rec_tour = dfs_non_rec()
	population.append(rec_tour)
	population.append(non_rec_tour)
	distances[2 * i] = calc_distance(rec_tour)
	distances[2 * i + 1] = calc_distance(non_rec_tour)

	# shuffle children of a node to make next dfs walks different
	shuffled_node = i % number_of_branching_nodes
	prev_state = nodes[shuffled_node].children.copy()

	random.shuffle(nodes[shuffled_node].children) 

	while prev_state == nodes[shuffled_node]:
		random.shuffle(nodes[shuffled_node].children)

	
population, distances = sort_population(population, distances, p_size)


for i in range(iterations - 1):
	children = []
	children_distances = np.zeros(half_p_size)

	#generate children
	for j in range(half_p_size):
		child = (crossover_and_mutation(population[2 * j], population[2 * j + 1]))
		children.append(child)
		children_distances[j] = calc_distance(child)

	children, children_distances = sort_population(children, children_distances, children_replaces)

	population = population[:parents_remaining] + children
	distances = distances[:parents_remaining] + children_distances

	population, distances = sort_population(population, distances, p_size)




print(distances[0])


	



# for i in range(p_size - 1):
# 	if sorted_distances[i] > sorted_distances[i + 1]:
# 		print(i)




# for i in range(p_size):

	








# print(calc_distance(tour))

# f = open("solution.csv", "w")

# for i in range(dim):
# 	f.write(str(int(tour[i] + 1)) + "\n")


# f.close()

# best tour


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