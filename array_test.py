import numpy as np
import numpy as np



y = np.array([1, 4, 2, 3])

z = np.argsort(y, kind="mergesort")

x = np.array([10, 40, 20, 30])

x = x[z]
y = y[z]

print(x)

print(y)














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


