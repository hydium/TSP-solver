f = open("solution.csv", "r")
t = open("test.csv", "w+")

dim = 11849

# for line in t:
# 	print(line)


for line in f:
	city = int(float(line.replace("\n", "")))

	t.write(str(city) + " \n")
	# print(city)


	if city < 1 or city > dim:
		print(city)

t.close()