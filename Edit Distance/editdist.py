import numpy as np

def min_edit_dist(s1,s2, cost_ins = 1, cost_del = 1, cost_sub = 2):
	s1 = " "+s1
	s2 = " "+s2
	n = len(s1)
	m = len(s2)
	if n==0 and m==0:
		return 0
	elif n==0:
		return m*cost_ins
	elif m==0:
		return n*cost_del
	mat = [0] * (n*m)
	D = np.reshape(mat,(n,m))
	# initialization cases
	for i in range(0,n):
		D[i][0] = i
	for j in range(0,m):
		D[0][j] = j
	
	# recursion
	for i in range(1,n):
		for j in range(1,m):
			if s1[i]!=s2[j]:
				act_sub_cost = cost_sub
			else:
				act_sub_cost = 0
			D[i][j] = min(D[i-1][j]+cost_del, D[i][j-1]+cost_ins, D[i-1][j-1]+act_sub_cost)
	
	# print D
	# termination
	return D[n-1][m-1] 
	
def part2(s,words):
	min_dists = []
	for i in range(len(words)):
		min_dists.append(min_edit_dist(s,words[i]))
	ind = min_dists.index(min(min_dists))
	return words[ind],min(min_dists)


if __name__=="__main__":
	while True:
		mode = raw_input("Enter 1 for Part1 or 2 for Part2 or anything else to exit: ")
		if mode=="1":
			s1 = raw_input("Enter string 1: ")
			s2 = raw_input("Enter string 2: ")
			mdist = min_edit_dist(s1,s2)
			print mdist
		elif mode=="2":
			s_inp = "graffe"
			words = ["graf","graB","grail","giraffe"]
			ans = part2(s_inp,words)
			print ans
		else:
			break
