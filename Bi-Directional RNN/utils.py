import math, heapq
import pickle

def perplexity(pic_file):
	words = pickle.load(open(pic_file,"rb"))
	M = 1 # No sentences in our document! just a culmination of words, so M =1
	p = []
	# Taking P(Si) over sentences itself for the entire corpus
	# MLE will be count(ci-1, ci)/ count(ci-1)
	# for a sentence, multiply all MLEs for characters to get P(Si)
	words = [w.lower() for w in words]
	uni = {}
	bi = {}
	for w in words:
		if uni.has_key(w):
			uni[w] += 1
		else:
			uni[w] = 1
	q = float(uni[words[0]])/len(words)		# Start with unigram probability
	for i in range(len(words[1:])):
		if bi.has_key((words[i],words[i-1])):
			bi[(words[i],words[i-1])] += 1
		else:
			bi[(words[i],words[i-1])] = 1
	for i in range(len(words[1:])):
		q *= float(bi[(words[i], words[i-1])])/float(uni[words[i-1]])
	p.append(q)
	pi = map(lambda x:math.log(x), p)
	s = sum(pi)
	l = s/float(M)
	return 2**(-l)


if __name__=="__main__":
	perp = perplexity("output_5k_1L.pkl")
	print "Perplexity = "+str(perp)
	
