import pickle

word_pairs = []
word_pairs.append(['dubai','india'])
word_pairs.append(['namo','modi'])
word_pairs.append(['dubai','speech'])
word_pairs.append(['cricket','stadium'])
word_pairs.append(['nation','country'])
word_pairs.append(['india','pakistan'])
word_pairs.append(['mosque','temple'])
word_pairs.append(['prince','leader'])
word_pairs.append(['crowd','people'])
word_pairs.append(['minister','cricket'])
word_pairs.append(['congress','love'])
word_pairs.append(['terrorism','love'])
word_pairs.append(['nation','land'])

def count(tups,pair):
	temp = []
	c1=0
	c2=0
	for tup in tups:
		if(tup[1] == pair[0]):
			temp.append(tup)
			c1 +=1
	for v in temp:
		v[1] = pair[1]
		if v in tups:
			c2+=1
	d1 = abs(c2-c1)
	for tup in tups:
		if(tup[1] == pair[1]):
			temp.append(tup)
			c1 +=1
	for v in temp:
		v[1] = pair[0]
		if v in tups:
			c2+=1
	d2 = abs(c2-c1)
	return d1+d2
	
def new_check():
	tups = pickle.load(open("tups.pkl","rb"))
	vocab = pickle.load(open("vocab.pkl","rb"))
	dd = {}
	for pair in word_pairs:
		if not(dd.has_key(pair[0])):
			dd[pair[0]]=0
		if not(dd.has_key(pair[1])):
			dd[pair[1]]=0
	for tup in tups:
		if tup[1] in dd.keys():
			dd[tup[1]]+=1
	for pair in word_pairs:
		d = []
		D = count(tups,pair)
		Z = dd[pair[0]] + dd[pair[1]]
		score = 1 - (float(D)/Z)
		pair.append(score)
		print pair
	
if __name__=="__main__":
	new_check()
