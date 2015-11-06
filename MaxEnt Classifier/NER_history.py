def create_hist(w,tagged):
	'''
		Create history tuples given a sentence and its NLTK-tag sequence
	'''
	h = []
	for i in range(len(w)):
		if i==0:
			tup = ('*','*',w,i)
		elif i==1:
			tup = ('*',tagged[0][1],w,i)
		else:
			tup = (tagged[i-2][1], tagged[i-1][1],w,i)
		h.append(tup)
	return h
