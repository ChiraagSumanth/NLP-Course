import os, re
import pickle
from nltk import sent_tokenize
from nltk.parse import stanford
from nltk.tree import ParentedTree
import nltk
from bfs_nltk import breadth_first


os.environ['STANFORD_PARSER'] = r"/home/pydev/stanford-corenlp-python/stanford-corenlp-full-2014-08-27"
os.environ['STANFORD_MODELS'] = r"/home/pydev/stanford-corenlp-python/stanford-corenlp-full-2014-08-27"
os.environ['JAVAHOME'] = "/usr/lib/jvm/java-8-oracle"

parser = stanford.StanfordParser()
pick = True
trees = []

if pick:
	# Using the Stanford Parser to get NLTK Parse Trees
	dataset = open("dataset.txt","r").read()
	sentences = parser.raw_parse_sents(sent_tokenize(dataset))
	for line in sentences:
		for sentence in line:
			trees.append(sentence)
	
	pickle.dump(trees, open("trees.pkl","wb"))
		
else:
	trees = pickle.load(open("trees.pkl","rb"))
	# Converting NLTK trees into parented trees for easy upward traversal
global ptrees
ptrees = []
for tree in trees:
	ptrees.append(ParentedTree.convert(tree))

# Function that performs Hobbs Algo
def hobbs(ptree, cur_sent):
	'''
		Hobbs Naive Algorithm for Anaphora Resolution
	'''
	#ptree.draw()
	resolutions = {}
	highest_S = None
	for root in ptree.subtrees():
		if root.label() == 'S':
			highest_S = root
			break
			
	pronouns = []	
	c = 0
	# Find PRP or PRP$ and its immediately dominating NP
	for root in ptree.subtrees():
		if root.label() == 'PRP' or root.label() == 'PRP$':
			pronouns.append(root)
	#print pronouns
	if len(pronouns)==0:
		return None, None
	# Resolve each pronoun
	while(True):
	
		pronoun =  pronouns[c]
		done = False
		resolutions[c] = []
		n =  pronoun.parent()	# begin at immediate NP
		#print n
		path = []
		path.append(n)
		#if c==1: print n
	
		# Must recurse here!
		while(True):
			while(True):
				try:
					n = n.parent()
					path.append(n)
					if n.label() == 'NP' or n.label() == 'S':
						break
				except:
					break
			#if c==1: print path
			X = n
			if not X:
				c+=1
				break
			#if c==1: print X
	
			# Step 2b
			bfs = breadth_first(X)
			#if c==1:print bfs
			for level in bfs[1:]:
				for stree in level:
					#print stree
					if stree == n or stree in path[:]:
						#print "Not"
						break
					#propose NP which has NP or S in between itself and X
					if(stree.label()=='NP'):
						k = stree
						#if c==1: print stree
						#print path
						f1 = False
						f2 = False
						while(True):
							try:
								k = k.parent()
								if k in path:
									# There is some ancestor connected to the NP immediately dominating PRP
									f1 = True
								if(k.label() == 'NP' or k.label() == 'S'):
									#There is an NP in between stree and X, so propose stree as a resolution
									f2 = True
								if (f1 and f2):
									# NP or S in between dominating NP for the PRP and current NP 
									resolutions[c].append(' '.join(stree.leaves()))
									#if c==1: print resolutions[c]
									break
							except Exception:
								break
			
			# Step 3			
			if X == highest_S: #and len(resolutions[c])==0:
				# Look in previous trees
				for t in range(cur_sent-1,0,-1):
					bfs = breadth_first(ptrees[t])
					for level in bfs:
						for stree in level:
							if(stree.label()=='NP'):
								resolutions[c].append(' '.join(stree.leaves()))
			else:
	
				# Step 4
				while(True):
					try:
						n = n.parent()
						path.append(n)
						if n.label() == 'NP' or n.label() == 'S':
							break
					except:
						break
				X = n
				if not X:
					c+=1
					break
				
				# Step 5 NN, NNP, CD are all N-Bars
				if X.label()=='NP':
					#print X
					bfs = breadth_first(X)[0]
					for nod in bfs:
						if nod.label() in ['NN','NNP','CD'] and nod not in path:
							resolutions[c].append(' '.join(X.leaves()))
	
				# Step 6
				bfs = breadth_first(X)
				for level in bfs:
					for stree in level:
						if stree == n or stree in path[:]:
							#print "Not"
							break
						elif(stree.label()=='NP'):
							resolutions[c].append(' '.join(stree.leaves()))
				
				# Step 7
				if X.label()=='S':
					bfs = breadth_first(X)
					for level in bfs[1:]:
						rlevel = []
						rflag = False
						for stuff in level:
							if stuff in path:
								rflag = True
								continue
							if rflag:
								rlevel.append(stuff) 
						for stree in rlevel:
							#propose NP which has NP or S in between itself and X
							if(stree.label()=='NP'):
								k = stree
								f1 = False
								f2 = False
								while(True):
									try:
										k = k.parent()
										if k in path:
											# There is some ancestor connected to the NP immediately dominating PRP
											f1 = True
										if(k.label() == 'NP' or k.label() == 'S'):
											#There is an NP in between stree and X, so propose stree as a resolution
											f2 = True
										if ((not f1) and f2):
											# NP or S in between dominating NP for the PRP and current NP
											s1 = ' '.join(stree.leaves())
											resolutions[c].append("R: " + s1)
											break
									except Exception:
										break
			
			if resolutions[c]:
				c+=1
				break	
		if len(pronouns)-1 < c:
			break
	return resolutions, pronouns	

def resolve_sel(r,p,sen):
	# Selectional Constraints
	for key in r.keys():
		prn = p[key].leaves()[0]
		#print prn
		# remove duplicates
		r[key] = list(set(r[key]))
		
		# if resolution contains the PRP itself, remove the resolution
		for i in range(len(r[key])):
			words = r[key][i].split()
			if prn in words:
				r[key][i] = "<REM>"
		
		# if subsets appear, remove supersets
		for i in range(len(r[key])-1):
			for j in range(i+1,len(r[key])):
				if r[key][i] in r[key][j]:
					r[key][j] = "<REM>"
				elif r[key][j] in r[key][i]:
					r[key][i] = "<REM>"
				
		while "<REM>" in r[key]:
			r[key].remove("<REM>")
	
	#remove everything to the right of the pronoun
	try:
		for key in r.keys():
			prn = p[key].leaves()[0]
			#print prn
			#print "--------------------------------------"
			i1 = sen.index(prn)
			for ind in range(len(r[key])):
				i2 = sen.index(r[key][ind])
				#print i2
				if i2 > i1:
					r[key][ind] = "<REM>"
	
			while "<REM>" in r[key]:
					r[key].remove("<REM>")
	except:
		pass
	return r,p

for i in range(len(ptrees)):
	res, pronouns = hobbs(ptrees[i], i)
	if pronouns:
		if not res:
			print str(pronouns)+" were not resolved"
		else:
			sent = ' '.join(ptrees[i].leaves())
			print 'Sentence:' + sent
			r,pr = resolve_sel(res,pronouns,sent)
			for num in range(len(pr)):
				pass
				print str(pr[num]) + ': '+str(r[num][0])
				print '-'*50

