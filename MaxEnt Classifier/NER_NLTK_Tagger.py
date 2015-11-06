import nltk 
import os
import re
import sys
reload(sys)  
sys.setdefaultencoding('utf8')
X=[]

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


for root, dirs, files in os.walk("NDTV_mobile_reviews_Classified"):
	for name in files:
		p = str(os.path.join(root, name))
		s2=""
		with open(p,'r') as f:
			sample = f.read();

		s1 = sample.decode("utf-8")
		sentences = nltk.sent_tokenize(s1)
		tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
		tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
		chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=False)

		result=[]
		def extract_entity_names(t):
			entity_names = []
			if hasattr(t, 'label') and t.label:
				if(t.label() == 'PERSON'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				elif(t.label() == 'LOCATION'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				elif(t.label() == 'ORGANIZATION'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				elif(t.label() == 'DATE'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				elif(t.label() == 'TIME'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				elif(t.label() == 'MONEY'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				elif(t.label() == 'PERCENT'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				elif(t.label() == 'FACILITY'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				elif(t.label() == 'GPE'):
				    ner = (' '.join([child[0] for child in t]))
				    lab = t.label()
				    result.append([ner,lab])
				else:
				    for child in t:
				        entity_names.extend(extract_entity_names(child))
			return result

		entity_names = []
		for tree in chunked_sentences:
			# Print results per sentence
			# print extract_entity_names(tree)
			entity_names.extend(extract_entity_names(tree))

		res_tag = []
		c=0
		for j in result:
			l1 = j[0].split(' ')
			t = j[1]
			j=[]
			for k in l1:
				l2=[k,t]
				j.append(l2)
			result[c]=j
			c+=1

		result1=[]
		for j in result:
			result1.extend(j)

		for j in result1:
			res_tag.append(j[0])

		ind=0
		l3=[]
		for i in tokenized_sentences:
			final_res=[]
			for j in i:
				if(j in res_tag):
					l3 = result1[ind]
					if(ind<len(res_tag)-1):
						ind+=1
				else:
					l3=[i,'OTHER']
				final_res.append(l3)
			his = create_hist(i,final_res)
			X.append(his)

			fp = open("history.txt","w")
			fp.write(str(X))
			fp.close()
