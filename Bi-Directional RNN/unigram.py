import nltk
from nltk.tokenize import RegexpTokenizer
import collections
import pickle

def get_unigram1000(filename):
	data = None
	with open(filename,'r') as f:
		data = f.read()
	re_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
	words = re_tokenizer.tokenize(data)
	unigram = {}
	for item in words:
		if item in unigram.keys():
			unigram[item]+=1
		else:
			unigram[item] = 1
	o_unigram = collections.OrderedDict(unigram)
	words = []
	# print o_unigram[:100]
	count = 0
	for key in o_unigram.keys():
		if count == 5000:
			break
		words.append(key)
		count +=1
	return words
	

if __name__ == '__main__':
	unigram1000 =  get_unigram1000('people.txt')
	print unigram1000
	pickle.dump(unigram1000,open("words5000.pkl","wb"))

	
