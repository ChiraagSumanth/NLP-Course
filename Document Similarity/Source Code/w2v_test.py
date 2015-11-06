import nltk
from sentence2vec.word2vec import Word2Vec 

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

def preprocess(filename):
	f = open(filename,"r")
	text = f.read()
	f.close()
	s = nltk.sent_tokenize(text.strip())
	trng = []
	for sent in s:
		trng.append(nltk.word_tokenize(sent.lower()))
		#trng.append(nltk.word_tokenize(sent.lower()))
	return trng

def train_w2v(filename='cleaned_tweets.txt', train=False):
	if train:
		trng = preprocess(filename)
		model = Word2Vec(trng, size=100, window=3, sg=1, min_count=3, workers=8)
	else:
		model = Word2Vec.load(filename+'.model')
	for pair in word_pairs:
		print "%s and %s similarity: %s"%(pair[0],pair[1],model.similarity(pair[0],pair[1]))

if __name__=="__main__":
	train_w2v(train=True)
