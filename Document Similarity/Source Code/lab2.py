import json
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer
import re

sep=re.compile(r"[ \t,;.?!]+")
hasht = re.compile(r"#[a-zA-Z0-9_]+")
atp = re.compile(r"@[a-zA-Z0-9_]+")

urlStart1  = "(?:https?://|\\bwww\\.)"
commonTLDs = "(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)";
ccTLDs     = '''(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|
    bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|
    er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|
    hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|
    lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|
    nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|
    sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|
    va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)'''  
urlStart2  = "\\b(?:[A-Za-z\\d-])+(?:\\.[A-Za-z0-9]+){0,3}\\." + "(?:"+commonTLDs+"|"+ccTLDs+")"+"(?:\\."+ccTLDs+")?(?=\\W|$)"
urlBody    = "(?:[^\\.\\s<>][^\\s<>]*?)?"
punctChars = r"['\".?!,:;]"
entity     = "&(?:amp|lt|gt|quot);"
urlExtraBeforeEnd = "(?:"+punctChars+"|"+entity+")+?"
urlEnd     = "(?:\\.\\.+|[<>]|\\s|$)"
url        = re.compile(r"(?:"+urlStart1+"|"+urlStart2+")"+urlBody+"(?=(?:"+urlExtraBeforeEnd+")?"+urlEnd+")")

def generate_tweet_corpus(filename):
	tweets = None
	with open(filename,"r") as f:
		tweets = f.readlines()
	corpus=[]
	for i in tweets:
		if(len(i)>1):
			corpus.append(i)
	return corpus

def filter_text(corpus):
	unknown = [hasht,atp,url]
	filtered_corpus = []
	for tweet in corpus:
		sentence = []
		try:
			tweet = tweet.decode("utf-8")
			tweet = tweet.encode("ascii")
			# tweet = tweet.split(" ")
			tweet = sep.split(tweet)
			for word in tweet:
				w = word
				if hasht.search(w) or atp.search(w) or url.search(w):
					w="*"
					break
				if "\n" in word:
					w = word.replace("\n", "")
				sentence.append(w)
			sentence = " ".join(sentence)
			filtered_corpus.append(sentence)
		except Exception:
			pass
	return filtered_corpus

def derive_vocabulary(corpus,stemming=True):
	vocab = []
	re_tokenizer = RegexpTokenizer('\w+')
	st = LancasterStemmer()
	for tweet in corpus:
		tweet = tweet.split(" ")
		for word in tweet:
			if stemming:
				vocab.append(st.stem(word.lower()))
			else:
				vocab.append(word.lower())
	return vocab


def get_unigram(words):
   	unigram = {}
   	unigram_list=[]
   	for item in words:
   		if item not in unigram:
   			unigram[item] = 1
   		else:
   			unigram[item] += 1
   	# for key in unigram:
   		# unigram[key] = float(unigram[key])/len(words)
   	# with open('unigram.pkl', 'wb') as output:
		# pickle.dump(unigram, output, pickle.HIGHEST_PROTOCOL)
	# print "Vocabulary size : {}".format(len(unigram))
	return unigram

def unigram_sub(unigram, count):
	unigram_filtered={}
	for item in unigram:
		if(int(unigram[item])>count):
			unigram_filtered[item]=unigram[item]
	return unigram_filtered

def get_triples(unigram,voc):
	triples=[]
	t=[]
	for i in range(len(voc)):
		if(i<len(voc)-2):
			try:
				if(unigram[voc[i]] and unigram[voc[i+1]] and unigram[voc[i+2]]):
					t=[voc[i],voc[i+1],voc[i+2]]
					triples.append(t)
			except:
				pass
	return triples


corpus = generate_tweet_corpus("tweets.txt")
corpus = filter_text(corpus)
vocab = derive_vocabulary(corpus)
unigram = get_unigram(vocab)
unigram_filtered = unigram_sub(unigram,1)
triples = get_triples(unigram_filtered,vocab)
print triples
