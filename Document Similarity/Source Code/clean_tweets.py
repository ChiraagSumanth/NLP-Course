import pickle
import re
import HTMLParser
from nltk import word_tokenize
import heapq
import sys
from nltk.stem.lancaster import LancasterStemmer
reload(sys)
sys.setdefaultencoding('utf8')

html_parser = HTMLParser.HTMLParser()

def clean_tweets(filename="tweets.txt"):
	"""
		Function to clean up tweets by:
		1) Escaping HTML
		2) Decoding
		3) Removing Retweets and Hashtags
		4) Replacing URLs and handles 
		5) Converting Case to lower
	"""
	f = open(filename, "r")
	l = f.readlines()
	f.close()
	clean_tweets = set()
	for line in l:
		tw = html_parser.unescape(line.strip())
		tw = tw.decode("utf8").encode('ascii','ignore') 									# Decoding string
		tw,subs = re.subn(r'RT @[a-zA-Z]*:\b|[a-zA-Z]*:\b|rt','RT',tw) 						# Removing retweets
		tw,subs = re.subn(r"'",'',tw) 														# Removing apostrophe due to string warping
		tw,subs = re.subn(r'@[a-zA-Z0-9_]*\b','handle',tw) 									# Removing screen names
		tw,subs = re.subn(r'#[a-zA-Z0-9_]*\b','hashtag',tw) 								# Removing hashtags
		tw,subs = re.subn(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|\
		(?:%[0-9a-fA-F][0-9a-fA-F]))+|http$','url',tw) 										# Removing URLs
		tw,subs = re.subn(r'(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]\
		|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)','emoticon',tw)			# Removing emoticons
		tw,subs = re.subn(r"""[-_/#"'|\`:%]|[.?!]{2,}""",'',tw) 							# Removing unnecessary punctuation
		tw,subs = re.subn(r'[-+]?[\d]+(.[\d]+(e[\d]+))?','num',tw)
		tweet = ' '.join(tw.strip().split()) 												# Removing whitespaces in the middle and end
		if(tweet and len(tweet)>20):
			clean_tweets.add(tweet.lower())													# Converting to lowercase
	f = open("cleaned_tweets.txt","w")
	for i in clean_tweets:
		f.write(i+'\n')
	f.close()
	return list(clean_tweets)
	

def construct_vocabulary(tweets,n,stemming=False):
	"""
		Utility function to return n words with highest unigram frequencies
	"""
	
	d = {}
	puncts = r'#=^+-!:-;,{}()[]/\_@'
	if not stemming:
		for i in tweets:
			l1 = word_tokenize(i)
			l = [x for x in l1 if x not in puncts]
			for word in l:
				if d.has_key(word):
					d[word]+=1
				else:
					d[word]=1
		return heapq.nlargest(n ,d, key = lambda k: d[k])
	else:
		st = LancasterStemmer()
		for i in tweets:
			l1 = word_tokenize(i)
			l = [x for x in l1 if x not in puncts]
			for word in l:
				stemmed = st.stem(word)
				if d.has_key(stemmed):
					d[stemmed]+=1
				else:
					d[stemmed]=1
		return heapq.nlargest(n ,d, key = lambda k: d[k])


def construct_tuples(samples,vocab,size=3):
	"""
		Make lists of given size from each sentence in samples using a sliding 
		window technique	
	"""
	d = {}
	tups = []
	puncts = r'#=^+-!:-;,{}()[]/\_@'
	for i in samples:
		l1 = word_tokenize(i)
		l = [x for x in l1 if x not in puncts]
		for j in range(len(l)-3):
			nlist = []
			for k in range(3):
				if l[j+k] in vocab:
					nlist.append(l[j+k])
				else:
					nlist.append('__UNK__')
			tups.append(nlist)
	return tups
			
if __name__=="__main__":
	tweets = clean_tweets()
	'''
	vocab = construct_vocabulary(tweets,2000)
	vocab.append('__UNK__')
	tups = construct_tuples(tweets,vocab)
	pickle.dump(tups,open("tups.pkl","wb"))
	pickle.dump(vocab,open("vocab.pkl","wb"))
	'''
	tups = pickle.load(open("tups.pkl","rb"))
	vocab = pickle.load(open("vocab.pkl","rb"))
	#print vocab[:200]
