import sys
import numpy as np
from scipy.optimize import minimize as mymin
import math
from utils import *
import pickle

class MyMaxEnt:
	'''
	Generic Maximum Entropy Class 
	'''
	def __init__(self,history,Funcs,Y,T):
		self.Tag = T
		self.model = map(lambda x:x/100, np.random.rand(1,len(Funcs)))
		#self.model = np.asarray([0.005,0.004,0.009,0.001,0.0405,0.006,0.002,0.008,0.002,0.007,0.005,0.001,0.007])
		#self.model = np.asarray([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
		self.Y = Y
		self.Funcs = Funcs
		self.X = history
		self.F = None

	def set_feature_vector(self,p=True):
		"""call feature functions for all x in X and for each label"""
		if not p:
			F = []
			for h,t in zip(self.X,self.Tag):
				f = []
				for func in self.Funcs:
					f.append(func(h,t))
				F.append(np.asarray(f))
			pickle.dump(F, open("feature_vecs.pkl","wb"))
			self.F = F
		else:
			self.F = pickle.load(open("feature_vecs.pkl","rb"))
				
	def cost(self,v):
		'''Function returns the negative of the loss function defined in the slides'''
		term1 = 0
		for i in self.F:
			term1 += np.dot(i,v)
		term2 = 0
		for h,t,f in zip(self.X,self.Tag,self.F):
			e_sum = 0
			for y in self.Y:
				if y != t:
					score = np.dot(f,v)
					e_sum += math.exp(score)
			term2 += math.log(e_sum)
		return term2 - term1
					
		
	def train(self, tr=False):
		''' Train the model using scipy's optimization of minimizing cost function'''
		if(tr):
			self.set_feature_vector()
			#print self.F
			#print '*'*200
			params = mymin(self.cost, self.model, method = 'L-BFGS-B',options={"disp":True})
			#print params
			if(params.success):
				self.model = params.x
			else:
				print "Training Failed"
			pickle.dump(self.model,open("model.pkl","wb"))
			return self.model
		else:
			self.model = pickle.load(open("model.pkl","rb"))
			return self.model
			
	def p_y_given(self,h,tag):
		fv = []
		fvs_prime = []
		for func in self.Funcs:
			fv.append(func(h,tag))
		for y in self.Y:
			if y!=tag:
				fvp = []
				for func in self.Funcs:
					fvp.append(func(h,y))
				fvs_prime.append(fvp)
		e_sum = 0
		for vect in fvs_prime:
			#print vect
			#print self.model
			e_sum += math.exp(np.dot(self.model,vect))
		norm = math.log(e_sum)
		logp = np.dot(self.model,fv) - norm
		return math.exp(logp)
		
	def classify(self,h):
		probs = []
		for y in self.Y:
			probs.append(self.p_y_given(h,y))
		i = probs.index(max(probs))
		return self.Y[i]
		
def accuracy(tes1,tes2):
	''' 
	Input is tagged token list obtained from testing
	'''
	cou = 0
	for i in range(len(tes1)):
		#Comparing tags
		if(tes1[i][1]==tes2[i]):
			cou+=1
	return float(cou)/len(tes1)
		
if __name__=="__main__":
	'''hist=[]
	tagged=[]
	with open("history_sample.txt","r") as f:
		hist = eval(f.read())[0]
	with open("tagged_sample.txt","r") as f:
		tagged = eval(f.read())
	'''
	hist = pickle.load(open('history.pkl','r'))
	tagged = pickle.load(open('tagged.pkl','r'))
	hist_trng = hist[:int(0.8 * len(hist))]
	tagged_trng = list(zip(*tagged[:int(0.8 * len(hist))])[1])
	hist_test = hist[int(0.8 * len(hist)):]
	tagged_test = list(zip(*tagged[int(0.8 * len(hist)):])[1])
	tagSet = ['ORGANIZATION', 'PERSON', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'FACILITY', 'GPE', 'OTHER']
	functs = [f1,f2,f3,f4,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]
	maxEnt = MyMaxEnt(hist_trng,functs,tagSet,tagged_trng)
	print "Before training: ",maxEnt.model
	print(maxEnt.train())	
	output_tag = []
	for h in hist_test:
		op = maxEnt.classify(h)
		print op
		output_tag.append(op)
	print tagged_test
	print output_tag	
	acc = accuracy(tagged_test,output_tag)
	print "Accuracy = {}".format(acc)

