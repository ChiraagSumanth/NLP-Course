import numpy as np
from nltk import word_tokenize
import pickle

# data I/O

data = open('people.txt', 'r').read() # should be simple plain text file
#words = word_tokenize(data)
#pickle.dump(words, open("words.pkl","wb"))
words1 = pickle.load(open("words.pkl","rb"))
#chars = list(set(data))
vocab = pickle.load(open("words5000.pkl","rb"))
words2 = []
for w in words1:
  if w in vocab:
    words2.append(w)
  else:
    words2.append('__UNK__')

words = words2[:int(len(words2) * 0.8)]
test_words = words2[int(len(words2) * 0.8):]

vocab.append('__UNK__')

itnum = 100000
data_size, vocab_size = len(words), len(vocab)
print 'data has %d words, %d is the unigram based vocab size.' % (data_size, vocab_size)
word_to_ix = { w:i for i,w in enumerate(vocab) }
ix_to_word = { i:w for i,w in enumerate(vocab) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# Model Parameters
#Forward Direction params
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

#Backward Direction params
Wxh2 = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh2 = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why2 = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh2 = np.zeros((hidden_size, 1)) # hidden bias
by2 = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev, hprev2):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  xs2, hs2 = {}, {}
  hs[-1] = np.copy(hprev)
  hs2[len(inputs)] = np.copy(hprev2)
  loss = 0
  
  # BRNN Forward Pass
  
  # Forward pass (regular)
  for t in xrange(len(inputs)):
	# encode in 1-of-k representation
    xs[t] = np.zeros((vocab_size,1)) 
    xs[t][inputs[t]] = 1
	# hidden state
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) 

  # Forward pass for reverse direction
  for t in reversed(xrange(len(inputs))):
	# encode in 1-of-k representation
    xs2[t] = np.zeros((vocab_size,1)) 
    xs2[t][inputs[t]] = 1
	# hidden state
    hs2[t] = np.tanh(np.dot(Wxh2, xs2[t]) + np.dot(Whh2, hs2[t+1]) + bh2) 
    
  # Output layer pass  
  for t in xrange(len(inputs)):
	# unnormalized log probabilities for next word
    ys[t] = np.dot(Why, hs[t]) + np.dot(Why2, hs2[t]) + (by+by2)
	# probabilities for next word
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
	# softmax (cross-entropy loss)
    loss += -np.log(ps[t][targets[t],0]) 
    
  # BRNN Backward Pass
  
  #Gradient params for forward direction
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  
  #Gradient params for reverse direction
  dWxh2, dWhh2, dWhy2 = np.zeros_like(Wxh2), np.zeros_like(Whh2), np.zeros_like(Why2)
  dbh2, dby2 = np.zeros_like(bh2), np.zeros_like(by2)
  dhnext2 = np.zeros_like(hs2[0])
  
  # Propogating back from output layer first
  for t in xrange(len(inputs)):
	# unnormalized log probabilities for next word
    ys[t] = np.dot(Why, hs[t]) + np.dot(Why2, hs2[t]) + (by+by2)
	# probabilities for next word
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
	# softmax (cross-entropy loss)
    loss += -np.log(ps[t][targets[t],0]) 

    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    
    dy2 = np.copy(ps[t])
    dy2[targets[t]] -= 1 # backprop into y
    dWhy2 += np.dot(dy2, hs2[t].T)
    dby2 += dy2
    dh2 = np.dot(Why2.T, dy2) + dhnext2 # backprop into h
    
  # backward pass: compute gradients going backwards for forward direction
  for t in reversed(xrange(len(inputs))):

    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)

  # backward pass: compute gradients going backwards for reverse direction
  for t in xrange(len(inputs)):
    dhraw2 = (1 - hs2[t] * hs2[t]) * dh2 # backprop through tanh nonlinearity
    dbh2 += dhraw2
    dWxh2 += np.dot(dhraw2, xs2[t].T)
    dWhh2 += np.dot(dhraw2, hs2[t+1].T)
    dhnext2 = np.dot(Whh2.T, dhraw2)
  
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    
  for dparam2 in [dWxh2, dWhh2, dWhy2, dbh2, dby2]:
    np.clip(dparam2, -5, 5, out=dparam2) # clip to mitigate exploding gradients
    
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1], dWxh2, dWhh2, dWhy2, dbh2, dby2, hs2[len(inputs)-1]


def sample(h, h2, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    h2 = np.tanh(np.dot(Wxh2, x) + np.dot(Whh2, h2) + bh2)
    y = np.dot(Why, h) + np.dot(Why2, h2) + (by + by2)
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes
  

# ------------------ Main program ----------------------#

n, p = 0, 0
i = 0

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad

mWxh2, mWhh2, mWhy2 = np.zeros_like(Wxh2), np.zeros_like(Whh2), np.zeros_like(Why2)
mbh2, mby2 = np.zeros_like(bh2), np.zeros_like(by2) # memory variables for Adagrad

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while i<itnum:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(words) or n == 0:
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    hprev2 = np.zeros((hidden_size,1)) # Reset RNN backward too
    p = 0 # go from start of data
  #print words[p:p+seq_length]
  inputs = [word_to_ix[w] for w in words[p:p+seq_length]]
  targets = [word_to_ix[w] for w in words[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, hprev2, inputs[0], 1)
    txt = ''.join(ix_to_word[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev, dWxh2, dWhh2, dWhy2, dbh2, dby2, hprev2 = lossFun(inputs, targets, hprev, hprev2)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
  
  for param2, dparam2, mem2 in zip([Wxh2, Whh2, Why2, bh2, by2],
                                [dWxh2, dWhh2, dWhy2, dbh2, dby2],
                                [mWxh2, mWhh2, mWhy2, mbh2, mby2]):
    mem2 += dparam2 * dparam2
    param2 += -learning_rate * dparam2 / np.sqrt(mem2 + 1e-8) # adagrad update
  
  i+=1
  p += seq_length # move data pointer
  n += 1 # iteration counter
  
# Testing with a forward pass
save_this = [Whh, Wxh, Why, bh, by, Whh2, Wxh2, Why2, bh2, by2]
pickle.dump(save_this, open("trained_model_5k_1L.pkl", "wb"))

#Whh, Wxh, Why, bh, by, Whh2, Wxh2, Why2, bh2, by2 = tuple(pickle.load(open("trained_model_first.pkl", "rb")))
test_inputs = []
test_outputs = []


for i in xrange(len(test_words)-seq_length-1):
  tinputs = [word_to_ix[w] for w in test_words[i : i+seq_length]]
  ttargets = [word_to_ix[test_words[i + seq_length+1]]]
  test_inputs.append(tinputs)
  test_outputs.append(ttargets[0])
	
xso, hso, yso, pso = {}, {}, {}, {}
hso[-1] = np.zeros((hidden_size,1))

perp_check = []
count = 0
for test_input in test_inputs:
  for t in xrange(len(test_input)):
    xso[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xso[t][test_input[t]] = 1
    hso[t] = np.tanh(np.dot(Wxh, xso[t]) + np.dot(Whh, hso[t-1]) + bh) # hidden state
    yso[t] = np.dot(Why, hso[t]) + by # unnormalized log probabilities for next chars
    pso[t] = np.exp(yso[t]) / np.sum(np.exp(yso[t])) # probabilities for next chars
    p = pso[t]
  p_fin = list(p)
  expected = ix_to_word[int(test_outputs[test_inputs.index(test_input)])]
  predicted = str(vocab[p_fin.index(max(p_fin))])
  perp_check.append(predicted)
  #print "Input: "+str(test_input)+" Expected: "+expected+" Prediction: "+predicted
  if expected == predicted:
    count += 1
pickle.dump(perp_check,open("output_5k_1L.pkl","wb"))
print "Efficiency: {}%".format(float(count)*100/len(test_outputs))
