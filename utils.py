import argparse
import gzip 
import cPickle as pickle
import sys
import numpy as np
import theano

def prepare_data(in_file_name):
	try:
		f = gzip.open(in_file_name)
	except:
		f = open(file_name)

	voc = {}
	word2id = {}
	id2word = {}
	for index,line in enumerate(f):
		l = line.strip().split()
		word = l[0]
		vec = [ float(v) for v in l[1:]]
		voc[word] = vec
		K = len(vec)

		word2id[word] = index
		id2word[index] = word

	V = len(voc)

	X = np.zeros((V, K), dtype=theano.config.floatX)
	for i in xrange(K):
		word = id2word[i]
		X[i] = voc[word]

	print >> sys.stderr, "Vocab size %d, dimensions %d" % (V,K)
	return X,word2id,id2word

def save_data(out_file_name,data,zipped = True):
	if zipped:
		f = gzip.open(out_file_name,'w')
	else:
		f = open(out_file_name,'w')

	pickle.dump(data,f)
	print >> sys.stderr, "Data is dumped into",out_file_name
	f.close()

def load_data(file_name):
	try:
		f = gzip.open(file_name)
	except:
		f = open(file_name)

	try:
		data = pickle.load(f)
	except:
		print >> sys.stderr, "cannot load data from %s !" % (file_name)
		quit(1)
	return data

def get_parser_AE():
	parser = argparse.ArgumentParser()
	parser.add_argument('--lrate', action='store', dest='learning_rate',help='Learning Rate, default = 0.5',type=float,default = 0.5)

	parser.add_argument('--epochs', action='store', dest='n_epochs',help='# of epochs, default = 50',type=int,default = 50)

	parser.add_argument('--vector-file', action='store', dest='vector_file',help='word vector files',default = '')

	parser.add_argument('--batch-size', action='store', dest='batch_size',help='batch size, default 20', type=int, default = 20)

	parser.add_argument('--hidden', action='store', dest='hidden_size',help='hidden layer size, default = 500',type=int,default = 1000)

	parser.add_argument('--contraction-flag', action='store_true', dest='contraction_flag',help='Contractive AE {True | False} , default = False')

	parser.add_argument('--contraction-level', action='store', dest='contraction_level',help='contraction level, default = 0.1 (if contraction-flag is set)',type=float,default = 0.1)

	parser.add_argument('--sparse-flag', action='store_true', dest='sparse_flag',help='Sparse AE {True | False} , default = False')

	parser.add_argument('--sparsity-penalty', action='store', dest='sparsity_penalty',help='sparsity penalty, default = 0.001 (if sparsity-flag is set)',type=float,default = 0.001)

	parser.add_argument('--sparsity-level', action='store', dest='sparsity_level',help='sparsity level, default = 0.05 (if sparsity-flag is set)',type=float,default = 0.05)

	parser.add_argument('--corruption-rate', action='store', dest='corruption_rate',help='corruption rate, default = 0.0',type=float,default = 0.0)

	parser.add_argument('--activation', action='store', dest='activation',help='activation function {sigmoid,tanh,softplus}, default = sigmoid',default = "sigmoid")

	parser.add_argument('--cost-type', action='store', dest='cost_type',help='cost type {MeanSquaredCost,CrossEntropy,CategoricalCrossEntropy}, default = MeanSquaredCost',default = "MeanSquaredCost")

	parser.add_argument('--output-fname', action='store', dest='fname',help='output filename, default = ae with parameter details appended +.pkl',default = "ae")

	parser.add_argument('--tied-weights', action='store_true', dest='tied_weights',help='FLAG : tie weight matrix for encoding and decoding')

	parser.add_argument('--opt', action='store', dest='optimization',help='optimization method { adagrad | adadelta | adam | rmsprop | nag | momentum | sgd }, default = adadelta',default = 'adadelta')

	parser.set_defaults(contraction_flag = False)
	parser.set_defaults(sparse_flag = False)
	parser.set_defaults(tied_weights = False)

	parser.add_argument('--device', action='store', dest='device',help='use gpu or cpu, default = gpu0',default='gpu0')

	return parser
