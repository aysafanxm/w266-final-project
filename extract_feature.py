# --*-- encoding:utf-8 --*--
import logging
import multiprocessing
import numpy as np

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors

feature_size = 4

'''
word2vec
dimension：feature_size，Iterations：200
'''
def word_to_vec(file_in, file_out1, file_out2):
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	model = Word2Vec(LineSentence(file_in), size=feature_size, window=5, min_count=1, workers=multiprocessing.cpu_count(), iter=200)
	model.save(file_out1)
	model.wv.save_word2vec_format(file_out2, binary=False)

def read_sentence(file):
	res = []
	with open(file, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	for i in range(len(lines)):
		line = lines[i].split()
		res.append(line)
	return res

def main():
	train_vector = True		# True = Train, False = Take a trained model

	if train_vector:
		word_to_vec('data/sentence.txt', 'model/word.model', 'model/word.vector')
	
	# tale trained model
	word_vectors = KeyedVectors.load_word2vec_format('model/word.vector', binary=False)

	feature = []		# Feature vector of all sentences
	sentence = read_sentence('data/sentence.txt')
	for i in range(len(sentence)):
		fea = np.zeros(feature_size)		# Feature vector of a specific sentence
		for j in range(len(sentence[i])):
			v = word_vectors[sentence[i][j]]		# word vector in the sentence
			fea += v
		fea = fea / len(sentence[i])		# feature vector of the sentence, weighted average of the word vectors
		feature.append(fea)

	# Add feature vector into the feature.txt file
	with open('data/feature.txt', 'w', encoding='utf-8') as f:
		for i in range(len(feature)):
			for j in range(feature[i].shape[0]):
				f.write(str(feature[i][j]))
				if j != feature[i].shape[0]-1:
					f.write('\t')
			f.write('\n')

if __name__ == '__main__':
	main()