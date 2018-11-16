import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
from scipy import spatial
import re
import sys

class Embedding	:
	def __init__(self, embedDim):
		self.fileName = "../data/glove/glove.6B."+str(embedDim)+"d.txt"
		self.embedDim = embedDim
		self.wordIdxDic = {}
		self.wordEmbedding = None

		embeddingList = []
		wordIdx = 0
		fin = open(self.fileName,'r')
		while True:
			if(not((wordIdx+1) % 100000)):
			   print ("reading embedding "+str(wordIdx+1))
			line = fin.readline()

			# end of line, stop read
			if not line: break

			token = line.split()
			vector = token[1:]
			embeddingList.append(vector)
			self.wordIdxDic[token[0]] = wordIdx
			wordIdx += 1
		fin.close()

		embedNumpy = np.asarray(embeddingList).astype(float)

		# Out of vocabulary processing
		self.wordIdxDic["<unk>"]=wordIdx
		#embedNumpy = np.append(embedNumpy,[(np.random.rand(embedDim)-0.5)*2],axis=0)
		embedNumpy = np.append(embedNumpy,[(np.zeros(embedDim))],axis=0)
		embedShape = embedNumpy.shape
		
		embedTensor = torch.zeros(embedShape,dtype=torch.float)
		embedTensor =embedTensor.new_tensor(embedNumpy)

		self.wordEmbedding = torch.nn.Embedding.from_pretrained(embedTensor)
	
	def getEmbed(self):
		return self.wordEmbedding
	
	def getWordIdxDic(self):
		return self.wordIdxDic
