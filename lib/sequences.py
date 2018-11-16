import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

import sys

def padZero(dataEmbedList,maxLength):
	batchSize = len(dataEmbedList)
	embedDim = len(dataEmbedList[0][0])

	paddingValue = torch.zeros(1,embedDim)
	tempEmbedList=dataEmbedList

	for i in range(batchSize):
		dataLen = list(dataEmbedList[i].size())[0]
		for j in range(maxLength):
			if (j>(dataLen-1)):
				tempEmbedList[i] = torch.cat((tempEmbedList[i],paddingValue),0)

	for i in range(len(tempEmbedList)):
		if (i==0):
			tempEmbedTensor = tempEmbedList[i].unsqueeze(0)
		else:
			tempEmbedTensor = torch.cat((tempEmbedTensor,tempEmbedList[i].unsqueeze(0)),0 )
	return tempEmbedTensor

class Seq:
	def __init__(self,stringList):
		self.seqList = []
		self.maxLen = 0
		# for seq info
		self.seqIndexList = None
		self.seqDataList = None
		self.seqLenList = None

		dataLenList = []
		for i in range(len(embedBatchList)):
			index = i
			embedData =embedBatchList[i]
			dataLen = list(embedBatchList[i].size())[0] # for tensor
			dataLenList.append(dataLen)
			self.seqList.append((index,embedData,dataLen))

		self.maxLen = max(dataLenList)

		# Order for Sequence
		self.seqList.sort(key=lambda x:len(x[1]),reverse=True)
		self.seqIndexList,self.seqDataList,self.seqLenList = zip(*self.seqList)
		self.seqIndexList = list(self.seqIndexList)
		self.seqDataList = list(self.seqDataList)
		self.seqLenList = list(self.seqLenList)

		self.seqInput = padZero(self.seqDataList,self.maxLen)

		self.seq = torch.nn.utils.rnn.pack_padded_sequence(self.seqInput,self.seqLenList,batch_first=True)

	def getSeq(self):
		return self.seq
	
	def setResultSeq(self,resultSeq):
		self.resultSeq = resultSeq
		return None

	def getOutputList(self):
		self.unpackSeq, self.unpackLen = torch.nn.utils.rnn.pad_packed_sequence(self.resultSeq,batch_first=True,padding_value=0)

		# index list for unsort
		self.sortIdx = [0]*len(self.seqIndexList)
		for i in range(len(self.seqIndexList)):
			self.sortIdx[self.seqIndexList[i]] = i

		# reorder
		self.sortIdxBatch =torch.LongTensor(self.sortIdx).unsqueeze(1).unsqueeze(1).expand(self.unpackSeq.size(0),self.unpackSeq.size(1),self.unpackSeq.size(2))
		self.unpackReorder = self.unpackSeq.gather(0,self.sortIdxBatch)
		self.origLenList = self.unpackLen.gather(0,torch.LongTensor(self.sortIdx))

		return (self.unpackReorder,self.origLenList)

