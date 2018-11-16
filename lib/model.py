import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

from wordEmbed import Embedding

import sys
import numpy
import time

class Net(nn.Module):
	def __init__(self,args):
		nn.Module.__init__(self)
		self.args = args
		self.device = torch.device("cuda:"+self.args.gpu if torch.cuda.is_available() else "cpu")

		self.biDirect = True
		if(self.biDirect):
			self.directDim = 2
		else:
			self.directDim = 1

		if(self.args.wiq=='true'):
			self.wiqWDim = 1
			self.wiqBDim = 1
		else:
			self.wiqWDim = 0
			self.wiqBDim = 0

		if(self.args.debug=='false'):
			wordEmbed = Embedding(self.args.embed)
			self.wordIdxDic = wordEmbed.getWordIdxDic()
			self.wordEmbedding = wordEmbed.getEmbed()
			if(self.args.embedFix == 'false'):
				self.wordEmbedding.weight.requires_grad = True
			else:
				self.wordEmbedding.weight.requires_grad = False

		if(self.args.sameLSTM == 'true'):
			# projection matrix
			self.qB = nn.Parameter(torch.cat((torch.eye(self.args.hidden),torch.eye(self.args.hidden)),1).to(self.device))
			self.cB = nn.Parameter(torch.cat((torch.eye(self.args.hidden),torch.eye(self.args.hidden)),1).to(self.device))
			self.lstm = nn.LSTM(self.wiqWDim+self.wiqBDim+self.args.embed,self.args.hidden,self.args.layerNum,batch_first=True,bidirectional=self.biDirect)

#			self.qAttV = nn.Parameter(torch.zeros(self.args.hidden).to(self.device))
#
#			# wiq
#			self.wiqV = nn.Parameter(torch.zeros(self.args.embed).to(self.device))
#
#			# start location
#			self.sLinear = nn.Linear(3*self.args.hidden,self.args.hidden)
#			self.sV = nn.Parameter(torch.zeros(self.args.hidden).to(self.device))
#
#			# end location
#			self.eLinear = nn.Linear(5*self.args.hidden,self.args.hidden)
#			self.eV = nn.Parameter(torch.zeros(self.args.hidden).to(self.device))

			self.qAttV = nn.Parameter(torch.rand(self.args.hidden).to(self.device))

			# wiq
			self.wiqV = nn.Parameter(torch.rand(self.args.embed).to(self.device))

			# start location
			self.sLinear = nn.Linear(3*self.args.hidden,self.args.hidden)
			self.sV = nn.Parameter(torch.rand(self.args.hidden).to(self.device))

			# end location
			self.eLinear = nn.Linear(5*self.args.hidden,self.args.hidden)
			self.eV = nn.Parameter(torch.rand(self.args.hidden).to(self.device))


		else:
			# biLSTM layer
			self.qLSTM = nn.LSTM(self.wiqWDim+self.wiqBDim+self.args.embed,self.args.hidden,self.args.layerNum,batch_first=True,bidirectional=self.biDirect)
			self.cLSTM = nn.LSTM(self.wiqWDim+self.wiqBDim+self.args.embed,self.args.hidden,self.args.layerNum,batch_first=True,bidirectional=self.biDirect)

			self.qAttV = nn.Parameter(torch.rand(self.directDim*self.args.hidden).to(self.device))

			# wiq
			self.wiqV = nn.Parameter(torch.rand(self.args.embed).to(self.device))

			# start location
			self.sLinear = nn.Linear(3*self.directDim*self.args.hidden,self.args.hidden)
			self.sV = nn.Parameter(torch.rand(self.args.hidden).to(self.device))

			# end location
			self.eLinear = nn.Linear(5*self.directDim*self.args.hidden,self.args.hidden)
			self.eV = nn.Parameter(torch.rand(self.args.hidden).to(self.device))

		# no grad for parameters
#		self.qAttV.requires_grad = False
#		self.wiqV.requires_grad = False
#		for param in self.sLinear.parameters():
#			param.requires_grad = False
#		self.sV.requires_grad = False
#		for param in self.eLinear.parameters():
#			param.requires_grad = False
#		self.eV.requires_grad = False

	def init_hidden(self):
#		self.qHidden = (torch.randn(self.directDim*self.args.layerNum, self.args.batch, self.args.hidden)).to(self.device)
#		self.qCell = (torch.randn(self.directDim*self.args.layerNum, self.args.batch, self.args.hidden)).to(self.device)
#		self.cHidden =(torch.randn(self.directDim*self.args.layerNum, self.args.batch, self.args.hidden)).to(self.device)
#		self.cCell = (torch.randn(self.directDim*self.args.layerNum, self.args.batch, self.args.hidden)).to(self.device)

		self.qHidden = (torch.zeros(self.directDim*self.args.layerNum, self.args.batch, self.args.hidden)).to(self.device)
		self.qCell = (torch.zeros(self.directDim*self.args.layerNum, self.args.batch, self.args.hidden)).to(self.device)
		self.cHidden =(torch.zeros(self.directDim*self.args.layerNum, self.args.batch, self.args.hidden)).to(self.device)
		self.cCell = (torch.zeros(self.directDim*self.args.layerNum, self.args.batch, self.args.hidden)).to(self.device)


	def forward(self,queryList,contextList, startIndexList=None, ansSenSelList=None):
		self.init_hidden()

		# pack sequence
		qEmbedPack = self.getEmbedPack(queryList)
		cEmbedPack = self.getEmbedPack(contextList)

		if(self.args.wiq=='true'):
			qEmbedPack,cEmbedPack = self.putWiq(qEmbedPack,cEmbedPack)
#(qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList), (cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList)

		# check for debug 1026
		if(self.args.mode =='debug'):
			qTensor, qList, _,_,_,_ = qEmbedPack
			cTensor, cList, _,_,_,_ = cEmbedPack
			print("query")
			print(qList[0])

			print("context")
			print(cList[0])

		qSortedEmbedPack = self.sortEmbedPack(qEmbedPack)
		cSortedEmbedPack = self.sortEmbedPack(cEmbedPack)

		qSeq = self.packSequence(qSortedEmbedPack)
		cSeq = self.packSequence(cSortedEmbedPack)

		if(self.args.sameLSTM == 'true'):
			qSeqResult,(self.qHidden,self.qCell) = self.lstm(qSeq,(self.qHidden,self.qCell))
			cSeqResult,(self.cHidden,self.cCell) = self.lstm(cSeq,(self.cHidden,self.cCell))
		else:
			qSeqResult,(self.qHidden,self.qCell) = self.qLSTM(qSeq,(self.qHidden,self.qCell))
			cSeqResult,(self.cHidden,self.cCell) = self.cLSTM(cSeq,(self.cHidden,self.cCell))

		qSortedEmbedPack = self.unPackSequence(qSeqResult,qEmbedPack)
		cSortedEmbedPack = self.unPackSequence(cSeqResult,cEmbedPack)

		qEmbedPack = self.unsortEmbedPack(qSortedEmbedPack)
		cEmbedPack = self.unsortEmbedPack(cSortedEmbedPack)
		if(self.args.mode =='debug'):
			qTensor, qList, _,_,_,_ = qEmbedPack
			cTensor, cList, _,_,_,_ = cEmbedPack

			print("LSTM passed queryTensor")
			print(qTensor)
			print("LSTM passed contextTensor")
			print(cTensor.size())
			print(cTensor)


		# projection
		if(self.args.sameLSTM == 'true'):
			qEmbedPack, cEmbedPack = self.project(qEmbedPack,cEmbedPack)
		
		# train
		if(startIndexList is not None):
			qEmbedPackAtt = self.getAttQ(qEmbedPack)
			outputTensor = self.getTrainAnswerProb(qEmbedPackAtt,cEmbedPack,startIndexList)
			# senAtt
			if(ansSenSelList is not None):
				# train senAtt
				senProbTensor =self.getSenProb(qEmbedPack,cEmbedPack,ansSenSelList)
				return outputTensor, senProbTensor
			else:
				return outputTensor
		# test
		else:
			qEmbedPackAtt = self.getAttQ(qEmbedPack)	
			#outputTensor = self.getTrainAnswerProb(qEmbedPackAtt,cEmbedPack)
			#return outputTensor
			outputTensorList = self.getTestAnswerCandidateProbList(qEmbedPackAtt,cEmbedPack)
			return outputTensorList

	def project(self,qEmbedPack, cEmbedPack):
		qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList = qEmbedPack
		cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList = cEmbedPack

		tanh = nn.Tanh()
		qBatchTensor = qBatchTensor.transpose(1,2)
		cBatchTensor = cBatchTensor.transpose(1,2)

		qBatchTensor = tanh(torch.matmul(self.qB,qBatchTensor))
		cBatchTensor = tanh(torch.matmul(self.cB,cBatchTensor))

		qBatchTensor = qBatchTensor.transpose(2,1)
		cBatchTensor = cBatchTensor.transpose(2,1)

		return (qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList), (cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList)

	def getTrainAnswerProb(self,qEmbedPack, cEmbedPack, startIndexList =None):
		qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList = qEmbedPack
		cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList = cEmbedPack

		startProbTensor = self.getSj(qEmbedPack,cEmbedPack)

		# for test
		if(startIndexList is None):
			startIndexList = []
			for i in range(self.args.batch):
				contextLen = cUnsortedTokenLenList[i]
				startIndex = torch.max(startProbTensor[i][:contextLen],0)[1]
				startIndexList.append(startIndex)

		endProbTensor =self.getEj(qEmbedPack,cEmbedPack,startIndexList)
		return torch.cat((startProbTensor,endProbTensor),0)

	def getTestAnswerCandidateProbList(self,qEmbedPack, cEmbedPack):
		qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList = qEmbedPack
		cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList = cEmbedPack

		startProbTensor = self.getSj(qEmbedPack,cEmbedPack)

		answerBeamTensorList = []
		for i in range(self.args.beam):
			startIndexList = []
			for j in range(self.args.batch):
				contextLen = cUnsortedTokenLenList[j]
				startIndexBeamList = torch.topk(startProbTensor[j][:contextLen],self.args.beam,0)
				startIndex = startIndexBeamList[1][i].data.cpu().numpy().item()
				startIndexList.append(startIndex)
			endProbTensor =self.getEj(qEmbedPack,cEmbedPack,startIndexList)
			answerTensor = torch.cat((startProbTensor,endProbTensor),0)
			answerBeamTensorList.append(answerTensor)
		return answerBeamTensorList

	def packSequence(self,embedPack):
		batchTensor, sentenceList, sortIndexList, unsortIndexList, sortedTokenLenList, unsortedTokenLenList = embedPack
		resultSeq = torch.nn.utils.rnn.pack_padded_sequence(batchTensor,sortedTokenLenList,batch_first=True)
		return resultSeq

	def unPackSequence(self,inputSeq,embedPack):
		batchTensor, sentenceList, sortIndexList, unsortIndexList, sortedTokenLenList, unsortedTokenLenList = embedPack
		unPackedTensor, _ = torch.nn.utils.rnn.pad_packed_sequence(inputSeq,batch_first=True,padding_value=0)
		return unPackedTensor,sentenceList, sortIndexList, unsortIndexList, sortedTokenLenList, unsortedTokenLenList

	def getEmbedPack(self,sentenceList):
		sortedSentenceList, sortIndexList,unsortIndexList,sortedTokenLenList,unsortedTokenLenList = self.sortSentenceList(sentenceList)

		maxTokenLen = max(sortedTokenLenList)

		# make padded tensor
		sortedSenTensorList = []

		for i in range(self.args.batch):
			sortedSenTokenList = sortedSentenceList[i].split(' ')
			sortedSenTokenLen = len(sortedSenTokenList)

			# embedding on
			if(self.args.debug=='false'):
				# oov process
				for j in range(sortedSenTokenLen):
					sortedSenTokenList[j] = sortedSenTokenList[j].lower()
					if(sortedSenTokenList[j] not in self.wordIdxDic.keys()):
						sortedSenTokenList[j] = "<unk>"

				idxs = [self.wordIdxDic[w] for w in sortedSenTokenList]
				idxTensor = torch.LongTensor(idxs).to(self.device)
				sortedSenTensor = self.wordEmbedding(idxTensor)
			else:
				sortedSenTensor = torch.rand(sortedSenTokenLen,self.args.embed).to(self.device)

			# word dropout
			if(self.args.mode == 'train'):
				word_dropout = nn.Dropout(0.5)
				sortedSenTensor = word_dropout(sortedSenTensor)
			sortedSenTensorList.append(sortedSenTensor)

		# pad and make batch
		batchTensor = torch.nn.utils.rnn.pad_sequence(sortedSenTensorList,batch_first =True)
		unsortIndexTensor = torch.from_numpy(numpy.asarray(unsortIndexList)).to(self.device).unsqueeze(1).unsqueeze(1).expand(batchTensor.size(0),batchTensor.size(1),batchTensor.size(2))
		unsortedTensor = batchTensor.gather(0,unsortIndexTensor)

		return unsortedTensor, sentenceList, sortIndexList, unsortIndexList, sortedTokenLenList, unsortedTokenLenList

	def sortEmbedPack(self,embedPack):
		batchTensor, sentenceList, sortIndexList, unsortIndexList, sortedTokenLenList, unsortedTokenLenList = embedPack
		sortIndexTensor = torch.from_numpy(numpy.asarray(sortIndexList)).to(self.device).unsqueeze(1).unsqueeze(1).expand(batchTensor.size(0),batchTensor.size(1),batchTensor.size(2))
		sortedTensor = batchTensor.gather(0,sortIndexTensor)
		return sortedTensor, sentenceList, sortIndexList, unsortIndexList, sortedTokenLenList, unsortedTokenLenList

	def unsortEmbedPack(self,embedPack):
		batchTensor, sentenceList, sortIndexList, unsortIndexList, sortedTokenLenList, unsortedTokenLenList = embedPack
		unsortIndexTensor = torch.from_numpy(numpy.asarray(unsortIndexList)).to(self.device).unsqueeze(1).unsqueeze(1).expand(batchTensor.size(0),batchTensor.size(1),batchTensor.size(2))
		unsortedTensor = batchTensor.gather(0,unsortIndexTensor)
		return unsortedTensor, sentenceList, sortIndexList, unsortIndexList, sortedTokenLenList, unsortedTokenLenList

	def putWiq(self,qEmbedPack,cEmbedPack):
		qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList = qEmbedPack
		cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList = cEmbedPack

		qWiqWTensor = self.getWiqW(qBatchTensor)
		cWiqWTensor = self.getWiqW(cBatchTensor, (qBatchTensor,qUnsortedTokenLenList,cUnsortedTokenLenList))

		qWiqBTensor = self.getWiqB(qBatchTensor)
		cWiqBTensor = self.getWiqB(cBatchTensor, (qSentenceList, cSentenceList))

		qBatchTensor = torch.cat((qWiqWTensor,qWiqBTensor,qBatchTensor),2)
		cBatchTensor = torch.cat((cWiqWTensor,cWiqBTensor,cBatchTensor),2)
		if(self.args.mode =='debug'):
			print("cWiqW")
			print(cWiqWTensor.size())
			print(cWiqWTensor)
			print("cWiqB")
			print(cWiqBTensor.size())
			print(cWiqBTensor[0])
#			print("cBatchTensor")
#			print(cBatchTensor.size())
#			print(cBatchTensor[0])

		return (qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList), (cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList)

	def getWiqW(self,batchTensor, contextParameter= None):
		# query
		if(contextParameter is None):
			wiqWTensor = torch.ones(batchTensor.size(0),batchTensor.size(1),1).to(self.device)
		else:
			cBatchTensor = batchTensor
			qBatchTensor, qUnsortedTokenLenList, cUnsortedTokenLenList = contextParameter[0],contextParameter[1],contextParameter[2]
			softMaxMaskTensor1 = torch.zeros(self.args.batch, list(qBatchTensor.size())[1], list(cBatchTensor.size())[1]).to(self.device)
			for i in range(self.args.batch):
				if(qUnsortedTokenLenList[i]<softMaxMaskTensor1.size(1)):
					softMaxMaskTensor1[i,qUnsortedTokenLenList[i]:,:] = -10**10
				if(cUnsortedTokenLenList[i]<softMaxMaskTensor1.size(2)):
					softMaxMaskTensor1[i,:,cUnsortedTokenLenList[i]:] = -10**10

			softMaxMaskTensor2 = torch.ones(self.args.batch, list(qBatchTensor.size())[1], list(cBatchTensor.size())[1]).to(self.device)
			for i in range(self.args.batch):
				if(qUnsortedTokenLenList[i]<softMaxMaskTensor2.size(1)):
					softMaxMaskTensor2[i,qUnsortedTokenLenList[i]:,:] = 0
				if(cUnsortedTokenLenList[i]<softMaxMaskTensor2.size(2)):
					softMaxMaskTensor2[i,:,cUnsortedTokenLenList[i]:] = 0

			# complete wiqW
			xj = cBatchTensor.unsqueeze(1)
			qi = qBatchTensor.unsqueeze(2)
			xj_qi = torch.mul(xj,qi)
			sim_ij = torch.squeeze(torch.matmul(self.wiqV.unsqueeze(0).unsqueeze(0).unsqueeze(0),xj_qi.transpose(2,3)))
			if(self.args.batch==1):
				sim_ij = sim_ij.unsqueeze(0)

			softmax = nn.Softmax(dim=2)
			#wiqWTensor= torch.sum(softmax(sim_ij),1).unsqueeze(2)
			# this mask for cWiqW
			wiqWTensor= torch.sum(softmax(sim_ij+softMaxMaskTensor1)*softMaxMaskTensor2,1).unsqueeze(2)

		return wiqWTensor


	def getWiqB(self,batchTensor, contextParameter = None):
		if(contextParameter is None):
			wiqBTensor = torch.ones(batchTensor.size(0),batchTensor.size(1),1).to(self.device)
		else:
			cBatchTensor = batchTensor
			qSentenceList,cSentenceList = contextParameter[0],contextParameter[1]
			wiqBTensor = torch.zeros(cBatchTensor.size(0),cBatchTensor.size(1),1).to(self.device)
			for i in range(self.args.batch):
				qTokenList = qSentenceList[i].split(' ')
				qTokenLen = len(qTokenList)

				cTokenList = cSentenceList[i].split(' ')
				cTokenLen = len(cTokenList)

				for j in range(cTokenLen):
					for k in range(qTokenLen):
						if(cTokenList[j].lower() == qTokenList[k].lower()):
							wiqBTensor[i][j][0] = 1
							continue
		return wiqBTensor

	def sortSentenceList(self,sentenceList):
		batchSize = len(sentenceList)

		seqInfoList = []
		unsortedTokenLenList = []
		for i in range(batchSize):
			sentence = sentenceList[i]
			tokenLen = len(sentence.split(' '))
			unsortedTokenLenList.append(tokenLen)
			seqInfoList.append((i,sentence,tokenLen))

		seqInfoList.sort(key=lambda x:x[2],reverse=True)
		sortIndexList,sortedSentenceList,sortedTokenLenList = zip(*seqInfoList)

		sortIndexList = list(sortIndexList)
		sortedSentenceList = list(sortedSentenceList)
		sortedTokenLenList = list(sortedTokenLenList)

		unsortIndexList = [0]*len(sortIndexList)
		for i in range(len(sortIndexList)):
			unsortIndexList[sortIndexList[i]]=i

		return sortedSentenceList,sortIndexList,unsortIndexList,sortedTokenLenList,unsortedTokenLenList

	def getAttQ(self,qEmbedPack):
		qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList = qEmbedPack
		softMaxMaskTensor = torch.zeros(qBatchTensor.size(0),qBatchTensor.size(1)).to(self.device)
		for i in range(self.args.batch):
			if(qUnsortedTokenLenList[i]<softMaxMaskTensor.size(1)):
				softMaxMaskTensor[i][qUnsortedTokenLenList[i]:]= -10**10

		softmax = nn.Softmax(dim=1)
		vqZ = torch.squeeze(torch.matmul(self.qAttV.unsqueeze(0).unsqueeze(0),qBatchTensor.transpose(1,2)))
		if(self.args.batch==1):
			vqZ = vqZ.unsqueeze(0)
		# no masking 1027
		#alpha = softmax(vqZ).unsqueeze(2)
		alpha = softmax(vqZ+softMaxMaskTensor).unsqueeze(2)
		if(self.args.mode =='debug'):
#			print("qAttV")
#			print(self.qAttV.size())
#			print(self.qAttV)
			print("qBatchTensor")
			print(qBatchTensor.size())
			print(qBatchTensor)
#			print("vqZ")
#			print(vqZ.size())
#			print(vqZ)
#			print("softMaxMaskTensor")
#			print(softMaxMaskTensor)
			print("before softmax")
			print(vqZ)
			#print(vqZ+softMaxMaskTensor)
			print("alpha")
			print(alpha.size())
			print(alpha)
			print("qBatchTensorBefore")
			print(qBatchTensor)
			print("qBatchTensorAfter")
			print(torch.sum(qBatchTensor*alpha,1).unsqueeze(1))
			sys.exit()

		qBatchTensor = torch.sum(qBatchTensor*alpha,1).unsqueeze(1)
		return 	qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList

	def splitAnsSenList(self, ansSenList):
		batchAnsSenBeginIdxList = []
		batchAnsSenEndIdxList = []
		senNumList = []
		for i in range(len(ansSenList)):
			ansSenTupleList = ansSenList[i][0]
			ansSenBeginList = []
			ansSenEndList = []
			for j in range(len(ansSenTupleList)):
				ansSenBeginList.append(ansSenTupleList[j][0])
				ansSenEndList.append(ansSenTupleList[j][1])
			senNumList.append(len(ansSenBeginList))
			batchAnsSenBeginIdxList.append(ansSenBeginList)
			batchAnsSenEndIdxList.append(ansSenEndList)

		# padding list to zero
		maxSenNum = max(senNumList)
		for i in range(self.args.batch):
			ansSenBeginList = batchAnsSenBeginIdxList[i]
			ansSenEndList = batchAnsSenEndIdxList[i]
			for j in range(maxSenNum):
				if(j>len(ansSenBeginList)-1):
					ansSenBeginList.append(0)
				if(j>len(ansSenEndList)-1):
					ansSenEndList.append(0)

		return batchAnsSenBeginIdxList, batchAnsSenEndIdxList, senNumList

	def getSenProb(self,qEmbedPack,cEmbedPack,ansSenList):
		qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList = qEmbedPack
		cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList = cEmbedPack

		ansSenBeginList, ansSenEndList, senLenList = self.splitAnsSenList(ansSenList)

		maxSenLen = max(senLenList)
		ansSenBatchTensor = torch.zeros(cBatchTensor.size(0),maxSenLen,cBatchTensor.size(2)).to(self.device)
		for i in range(self.args.batch):
			for j in range(senLenList[i]):
				ansSenTensor = torch.max(cBatchTensor[i,ansSenBeginList[i][j]:ansSenEndList[i][j],:],0)[0]
				ansSenBatchTensor[i,j]=ansSenTensor

		maxQueryLen = max(qUnsortedTokenLenList)
		querySenBatchTensor = torch.zeros(qBatchTensor.size(0),1,qBatchTensor.size(2)).to(self.device)
		for i in range(self.args.batch):
			temp = qBatchTensor[i,:qUnsortedTokenLenList[i],:]
			querySenTensor = torch.max(temp,0)[0]
			querySenBatchTensor[i,0,:] = querySenTensor
		querySenBatchTensor = querySenBatchTensor.expand(ansSenBatchTensor.size(0),ansSenBatchTensor.size(1),ansSenBatchTensor.size(2))
	
		psenj =F.cosine_similarity(querySenBatchTensor,ansSenBatchTensor,2)

		psenjMaskTensor = torch.zeros(self.args.batch, psenj.size(1)).to(self.device)

		for i in range(self.args.batch):
			psenjMaskTensor[i][senLenList[i]:]= -10**10

		# no mask tensor 1027
		#psenj += psenjMaskTensor

		return psenj

	def getSj(self,qEmbedPack,cEmbedPack):
		qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList = qEmbedPack
		cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList = cEmbedPack

		hj=cBatchTensor
		z = qBatchTensor.expand(qBatchTensor.size(0),cBatchTensor.size(1),qBatchTensor.size(2))
		hj_z = torch.mul(hj,z)

		sj = F.relu(self.sLinear(torch.cat((hj,z,hj_z),2)))
		psj = torch.squeeze(torch.matmul(self.sV.unsqueeze(0).unsqueeze(0),sj.transpose(1,2)))
		if(self.args.batch==1):
			psj = psj.unsqueeze(0)

		psjMaskTensor = torch.zeros(self.args.batch, psj.size(1)).to(self.device)
		for i in range(self.args.batch):
			psjMaskTensor[i][cUnsortedTokenLenList[i]:]= -10**10
		psj = psj+psjMaskTensor

		return psj


	def getEj(self,qEmbedPack, cEmbedPack, startIndexList):
		qBatchTensor, qSentenceList, qSortIndexList, qUnsortIndexList, qSortedTokenLenList, qUnsortedTokenLenList = qEmbedPack
		cBatchTensor, cSentenceList, cSortIndexList, cUnsortIndexList, cSortedTokenLenList, cUnsortedTokenLenList = cEmbedPack

		sjIndexTensor = torch.from_numpy(numpy.asarray(startIndexList)).to(self.device).unsqueeze(1).unsqueeze(1).expand(cBatchTensor.size(0),cBatchTensor.size(1),cBatchTensor.size(2))

		hj=cBatchTensor
		hs=cBatchTensor.gather(1,sjIndexTensor)
		z = qBatchTensor.expand(qBatchTensor.size(0),cBatchTensor.size(1),qBatchTensor.size(2))

		hj_z = torch.mul(hj,z)
		hj_hs = torch.mul(hj,hs)

		ej = F.relu(self.eLinear(torch.cat((hj,hs,z,hj_z,hj_hs),2)))
		pej = torch.squeeze(torch.matmul(self.eV.unsqueeze(0).unsqueeze(0),ej.transpose(1,2)))
		if(self.args.batch==1):
			pej = pej.unsqueeze(0)

		pejMaskTensor = torch.zeros(self.args.batch, pej.size(1)).to(self.device)
		for i in range(self.args.batch):
			pejMaskTensor[i][:startIndexList[i]]= -10**10
			pejMaskTensor[i][cUnsortedTokenLenList[i]:]= -10**10

		pej = pej+pejMaskTensor

		return pej
