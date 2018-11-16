import torch
import csv
import sys
import nltk
import time
import string
import re

def rangeToPair(rangeStr):
	rangeIndex = rangeStr.split(":")
	return (int(rangeIndex[0]),int(rangeIndex[1]))

def getAnswerList(context, answerTokenList):
	answerList = []
	contextToken = context.split(' ')
	for i in range(len(answerTokenList)):
		answerString = ""
		for j in range(answerTokenList[i][0],answerTokenList[i][1]):
			if(j==answerTokenList[i][0]):
				answerString += contextToken[j]
			else:
				answerString += " "+ contextToken[j]
		answerList.append(answerString)
	return answerList

def listToString(inputList):
	resultString = ""
	for i in range(len(inputList)):
		if(i==0):
			resultString += inputList[i]
		else:
			resultString += ", "+inputList[i]
	return resultString

def normalize_answer(s):
	def remove_articles(text):
		return re.sub(r'\b(a|an|the)\b',' ',text)
	def white_space_fix(text):
		return ' '.join(text.split())
	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)
	def lower(text):
		return text.lower()

	return white_space_fix(remove_articles(remove_punc(lower(s))))


class DataList:
	def __init__(self,fileName,batchSize, maxTokenLen,queryFilter):
		self.batchSize = batchSize
		self.maxTokenLen = maxTokenLen
		self.queryFilter = queryFilter
		self.dataIndex = 0
		self.fin = open(fileName,"r")
		self.reader = csv.DictReader(self.fin, delimiter=',',quotechar = '"')
		self.dataPack = []
		self.contextDic = {}
		self.dataNum =0

		for row in self.reader:
			query = row['question']
			context_id = row['story_id']
			context_text = row['story_text']
			context_len = len(context_text)
			answer_token_ranges = row['answer_token_ranges']

			answerRangePair = rangeToPair(answer_token_ranges)
			self.dataPack.append((query,context_id,answerRangePair,context_len))

			if(context_id not in self.contextDic.keys()):
				self.contextDic[context_id]=context_text
			self.dataNum +=1
		# sort by context len
		self.dataPack = sorted(self.dataPack,key=lambda x: x[3])

	def getBatchData(self,device):
		# queryList, contextList, startIndexList, endIndexList
		time1 = time.time()
		queryList, contextList, contextIdList, startIndexList, endIndexBoundaryList = self.unPack()
		time2 = time.time()

		# (senIdxList,ansSenIdx)
		senIdxList = self.getSenIdxList(contextList,startIndexList)
		time3 = time.time()

		# targetAnswerTensor
		targetStartTensor = self.getTargetTensor(startIndexList,device)
		# copy endIndex for endTensor
		#tensorEndIndexList = endIndexBoundaryList[:]
		tensorEndIndexList = []
		for i in range(len(endIndexBoundaryList)):
			tensorEndIndexList.append(endIndexBoundaryList[i]-1)
		time4 = time.time()

		targetEndTensor = self.getTargetTensor(tensorEndIndexList,device)
		targetAnswerTensor = torch.cat((targetStartTensor,targetEndTensor),0)
		time5 = time.time()

		# targetSenTensor
		ansSenIdxList = []
		for i in range(len(senIdxList)):
			ansSenIdxList.append(senIdxList[i][1])
		targetSenTensor = self.getTargetTensor(ansSenIdxList,device)
		time6 = time.time()

		#print(time2-time1)
		#print(time3-time2)
		#print(time4-time3)
		#print(time5-time4)
		#print(time6-time5)
		#sys.exit()

		return queryList, contextList, contextIdList, startIndexList, endIndexBoundaryList, senIdxList, targetAnswerTensor, targetSenTensor

	def getTargetTensor(self, indexList,device):
		resultTensor = torch.zeros(self.batchSize,dtype=torch.long).to(device)
#		resultTensor = torch.zeros(self.batchSize).to(device)

		for i in range(self.batchSize):
			resultTensor[i] = indexList[i]
		return resultTensor

	def getSenIdxList(self,contextList, startIndexList):
		batchSenIdxList = []
		for i in range(self.batchSize):
			contextSenList = nltk.sent_tokenize(contextList[i])
			senCount = len(contextSenList)
			senIdxList=[]
			endIdx = 0
			ansSenIdx = 0
			for j in range(senCount):
				startIdx = endIdx
				endIdx = startIdx+len(contextSenList[j].split(' '))
				senIdxList.append((startIdx,endIdx))
				if(startIdx<=startIndexList[i] and endIdx>startIndexList[i]):
					ansSenIdx = j
			batchSenIdxList.append((senIdxList,ansSenIdx))
		return batchSenIdxList

	def unPack(self):
		time0 = time.time()
		queryList = []
		contextList = []
		contextIdList = []
		startIndexList = []
		endIndexBoundaryList= []

		dataCount=0
		while (dataCount <self.batchSize):
			time1 = time.time()
			dataIdx = self.dataIndex%self.dataNum
			answerBeginIndex = self.dataPack[dataIdx][2][0]
			answerEndIndexBoundary = self.dataPack[dataIdx][2][1]
			contextId = self.dataPack[dataIdx][1]
			context = self.contextDic[contextId]
			query = self.dataPack[dataIdx][0]
			
			# whitespace processing
			#context = ' '.join(context.split())
			#query = ' '.join(query.split())

			contextTokenList = context.split(' ')
			contextTokenLen = len(contextTokenList)

			# filter wrong query
			if(self.queryFilter=='true'):
				if(query[-1] != '?'):
					self.dataIndex+=1
					continue
			time2 = time.time()

			# out of bound
			if(answerEndIndexBoundary > contextTokenLen or answerEndIndexBoundary > self.maxTokenLen):
				self.dataIndex+=1
				continue
			# context size limit
			elif(contextTokenLen > self.maxTokenLen):
				context=""
				for j in range(self.maxTokenLen):
					if(j==0):
						context = contextTokenList[j]
					else:
						context += ' '+contextTokenList[j]
				contextTokenList = context.split(' ')
				contextTokenLen = len(contextTokenList)
			time3 = time.time()

			# check nltk token sync with original token
			# 1031 tokenize changed by preprocessing
#			contextSenList = nltk.sent_tokenize(context)
#			senTokenCount =0
#			for j in range(len(contextSenList)):
#				senTokenCount += len(contextSenList[j].split(' '))
#			if(senTokenCount != contextTokenLen):
#				self.dataIndex+=1
#				continue
			time4 = time.time()

			queryList.append(query)
			contextIdList.append(contextId)
			contextList.append(context)
			startIndexList.append(answerBeginIndex)
			endIndexBoundaryList.append(answerEndIndexBoundary)
			time5 = time.time()

			#print(time5-time4)
			#print(time4-time3)
			#print(time3-time2)
			#print(time2-time1)
			#sys.exit()

			self.dataIndex +=1
			dataCount+=1
		time6=time.time()
		#print(time6-time0)
		#sys.exit()
		return queryList, contextList, contextIdList, startIndexList, endIndexBoundaryList

	def getStoryDic(self):
		return self.contextDic

	def getDataLen(self):
		return self.dataNum
