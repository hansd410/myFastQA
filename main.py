# python .py data epochNum LR optimizer(0,1) modelName
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# LIB
import sys
sys.path.insert(0,'lib')
from model import Net
from readData import DataList
from readData import normalize_answer
from readData import listToString

#from readData import DataList
from sequences import Seq

# etc
import re
import os
import time
import math
import nltk
import argparse
from collections import Counter
import string
import json


def get_args():
	parser = argparse.ArgumentParser(description="fastQA parameter")

	# parameter
	parser.add_argument('--batch',default=64,type=int,help="batch size")
	parser.add_argument('--embed',default=300,type=int,help="embed dim")
	parser.add_argument('--hidden',default=300,type=int,help="hidden dim")
	parser.add_argument('--layerNum',default=1,type=int,help="layer num")
	parser.add_argument('--epoch',default=10,type=int,help="epoch num")
	parser.add_argument('--lr',default=0.001,type=float,help="learning rate")

	# network option
	parser.add_argument('--sameLSTM',default='true',help="lstm difference")
	parser.add_argument('--embedFix',default='true',help="embed fix or not")
	parser.add_argument('--senSel',default='false',help="apply sentence selection")
	parser.add_argument('--wiq',default='true',help="wiq on off")

	# running option
	parser.add_argument('--mode',default='train',help="train or test")
	parser.add_argument('--modelFolder',default='/mnt/data/hansd410/savedModel/fastQA/',help="modelFolder")
	parser.add_argument('--loadModelDir',default='',help="model to load")
	parser.add_argument('--loadModelFile',default='',help="model to load")
	parser.add_argument('--gpu',default='0',help="gpu id")
	parser.add_argument('--maxLen',default=400,type=int,help="max token length of context")
	parser.add_argument('--testMaxLen',default=1000,type=int,help="max token length of context for test")
	parser.add_argument('--queryFilter',default='false',help="filtering query without ?")


	# etc
	parser.add_argument('--beam',default=5,help="beam size")
	parser.add_argument('--debug',default='false',help="debug mode")
	parser.add_argument('--tag',default='',help="tag for output file")
	parser.add_argument('--trainFile',default='data/SQuAD/train.csv',help="train file")
	parser.add_argument('--devFile',default='data/SQuAD/dev.csv',help="dev file")

	return parser.parse_args()

def run(args):
	# PARAMETER
	learningRate = args.lr

	device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
	print(device)
	print("network initializing")
	net = Net(args) 
	net.to(device)

	# check what parameters are initialized
#	for name,param in net.named_parameters():
#		if param.requires_grad:
#			print (name)

	if(args.loadModelDir != ''):
		modelLoadDir = args.modelFolder+args.loadModelDir
		# get the last model file
		if(args.loadModelFile == ''): 
			modelFileList = os.listdir(modelLoadDir)
			net.load_state_dict(torch.load(modelLoadDir+"/"+modelFileList[-1]))
		else:
			net.load_state_dict(torch.load(modelLoadDir+"/"+args.loadModelFile))

	if(args.mode == 'train' and args.debug == 'false'):
		if(args.loadModelDir != ''):
			modelSaveDir = args.modelFolder+args.loadModelDir+"_modelLoad"
		else:
			timestr = time.strftime("%m%d%H%M%S")
			modelSaveDir = args.modelFolder+timestr+"_maxlen_"+str(args.maxLen)+"_testmaxlen_"+str(args.testMaxLen)+"_queryFilter_"+str(args.queryFilter)+"_sensel_"+str(args.senSel)+"_epoch_"+str(args.epoch)+"_b_"+str(args.batch)+"_sameLSTM_"+str(args.sameLSTM)+"_wiq_"+str(args.wiq)+"_embedFix_"+str(args.embedFix)
			if(args.tag != ''):
				modelSaveDir = modelSaveDir+"_"+str(args.tag)
		fScoreLog = open(modelSaveDir+"_score",'w')
		try:
			os.stat(modelSaveDir)
		except:
			os.mkdir(modelSaveDir)


	print("network initialized")

	print("Data reading")
	if(args.mode =='train'):
		trainDataList = DataList(args.trainFile,args.batch,args.maxLen,args.queryFilter)
	devDataList = DataList(args.devFile,args.batch,args.testMaxLen,args.queryFilter)
	print("Data read")

	# test mode
	if(args.mode !='train'):
		if(args.mode == 'debug'):
			testDebug(args,net,devDataList,device)

		else:
			f1,em,outputDict = testCandidate(args,net,devDataList,device)
			fOutputDict = open("f1_"+str(f1)+"_em_"+str(em)+'_outputDict.json','w')
			fOutputDict.write(json.dumps(outputDict))
			
	# train mode
	else:
		if(args.debug == 'false'):
			# save initialized network
			modelName = "b_"+str(args.batch)+"_e_"+str(args.embed)+"_h_"+str(args.hidden)+"_0"
			torch.save(net.state_dict(),modelSaveDir+"/"+modelName)

		# define optimizer
		optimizer = optim.Adam(filter(lambda p : p.requires_grad, net.parameters()),lr = learningRate)

		# define loss function
		criterion=nn.CrossEntropyLoss()

		# score backup
		f1_before = 0
		f1_max = 0
		em_f1_max = 0
		for i in range(args.epoch*trainDataList.getDataLen()//args.batch+1):
			time1 = time.time()
			if(i!=0):
				loss_before = loss
				senLoss_before = senLoss
			else:
				loss_before = 0
				senLoss_before = 0

			# get batch data
			queryList, contextList, contextIdList, startIndexList, endIndexBoundaryList, senIdxList, targetAnswerTensor, targetSenTensor = trainDataList.getBatchData(device)
			time2 = time.time()
			# train
			if(args.senSel=='true'):
				answerTensor, answerSentenceTensor  = net(queryList,contextList,startIndexList,senIdxList)
			else:
				answerTensor = net(queryList,contextList,startIndexList)
			time3 = time.time()



			# loss calculation
			loss=0
			senLoss=0
			# since begin and end are separated, tensor size is 2*batch size
			loss += criterion(answerTensor,targetAnswerTensor)*2
			if(args.senSel=='true'):
				loss += criterion(answerSentenceTensor,targetSenTensor)
				senLoss = criterion(answerSentenceTensor,targetSenTensor)
			time4 = time.time()

			# backprop
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			torch.cuda.empty_cache()	# not sure about it
			time5 = time.time()

			# print log for 100 trials
			if(i%100==99):
				print("iter "+str(i))
				print(modelSaveDir)
				print("loss from:")
				print(loss)
				print("loss diff:")
				print(loss_before-loss)
				print("sen loss diff:")
				print(senLoss_before-senLoss)
		
			# for 1000 trials, test by dev file
			if(i%1000==999):
				modelName = "b_"+str(args.batch)+"_e_"+str(args.embed)+"_h_"+str(args.hidden)+"_"+str(i)
				# test
				f1,em, _ = test(args,net,devDataList,device)
				print("f1 from "+str(f1_before)+" to "+str(f1))
				if(f1_max<f1):
					f1_max = f1
					em_f1_max = em
				# if f1 goes down, update learning rate and optimizer
				if(f1<f1_before):
					print("learning rate from "+str(learningRate) + " to "+str(learningRate/2))
					learningRate = learningRate/2
					optimizer = optim.Adam(filter(lambda p : p.requires_grad, net.parameters()),lr = learningRate)

				# print score log to file
				f1_before=f1
				fScoreLog.write("model:"+modelName+"\t"+"f1:"+str(f1)+"\tem:"+str(em)+"\tlr"+str(learningRate)+"\n")
				fScoreLogTemp = open(modelSaveDir+"/model:"+modelName+"_"+"f1:"+str(f1)+"_em:"+str(em)+"_lr"+str(learningRate),'w')
				fScoreLogTemp.write(" ")


				torch.save(net.state_dict(),modelSaveDir+"/"+modelName)
		fScoreLog.write("maxF1:"+str(f1_max)+"\tem:"+str(em_f1_max)+"\n")


def test(args,net,devDataList,device):
	resultDict = {}
	for j in range(devDataList.getDataLen()//args.batch+1):
		queryList, contextList, contextIdList,startIndexList, endIndexBoundaryList, senIdxList, targetAnswerTensor, targetSenTensor = devDataList.getBatchData(device)
		# test
		answerTensorList = net(queryList,contextList)
		answerTensor = answerTensorList[0]
		#answerTensor = net(queryList,contextList)
		resultDict = batchMeasureDict(queryList, contextList, contextIdList, startIndexList,endIndexBoundaryList, answerTensor,resultDict)

		torch.cuda.empty_cache()
	f1 =0
	em =0
	outputDict = {}
	dataSize = len(resultDict)
	for key, value in resultDict.items():
		f1 += value[0][0]
		em += value[0][1]
		# output, confScore (f1,em,answer,output,normalizedAnswer,normalizedOutput,confScore)
		#outputDict[key] = (value[0][3],value[0][6])
		outputDict[key] = (value[0][0],value[0][1],value[0][2],value[0][3],value[0][4],value[0][5])
	f1 = f1/dataSize
	em = em/dataSize
	print("f1 : "+str(f1)+", em : "+str(em)+", data : "+str(dataSize))
	return f1,em,outputDict

def testCandidate(args,net,devDataList,device):
	resultDict = {}
	for j in range(devDataList.getDataLen()//args.batch+1):
		queryList, contextList, contextIdList,startIndexList, endIndexBoundaryList, senIdxList, targetAnswerTensor, targetSenTensor = devDataList.getBatchData(device)
		# test
		answerTensorList = net(queryList,contextList)
		resultDict = batchCandidateMeasureDict(queryList, contextList, contextIdList, startIndexList,endIndexBoundaryList, answerTensorList,resultDict)

		torch.cuda.empty_cache()
	f1 =0
	em =0
	outputDict = {}
	dataSize = len(resultDict)
	for key, value in resultDict.items():
		f1 += value[0][0]
		em += value[0][1]
		# (f1,em,answer,output,normalizedAnswer,normalizedOutput,confScore,answerCandidatePairList)
		answerPairFilteredList = []
		for testStartIdx,testEndBoundaryIdx,output,confScore in value[0][7]:
			answerPairFilteredList.append((output,confScore))
		outputDict[key] = (value[0][3],value[0][6],answerPairFilteredList)
	f1 = f1/dataSize
	em = em/dataSize
	print("f1 : "+str(f1)+", em : "+str(em)+", data : "+str(dataSize))
	return f1,em,outputDict

def testDebug(args,net,devDataList,device):
	resultDict = {}
	for j in range(devDataList.getDataLen()//args.batch+1):
		queryList, contextList, contextIdList,startIndexList, endIndexBoundaryList, senIdxList, targetAnswerTensor, targetSenTensor = devDataList.getBatchData(device)
		# test
		answerTensorList = net(queryList,contextList)
		answerTensor = answerTensorList[0]
		resultDict = batchMeasureDict(queryList, contextList, contextIdList, startIndexList,endIndexBoundaryList, answerTensor,resultDict)

		torch.cuda.empty_cache()
	f1 =0
	em =0
	outputDict = {}
	dataSize = len(resultDict)
	for key, value in resultDict.items():
		f1 += value[0][0]
		em += value[0][1]
		# output, confScore, answerCandidate
		outputDict[key] = (value[0][2],value[0][3])
	f1 = f1/dataSize
	em = em/dataSize
	print("f1 : "+str(f1)+", em : "+str(em)+", data : "+str(dataSize))
	return f1,em,outputDict



#def listToString(inputList):
#	resultString  = ""
#	for i in range(len(inputList)):
#		resultString+=inputList[i]
#		resultString+=" "
#	return resultString

def batchMeasureDict(queryList, contextList, contextIdList, startIndexList, endIndexBoundaryList, answerTensor, resultDict):
	batchSize = len(contextList)
	for i in range(batchSize):
		contextLen = len(contextList[i].split(' '))
		beginTensor =answerTensor[i]
		endTensor = answerTensor[i+batchSize]
		_,testStartIdx = torch.max(beginTensor[:contextLen],0)
		_,testEndIdx =  torch.max(endTensor[testStartIdx:contextLen],0)
		testEndBoundaryIdx = testStartIdx + testEndIdx + 1

		softmax = nn.Softmax(dim=0)
		softBeginTensor = softmax(beginTensor)
		softEndTensor = softmax(endTensor)
		confScore = (softBeginTensor[testStartIdx]*softEndTensor[testEndBoundaryIdx-1]).data.cpu().numpy()

		(f1,em,answer,output,normalizedAnswer,normalizedOutput) = measure(contextList[i], startIndexList[i],endIndexBoundaryList[i],testStartIdx, testEndBoundaryIdx)
		dictKey = contextIdList[i]+"\t"+queryList[i]
		# put the highest score result in resultDict
		if((dictKey in resultDict.keys() and resultDict[dictKey][0][0]<f1) or (dictKey not in resultDict.keys())):
			resultDict[dictKey]=((f1,em,answer,output,normalizedAnswer,normalizedOutput,confScore),(contextList[i],queryList[i],startIndexList[i],endIndexBoundaryList[i],testStartIdx, testEndBoundaryIdx))
	return resultDict

def batchCandidateMeasureDict(queryList, contextList, contextIdList, startIndexList, endIndexBoundaryList, answerTensorList, resultDict):
	batchSize = len(contextList)
	answerCandidatePairList = getAnswerCandidatePair(answerTensorList,contextList)
	for i in range(batchSize):
		testStartIdx,testEndBoundaryIdx,output,confScore = answerCandidatePairList[i][0]
		(f1,em,answer,output,normalizedAnswer,normalizedOutput) = measure(contextList[i], startIndexList[i],endIndexBoundaryList[i],testStartIdx, testEndBoundaryIdx)
		dictKey = contextIdList[i]+"\t"+queryList[i]
		# put the highest score result in resultDict
		if((dictKey in resultDict.keys() and resultDict[dictKey][0][0]<f1) or (dictKey not in resultDict.keys())):
			resultDict[dictKey]=((f1,em,answer,output,normalizedAnswer,normalizedOutput,confScore,answerCandidatePairList[i]),(contextList[i],queryList[i],startIndexList[i],endIndexBoundaryList[i]))
	return resultDict

def getAnswerCandidatePair(answerTensorList,contextList):
	answerCandidatePairList = []
	# answer tensor list is given in beam size
	beamSize = len(answerTensorList)
	batchSize = len(contextList)
	for j in range(batchSize):
		batchCandidatePairList = []
		for i in range(beamSize):
			contextToken = contextList[j].split(' ')
			contextLen = len(contextToken)
			beginTensor = answerTensorList[i][j]
			endTensor = answerTensorList[i][j+batchSize]
			#_,testStartIdx = torch.max(beginTensor[:contextLen],0)
			testStartIdx = torch.topk(beginTensor[:contextLen],i+1,0)[1][-1]
			for k in range(contextLen-testStartIdx):
				if(k>19):
					continue
				#_,testEndIdx =  torch.max(endTensor[testStartIdx:contextLen],0)
				testEndIdx =  torch.topk(endTensor[testStartIdx:contextLen],k+1,0)[1][-1]
				testEndBoundaryIdx = testStartIdx+testEndIdx+1

				softmax = nn.Softmax(dim=0)
				softBeginTensor = softmax(beginTensor)
				softEndTensor = softmax(endTensor)
				confScore = (softBeginTensor[testStartIdx]*softEndTensor[testEndBoundaryIdx-1]).data.cpu().numpy().item()

				outputToken = contextToken[testStartIdx:testEndBoundaryIdx]
				output = listToString(outputToken)

				batchCandidatePairList.append((testStartIdx,testEndBoundaryIdx,output,confScore))
		batchCandidatePairList.sort(reverse=True,key=lambda x:x[3])
		answerCandidatePairList.append(batchCandidatePairList[:21])
	return answerCandidatePairList
			
#def normalize_answer(s):
#	def remove_articles(text):
#		return re.sub(r'\b(a|an|the)\b',' ',text)
#	def white_space_fix(text):
#		return ' '.join(text.split())
#	def remove_punc(text):
#		exclude = set(string.punctuation)
#		return ''.join(ch for ch in text if ch not in exclude)
#	def lower(text):
#		return text.lower()
#
#	return white_space_fix(remove_articles(remove_punc(lower(s))))

def measure(context,startIndex,endBoundaryIndex,testStartIndex,testEndBoundaryIndex):
	contextToken = context.split(' ')
	answerToken = contextToken[startIndex:endBoundaryIndex]
	outputToken = contextToken[testStartIndex:testEndBoundaryIndex]
	answer = listToString(answerToken)
	output = listToString(outputToken)

	normalizedAnswerToken = normalize_answer(answer).split(' ')
	normalizedOutputToken = normalize_answer(output).split(' ')
	normalizedAnswer = listToString(normalizedAnswerToken)
	normalizedOutput = listToString(normalizedOutputToken)
#	print("----------- Answer and output")
#	print(answer)
#	print(output)
#	print("----------- Normalized")
#	print(normalizedAnswer)
#	print(normalizedOutput)
	
	common = Counter(normalizedAnswerToken) & Counter(normalizedOutputToken)
	num_same = sum(common.values())

	if(num_same==0):
		return (0,0,answer,output,normalizedAnswer,normalizedOutput)

	precision = 1.0 * num_same / len(normalizedOutputToken)
	recall = 1.0 * num_same / len(normalizedAnswerToken)
	f1 = (2*precision*recall) / (precision+recall)

	if(normalizedAnswer == normalizedOutput):
		em = 1
	else:
		em = 0
	return (f1, em,answer,output,normalizedAnswer,normalizedOutput)


if __name__ == "__main__":
	args = get_args()
	if(args.mode =='train'):
		print("train begins")
	else:
		print("test begins")
	if(args.senSel=='false'):
		print("without sensel")
	run(args)

