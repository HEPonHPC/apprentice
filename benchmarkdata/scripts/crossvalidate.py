import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools, readData

def runCrossValidation(infile,box=np.array([[-1,1],[-1,1]]),outfile="out.json",debug=0):
	trainingScale = "Cp"

	X, Y = tools.readData(infile)

	# Some param overrides for debug
	larr = np.array([10**i for i in range(3,-13,-1)])
	# larr = np.array([10**i for i in range(0,-5,-1)])

	k=10
	# k=2

	outJSON = {}
	for pdeg in range(2,5):
	# for pdeg in range(3,5):
		ppenaltybin = np.zeros(pdeg+1)
		for qdeg in range(2,5):
		# for qdeg in range(3,5):
			qpenaltybin = np.zeros(qdeg+1)
			avgerror = np.zeros(len(larr))
			avgerror_k = np.zeros(len(larr))
			for index in range(len(larr)):
				l = larr[index]
				kfold = KFold(k)
				error_l = 0
				for train, test in kfold.split(X):
					rappsip = RationalApproximationSIP(
												X[train],
												Y[train],
				                                m=pdeg,
				                                n=qdeg,
				                                trainingscale=trainingScale,
				                                box=box,
				                                strategy=2,
				                                penaltyparam=l,
				                                ppenaltybin=ppenaltybin.tolist(),
				                                qpenaltybin=qpenaltybin.tolist()
				    )
					error_l_k = np.sum([(rappsip(X[test])-Y[test])**2])
					error_l += error_l_k
				avgerror[index] = error_l / len(X)
				avgerror_k[index] = error_l / len(X[test])
			stderror = np.std(avgerror_k)/np.sqrt(k)
			minIndex = np.argmin(avgerror)
			minv = avgerror[minIndex]
			minl = larr[minIndex]

			currIndex = minIndex
			while currIndex >= 0:
				if(minv + stderror == avgerror[currIndex]):
					break
				elif(minv + stderror > avgerror[currIndex]):
					currIndex -= 1
				else:
					currIndex += 1
					break
			if(currIndex == -1):
				currIndex = 0
			currl = larr[currIndex]

			# print(minIndex)
			# print(currIndex)
			#
			# print(avgerror)
			# print(avgerror_k)
			# print(stderror)
			#
			# print(minl)
			# print(currl)

			rappsip_min = RationalApproximationSIP(
										X,
										Y,
										m=pdeg,
										n=qdeg,
										trainingscale=trainingScale,
										box=box,
										strategy=2,
										penaltyparam=minl,
										ppenaltybin=ppenaltybin.tolist(),
										qpenaltybin=qpenaltybin.tolist()
			)
			rappsip_minpse = rappsip_min
			if(currl != minl):
				rappsip_minpse = RationalApproximationSIP(
											X,
											Y,
											m=pdeg,
											n=qdeg,
											trainingscale=trainingScale,
											box=box,
											strategy=2,
											penaltyparam=currl,
											ppenaltybin=ppenaltybin.tolist(),
											qpenaltybin=qpenaltybin.tolist()
			)
			rappsip = {"min":rappsip_min.asDict, "min plus SE":rappsip_minpse.asDict, "avgerror":avgerror.tolist(),
						"avgerror_k":avgerror_k.tolist(), "stderror":stderror,"minIndex":minIndex,"minl":minl,
						"minv":minv, "mpseIndex":currIndex, "mpsel":currl}

			outJSON["p%s_q%s"%(str(pdeg),str(qdeg))] = rappsip

			if(debug == 1):
				import json
				with open("/tmp/cv_latest.json", "w") as f:
					json.dump(outJSON, f,indent=4, sort_keys=True)
			# exit(1)

	import json
	with open(outfile, "w") as f:
		json.dump(outJSON, f,indent=4, sort_keys=True)

def prettyPrint(jsonfile, testfile):
	import json
	if jsonfile:
		with open(jsonfile, 'r') as fn:
			datastore = json.load(fn)


	keylist = datastore.keys()
	keylist.sort()
	s=""
	# s = "pq deg\tcparam\tparam\tl2term\t\tl1term\n\n"
	# for key in keylist:
	#
	# 	iterationInfo = datastore[key]['min']["iterationinfo"]
	# 	lsqsplit = iterationInfo[len(iterationInfo)-1]["leastSqSplit"]
	# 	s += "%s\tmin\t%.0E\t%f\t%f\n"%(key,datastore[key]['minl'],lsqsplit['l2term'],lsqsplit['l1term'])
	#
	# 	iterationInfo = datastore[key]['min plus SE']["iterationinfo"]
	# 	lsqsplit = iterationInfo[len(iterationInfo)-1]["leastSqSplit"]
	# 	s += "%s\tmpse\t%.0E\t%f\t%f\n"%(key,datastore[key]['mpsel'],lsqsplit['l2term'],lsqsplit['l1term'])
	# 	s+="\naverage error\n"
	#
	#
	# 	avgerror =  datastore[key]['avgerror']
	# 	larr = np.array([10**i for i in range(3,-13,-1)])
	# 	# for i in larr:
	# 	# 	s += "%.1E\t"%(i)
	# 	s+="\n"
	# 	for i in avgerror:
	# 		s += "%.4E\t"%(i)
	# 	s+="\n\n"
	#

	# static for f8 and upto p4 and q4
	s+= "p coeffs obtained for lambda with minimum avg CV error\n"
	s+="origfn\t"
	for key in keylist:
		s+="%s\t\t"%(key)
	s+="\n"
	# P 0,1,2,3,5
	pcoeffO = [-1,-1,1,1,0,1]
	for i in range(15):
		if(i <= 3 or i == 5):
			s+= "%d\t"%(pcoeffO[i])
		else:
			s+="\t"
		for key in keylist:
			pcoeff = datastore[key]['min']["pcoeff"]
			if(i<len(pcoeff)):
				s+="%f\t"%(pcoeff[i])
			else: s+="\t\t"
		s+="\n"

	s+= "q coeffs obtained for lambda with minimum avg CV error\n"
	s+="origfn\t"
	for key in keylist:
		s+="%s\t\t"%(key)
	s+="\n"
	qcoeffO = [1.21,-1.1,-1.1,0,1,0]
	# Q 0,1,2,3,5
	for i in range(15):
		if(i <= 2 or i == 4):
			s+= "%.2f\t"%(qcoeffO[i])
		else:
			s+="\t"
		for key in keylist:
			qcoeff = datastore[key]['min']["qcoeff"]
			if(i < len(qcoeff)):
				s+="%f\t"%(qcoeff[i])
			else: s+="\t\t"
		s+="\n"
	s+="\n"

	X_test, Y_test = readData(testfile)

	for key in keylist:
		s+="\t%s\t"%(key)
	s+="\n"

	testerrarr = np.array([])
	s+= "testErr\t"
	for key in keylist:
		rappsip = RationalApproximationSIP(datastore[key]['min'])
		Y_pred = abs(rappsip(X_test))
		# print(np.c_[Y_pred,Y_test,abs(Y_pred-Y_test)])
		error = np.average(abs(Y_pred-Y_test))
		testerrarr = np.append(testerrarr,error)
		s += "%.8f\t"%(error)
	s+="\n"
	trainerrarr = np.array([])
	s+= "l2term\t"
	for key in keylist:
		iterationInfo = datastore[key]['min']["iterationinfo"]
		lsqsplit = iterationInfo[len(iterationInfo)-1]["leastSqSplit"]
		trainerrarr = np.append(trainerrarr,lsqsplit['l2term'])
		s += "%f\t"%(lsqsplit['l2term'])
	s+="\n"
	s+= "l1term\t"
	for key in keylist:
		iterationInfo = datastore[key]['min']["iterationinfo"]
		lsqsplit = iterationInfo[len(iterationInfo)-1]["leastSqSplit"]
		s += "%f\t"%(lsqsplit['l1term'])
	s+="\n"
	s+= "param\t"
	for key in keylist:
		s += "%.E\t\t"%(datastore[key]['minl'])
	s+="\n\n"

	s+="Min testing error was at %s with value %f.\n"%(keylist[np.argmin(testerrarr)],np.min(testerrarr))
	s+="Min training error was at %s with value %f.\n"%(keylist[np.argmin(trainerrarr)],np.min(trainerrarr))

	print(s)



infilePath = "../f8_noisepct10-3.txt"
outfile = "f8_noisepct10-3_out.299445.json"
testfile = "../f8_test.txt"
box = np.array([[-1,1],[-1,1]])
debug = 1

# runCrossValidation(infilePath,box,outfile,debug)

prettyPrint(outfile,testfile)





#end
