import numpy as np
from apprentice import RationalApproximationSIP
from sklearn.model_selection import KFold
from apprentice import tools

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
	# for pdeg in range(4,5):
		ppenaltybin = np.zeros(pdeg+1)
		for qdeg in range(2,5):
		# for qdeg in range(4,5):
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
						"mpseIndex":currIndex, "mpsel":currl}

			outJSON["p%s_q%s"%(str(pdeg),str(qdeg))] = rappsip

			if(debug == 1):
				import json
				with open("/tmp/cv_latest.json", "w") as f:
					json.dump(outJSON, f,indent=4, sort_keys=True)
			# exit(1)

	import json
	with open(outfile, "w") as f:
		json.dump(outJSON, f,indent=4, sort_keys=True)


outfile = "f8_noise_0.5_out.299445.json"
infilePath = "../f8_noise_0.5.txt"
box = np.array([[-1,1],[-1,1]])
debug = 1
runCrossValidation(infilePath,box,outfile,debug)
