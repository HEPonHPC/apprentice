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

def runRappsipBaseStrategy(infile,runs, box=np.array([[-1,1],[-1,1]]),trainingScale="1x", roboptstrategy="ms",outfile="out.json",debug=0,debugfile="/tmp/s0_latest.json"):
	X, Y = tools.readData(infile)
	return runRappsipBaseStrategyFromPoints(X,Y,runs, box,trainingScale, roboptstrategy,outfile,debug,debugfile)

def runRappsipBaseStrategyFromPoints(X, Y,runs, box=np.array([[-1,1],[-1,1]]),trainingScale="1x", roboptstrategy="ms",outfile="out.json",debug=0,debugfile="/tmp/s0_latest.json"):
	outJSON = {}
	for r in runs:
		pdeg=r[0]
		qdeg=r[1]
		rappsip = RationalApproximationSIP(
										X,
										Y,
										m=pdeg,
										n=qdeg,
										fitstrategy = 'filter',
										localoptsolver = 'scipy',
										trainingscale=trainingScale,
										roboptstrategy=roboptstrategy,
										scalemin = box[:,0],
										scalemax = box[:,1],
										strategy=0
		)
		outJSON["p%s_q%s"%(str(pdeg),str(qdeg))] = rappsip.asDict
		if(debug == 1):
			import json
			with open("/tmp/s0_latest.json", "w") as f:
				json.dump(outJSON, f,indent=4, sort_keys=True)

	import json
	with open(outfile, "w") as f:
		json.dump(outJSON, f,indent=4, sort_keys=True)

def runRappsipStrategy2(infile,runs, larr,l1strat="ho_p_q",box=np.array([[-1,1],[-1,1]]),trainingScale="0.5x",outfile="out.json",debug=0):

# l1strat="ho_p_q"
# l1strat="all_p_q"


	X, Y = tools.readData(infile)
	outJSON = {}

	# runs = [[2,2],[3,?3],[4,4],[5,5],[6,6]]
	for r in runs:
		for l in larr:
			pdeg=r[0]
			qdeg=r[1]
			if(l1strat == "ho_p_q"):
				ppenaltybin = np.ones(pdeg+1)
				ppenaltybin[pdeg] = 0

				qpenaltybin = np.ones(qdeg+1)
				qpenaltybin[qdeg] = 0
			elif(l1strat == "all_p_q"):
				ppenaltybin = np.zeros(pdeg+1)
				qpenaltybin = np.zeros(qdeg+1)


			rappsip = RationalApproximationSIP(
											X,
											Y,
											m=pdeg,
											n=qdeg,
											trainingscale=trainingScale,
											roboptstrategy="baron",
											box=box,
											strategy=2,
											penaltyparam=l,
				                            ppenaltybin=ppenaltybin.tolist(),
				                            qpenaltybin=qpenaltybin.tolist()

			)
			outJSON["p%s_q%s_%.E"%(str(pdeg),str(qdeg),l)] = rappsip.asDict
			if(debug == 1):
				import json
				with open("/tmp/s2_latest.json", "w") as f:
					json.dump(outJSON, f,indent=4, sort_keys=True)

	import json
	with open(outfile, "w") as f:
		json.dump(outJSON, f,indent=4, sort_keys=True)


def tableS0(jsonfile, testfile, runs):
	import json
	if jsonfile:
		with open(jsonfile, 'r') as fn:
			datastore = json.load(fn)

	X_test, Y_test = readData(testfile)
	karr = np.array([])
	aic = np.array([])
	bic = np.array([])
	X_l2 = np.array([])
	Z_testerr = np.array([])
	mn = np.array([])

	for r in runs:
		pdeg=r[0]
		qdeg=r[1]
		key = "p%s_q%s"%(str(pdeg),str(qdeg))
		iterationInfo = datastore[key]["iterationinfo"]
		lastii = iterationInfo[len(iterationInfo)-1]
		trainerr = lastii["leastSqObj"]
		X_l2 = np.append(X_l2,trainerr)

		rappsip = RationalApproximationSIP(datastore[key])
		Y_pred = rappsip(X_test)
		testerror = np.sum((Y_pred-Y_test)**2)
		Z_testerr = np.append(Z_testerr,testerror)

		k = 2
		pcoeff = datastore[key]["pcoeff"]
		qcoeff = datastore[key]["qcoeff"]
		maxp = abs(max(pcoeff, key=abs))
		maxq = abs(max(qcoeff, key=abs))
		# print(np.c_[pcoeff])
		# print(np.c_[qcoeff])
		# print(maxp,maxq)
		for pc in pcoeff:
			if(pc > 10**-2*maxp):
				k += 1
		for qc in qcoeff:
			if(qc > 10**-2*maxq):
				k += 1
		karr = np.append(karr,k)
		n = len(X_test)
		# AIC = 2k - 2log(L)
		# BIC = klog(n) - 2log(L)
		# -2log(L) becomes nlog(variance) = nlog(SSE/n) = nlog(testerror/n)
		a = 2*k + n*np.log(testerror/n)
		b = k*np.log(n) + n*np.log(testerror/n)

		aic = np.append(aic,a)
		bic = np.append(bic,b)
		mn = np.append(mn,rappsip.M+rappsip.N)

	sortedmnindex = np.argsort(mn)
	print("#\tpq\tl2 error\ttest err\tM+N\tnnz\taic\t\tbic")
	for i in sortedmnindex:
		r = runs[i]
		pdeg=r[0]
		qdeg=r[1]
		print("%d\tp%dq%d\t%f\t%f\t%d\t%d\t%f\t%f"%(i+1,pdeg,qdeg,X_l2[i],Z_testerr[i],mn[i],karr[i],aic[i],bic[i]))

	print("\nMIN\t\t%d\t\t%d\t\t%d\t%d\t\t%d\t\t%d\n"%(np.argmin(X_l2)+1,np.argmin(Z_testerr)+1,np.argmin(mn)+1,np.argmin(karr)+1,np.argmin(aic)+1,np.argmin(bic)+1))


def plotmntesterrperfile(jsonfile,testfile, desc,folder):
	minp = np.inf
	minq = np.inf
	maxp = 0
	maxq = 0
	miny0 = 0

	minl1 = np.inf
	minl2 = np.inf
	minlinf = np.inf
	pp = 0
	qq =0

	outfile1 = folder+"/"+desc+".299445.png"
	# outfile2 = folder+"/"+fno+"_index.299445.png"
	X_test, Y_test = readData(testfile)
	import json
	if jsonfile:
		with open(jsonfile, 'r') as fn:
			datastore = json.load(fn)
	from apprentice import monomial
	import apprentice

	for key in sorted(datastore.keys()):
		pdeg = datastore[key]['m']
		qdeg = datastore[key]['n']
		if(pdeg<minp):
			minp=pdeg
		if(qdeg<minq):
			minq=qdeg
		if pdeg > maxp:
			maxp = pdeg
		if qdeg > maxq:
			maxq = qdeg
	# print(minp,maxp,minq,maxq)
	error = np.zeros(shape = (maxp-minp+1,maxq-minq+1))
	for key in sorted(datastore.keys()):
		pdeg = datastore[key]['m']
		qdeg = datastore[key]['n']
		Y_pred = np.array([],dtype=np.float64)
		if('scaler' in datastore[key]):
			rappsip = RationalApproximationSIP(datastore[key])
			Y_pred = rappsip.predictOverArray(X_test)
			# print(Y_pred)
		else:
			structp = apprentice.monomialStructure(datastore[key]['dim'], pdeg)
			structq = apprentice.monomialStructure(datastore[key]['dim'], qdeg)
			for x in X_test:
				nnn = np.array(monomial.recurrence(x, structp))
				p = np.array(datastore[key]['pcoeff']).dot(nnn)
				ddd = np.array(monomial.recurrence(x, structq))
				q = np.array(datastore[key]['qcoeff']).dot(ddd)
				Y_pred = np.append(Y_pred,(p/q))
		# print(np.c_[Y_pred,Y_test])

		l1 = np.sum(np.absolute(Y_pred-Y_test))
		# print(l1)
		l2 = np.sqrt(np.sum((Y_pred-Y_test)**2))
		linf = np.max(np.absolute(Y_pred-Y_test))
		x000 = np.zeros(datastore[key]['dim'])
		y000 = rappsip.predict(x000)
		print("p%d q%d %f"%(pdeg,qdeg,y000))
		error[pdeg-minp][qdeg-minq] = l2
		if(minl2>l2):
			minl2 = l2
			minl1 = l1
			minlinf = linf
			print(linf)
			pp = pdeg
			qq = qdeg
			miny0 = y000
			# print(miiny0)

	import matplotlib as mpl
	import matplotlib.pyplot as plt

	mpl.rc('text', usetex = True)
	mpl.rc('font', family = 'serif', size=12)
	mpl.style.use("ggplot")
	cmapname   = 'viridis'
	plt.clf()
	print(error)

	markersize = 1000
	vmin = -4
	vmax = 2
	X,Y = np.meshgrid(range(minq,maxq+1),range(minp,maxp+1))
	# plt.scatter(X,Y , marker = 's', s=markersize, c = np.ma.log10(error), cmap = cmapname, vmin=vmin, vmax=vmax, alpha = 1)
	plt.scatter(X,Y , marker = 's', s=markersize, c = error, cmap = cmapname, alpha = 1)
	plt.xlabel("$n$")
	plt.ylabel("$m$")
	plt.xlim((minq-1,maxq+1))
	plt.ylim((minp-1,maxp+1))
	b=plt.colorbar()
	# b.set_label("$\log_{10}\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_2$")
	b.set_label("$\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_2$")
	plt.title("l1=%f, l2=%f, linf=%f y0=%f found at (%d,%d)"%(minl1,minl2,minlinf,pp,qq,miny0))
	plt.savefig(outfile1)
	# plt.show()


def plotmntesterr(jsonfilearr, jsonfiledescrarr, testfile, runs, fno,folder):
	# LT, RT, LB, RB
	maxpq = np.amax(runs,axis=0)
	outfile1 = folder+"/"+fno+".299445.png"
	outfile2 = folder+"/"+fno+"_index.299445.png"

	X_test, Y_test = readData(testfile)
	testerractuals = {}
	testerrindex = {}
	for i in range(len(jsonfilearr)):
		jsonfile = jsonfilearr[i]
		import json
		if jsonfile:
			with open(jsonfile, 'r') as fn:
				datastore = json.load(fn)
		testerrarr = np.zeros(shape=(maxpq[0],maxpq[1]),dtype=np.float64)
		testerrarr2n = np.zeros(shape=(maxpq[0],maxpq[1]),dtype=np.float64)
		testerrarr1n = np.zeros(shape=(maxpq[0],maxpq[1]),dtype=np.float64)
		testerrarrinfn = np.zeros(shape=(maxpq[0],maxpq[1]),dtype=np.float64)
		for r in runs:
			pdeg=r[0]
			qdeg=r[1]
			key = "p%s_q%s"%(str(pdeg),str(qdeg))
			print(key)
			# iterationInfo = datastore[key]["iterationinfo"]
			# lastii = iterationInfo[len(iterationInfo)-1]

			rappsip = RationalApproximationSIP(datastore[key])
			Y_pred = rappsip(X_test)
			testerror = np.average((Y_pred-Y_test)**2)
			testerrarr[pdeg-1][qdeg-1] = testerror
			testerrarr2n[pdeg-1][qdeg-1] = np.sqrt(np.sum((Y_pred-Y_test)**2))
			testerrarr1n[pdeg-1][qdeg-1] = np.sum(np.absolute((Y_pred-Y_test)))
			testerrarrinfn[pdeg-1][qdeg-1] = np.max(np.absolute((Y_pred-Y_test)))



		testerractuals[i] = testerrarr
		sortedindexarr = np.argsort(-testerrarr,axis=None)[::-1].argsort()
		sortedindexarr = np.reshape(sortedindexarr,(maxpq[0],maxpq[1]))
		testerrindex[i] = sortedindexarr

		# print(testerrarr)
		# print(testerrarr1n)
		# print(testerrarr2n)
		# print(testerrarrinfn)
		# print(sortedindexarr)
		print(np.argmin(testerrarr), np.min(testerrarr),np.min(testerrarr1n),np.min(testerrarr2n),np.min(testerrarrinfn))
		print(np.max(testerrarr))

	import matplotlib as mpl
	import matplotlib.pyplot as plt
	mpl.rc('text', usetex = True)
	mpl.rc('font', family = 'serif', size=12)
	mpl.style.use("ggplot")
	cmapname   = 'viridis'
	# X,Y = np.meshgrid(range(1,maxpq[0]+1), range(1,maxpq[1]+1))
	X,Y = np.meshgrid(range(1,maxpq[1]+1), range(1,maxpq[0]+1))
	f, axarr = plt.subplots(2,2, sharex=True, sharey=True, figsize=(15,15))
	f.suptitle(fno + " -- log(average test error)", fontsize = 28)
	markersize = 1000
	vmin = -6
	vmax = -1

	for i in range(2):
		for j in range(2):
			testerrarr = testerractuals[i*2+j]
			sc = axarr[i][j].scatter(X,Y, marker = 's', s=markersize, c = np.ma.log10(testerrarr), cmap = cmapname, vmin=vmin, vmax=vmax, alpha = 1)
			axarr[i][j].set_title(jsonfiledescrarr[i*2+j], fontsize = 28)

	for ax in axarr.flat:
		ax.set(xlim=(0,maxpq[1]+1),ylim=(0,maxpq[0]+1))
		ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
		ax.tick_params(axis = 'both', which = 'minor', labelsize = 18)
		ax.set_xlabel('$n$', fontsize = 22)
		ax.set_ylabel('$m$', fontsize = 22)
	for ax in axarr.flat:
		ax.label_outer()
	b=f.colorbar(sc,ax=axarr.ravel().tolist(), shrink=0.95)

    # b.set_label("Error = $log_{10}\\left(\\frac{\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_%i}{%i}\\right)$"%(norm,testSize), fontsize = 28)

	plt.savefig(outfile1)

	import matplotlib as mpl
	import matplotlib.pyplot as plt
	mpl.rc('text', usetex = True)
	mpl.rc('font', family = 'serif', size=12)
	mpl.style.use("ggplot")
	cmapname   = 'viridis'
	X,Y = np.meshgrid(range(1,maxpq[1]+1), range(1,maxpq[0]+1))
	f, axarr = plt.subplots(2,2, sharex=True, sharey=True, figsize=(15,15))
	f.suptitle(fno + " -- ordered enumeration of test error", fontsize = 28)
	markersize = 1000
	vmin = 0
	vmax = maxpq[0] * maxpq[1]

	for i in range(2):
		for j in range(2):
			sortedindexarr = testerrindex[i*2+j]
			sc = axarr[i][j].scatter(X,Y, marker = 's', s=markersize, c = sortedindexarr, cmap = cmapname, vmin=vmin, vmax=vmax, alpha = 1)
			axarr[i][j].set_title(jsonfiledescrarr[i*2+j], fontsize = 28)

	for ax in axarr.flat:
		ax.set(xlim=(0,maxpq[1]+1),ylim=(0,maxpq[0]+1))
		ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
		ax.tick_params(axis = 'both', which = 'minor', labelsize = 18)
		ax.set_xlabel('$n$', fontsize = 22)
		ax.set_ylabel('$m$', fontsize = 22)
	for ax in axarr.flat:
		ax.label_outer()
	b=f.colorbar(sc,ax=axarr.ravel().tolist(), shrink=0.95)

    # b.set_label("Error = $log_{10}\\left(\\frac{\\left|\\left|f - \\frac{p^m}{q^n}\\right|\\right|_%i}{%i}\\right)$"%(norm,testSize), fontsize = 28)

	plt.savefig(outfile2)





def tableS2(jsonfile, testfile, runs, larr):
	import json
	# Lcurve
	if jsonfile:
		with open(jsonfile, 'r') as fn:
			datastore = json.load(fn)

	X_test, Y_test = readData(testfile)

	# import matplotlib as mpl
	# import matplotlib.pyplot as plt
	# mpl.rc('text', usetex = True)
	# mpl.rc('font', family = 'serif', size=12)
	# mpl.style.use("ggplot")
	# cmapname   = 'viridis'
	#
	# f, axarr = plt.subplots(4,4, figsize=(15,15))
	# markersize = 1000
	# vmin = -4
	# vmax = 2.5

	mintesterrArr = np.array([])
	minaic = np.array([])
	minbic = np.array([])
	minparam = np.array([])
	minnnz = np.array([])
	minmn = np.array([])


	for r in runs:
		pdeg=r[0]
		qdeg=r[1]
		Y_l1 = np.array([])
		X_l2 = np.array([])
		Z_testerr = np.array([])
		karr = np.array([])
		aic = np.array([])
		bic = np.array([])
		mn = np.array([])
		param = np.array([])
		for l in larr:
			key = "p%s_q%s_%.E"%(str(pdeg),str(qdeg),l)
			iterationInfo = datastore[key]["iterationinfo"]
			lastii = iterationInfo[len(iterationInfo)-1]
			regerr = lastii["leastSqSplit"]["l1term"]
			trainerr = lastii["leastSqSplit"]["l2term"]
			X_l2 = np.append(X_l2,trainerr)
			Y_l1 = np.append(Y_l1,regerr)

			rappsip = RationalApproximationSIP(datastore[key])
			Y_pred = rappsip(X_test)
			testerror = np.sum((Y_pred-Y_test)**2)
			Z_testerr = np.append(Z_testerr,testerror)
			k = 2
			pcoeff = datastore[key]["pcoeff"]
			qcoeff = datastore[key]["qcoeff"]
			maxp = abs(max(pcoeff, key=abs))
			maxq = abs(max(qcoeff, key=abs))
			# print(np.c_[pcoeff])
			# print(np.c_[qcoeff])
			# print(maxp,maxq)
			for pc in pcoeff:
				if(pc > 10**-2*maxp):
					k += 1
			for qc in qcoeff:
				if(qc > 10**-2*maxq):
					k += 1


			karr = np.append(karr,k)
			n = len(X_test)
			# AIC = 2k - 2log(L)
			# BIC = klog(n) - 2log(L)
			# -2log(L) becomes nlog(variance) = nlog(SSE/n) = nlog(testerror/n)
			a = 2*k + n*np.log(testerror/n)
			b = k*np.log(n) + n*np.log(testerror/n)

			aic = np.append(aic,a)
			bic = np.append(bic,b)

			param = np.append(param,l)
			mn = np.append(mn,rappsip.M+rappsip.N)

		print("p = "+str(pdeg)+"; q = "+str(qdeg))
		print("#\tl2 error\tl1 error\ttest err\tnnz\taic\t\tbic")
		for i in range(len(larr)):
			print("%d\t%f\t%f\t%f\t%d\t%f\t%f"%(i+1,X_l2[i],Y_l1[i],Z_testerr[i],karr[i], aic[i],bic[i]))
		print("\nMIN\t%d\t\t%d\t\t%d\t\t%d\t\t%d\t\t%d\n"%(np.argmin(X_l2)+1,np.argmin(Y_l1)+1,np.argmin(Z_testerr)+1,np.argmin(karr)+1,np.argmin(aic)+1,np.argmin(bic)+1))




		# axarr[pdeg-2][qdeg-2].plot(X_l2, Y_l1, '-rD')
		# axarr[pdeg-2][qdeg-2].set_title("p = "+str(pdeg)+"; q = "+str(qdeg))

		# if min arg of aic, bic and test err match, then take that and put int min arrays
		minindexarr = [np.argmin(Z_testerr),np.argmin(aic),np.argmin(bic)]
		if all(x == minindexarr[0] for x in minindexarr):
			mintesterrArr = np.append(mintesterrArr,np.min(Z_testerr))
			minaic = np.append(minaic,np.min(aic))
			minbic = np.append(minbic,np.min(bic))
			minparam = np.append(minparam,param[minindexarr[0]])
			minnnz  = np.append(minnnz,karr[minindexarr[0]])
			minmn = np.append(minmn,mn[minindexarr[0]])
		# 2 elements match
		elif len(set(arr)) == 2:
			# find the 2 mathcing elements and take values from all arrays at that index
			if minindexarr[0]==minindexarr[1] or minindexarr[0]==minindexarr[2]:
				mintesterrArr = np.append(mintesterrArr,Z_testerr[minindexarr[0]])
				minaic = np.append(minaic,aic[minindexarr[0]])
				minbic = np.append(minbic,bic[minindexarr[0]])
				minparam = np.append(minparam,param[minindexarr[0]])
				minnnz  = np.append(minnnz,karr[minindexarr[0]])
				minmn = np.append(minmn,mn[minindexarr[0]])
			elif minindexarr[1]==minindexarr[2]:
				mintesterrArr = np.append(mintesterrArr,Z_testerr[minindexarr[1]])
				minaic = np.append(minaic,aic[minindexarr[1]])
				minbic = np.append(minbic,bic[minindexarr[1]])
				minparam = np.append(minparam,param[minindexarr[1]])
				minnnz  = np.append(minnnz,karr[minindexarr[1]])
				minmn = np.append(minmn,mn[minindexarr[1]])
		# no elements match. Highly unlikely that we will be here
		else:
			#take the case where test arr is minimum
			mintesterrArr = np.append(mintesterrArr,Z_testerr[minindexarr[0]])
			minaic = np.append(minaic,aic[minindexarr[0]])
			minbic = np.append(minbic,bic[minindexarr[0]])
			minparam = np.append(minparam,param[minindexarr[0]])
			minnnz  = np.append(minnnz,karr[minindexarr[0]])
			minmn = np.append(minmn,mn[minindexarr[0]])


	print("#\tpq\ttesterr\t\tM+N\tNNZ\taic\t\tbic\t\tlambda")
	for i in range(len(runs)):
		pdeg = runs[i][0]
		qdeg = runs[i][1]
		print("%d\tp%dq%d\t%f\t%d\t%d\t%f\t%f\t%.2E"%(i+1,pdeg,qdeg,mintesterrArr[i],minmn[i],minnnz[i],minaic[i],minbic[i],minparam[i]))

	print("\n")

	sortedmnindex = np.argsort(minmn)
	print("#\tpq\ttesterr\t\tM+N\tNNZ\taic\t\tbic\t\tlambda")
	for i in sortedmnindex:
		pdeg = runs[i][0]
		qdeg = runs[i][1]
		print("%d\tp%dq%d\t%f\t%d\t%d\t%f\t%f\t%.2E"%(i+1,pdeg,qdeg,mintesterrArr[i],minmn[i],minnnz[i],minaic[i],minbic[i],minparam[i]))

	print("\nMIN\t\t%d\t\t%d\t%d\t%d\t\t%d\n"%(np.argmin(mintesterrArr)+1,np.argmin(minmn)+1,np.argmin(minnnz)+1,np.argmin(minaic)+1,np.argmin(minbic)+1))


	# for ax in axarr.flat:
	# 	# ax.set(xlim=(-6,4))
	# 	if(ax.is_first_col()):
	# 		ax.set_ylabel("L1 error", fontsize = 15)
	# 	if(ax.is_last_row()):
	# 		ax.set_xlabel("L2 error", fontsize = 15)
	# # plt.show()



	# P 0,1,2,3,5

def createTable2structure(format = 'table'):
	if(format == 'table'):
		print("\t\tRational Approx\t\t\t\t\tPolynomial Approx")
		print("#vars\t\tdeg num\t\tdeg denom\tDoF\t\tdeg\tDoF")
		fmt = "%d\t\t%d\t\t%d\t\t%d\t\t%d\t%d"
	elif(format == 'latex'):
		print("Not implemented... Exiting Now!")
		exit(1)

	for dim in range(1,9):
		for m in range(1,9):
			for n in range(1,9):
				M = tools.numCoeffsPoly(dim, m)
				N = tools.numCoeffsPoly(dim, n)
				o = 1
				O = tools.numCoeffsPoly(dim, o)

				while(O < M+N):
					o += 1
					O = tools.numCoeffsPoly(dim, o)
				print(fmt%(dim,m,n,M+N,o,O))

def createTable1(folder, format='table'):
	def calcualteNonLin(dim, n):
		if(n==0):
			return 0
		N = tools.numCoeffsPoly(dim, n)
		return N - (dim+1)

	import glob
	import json
	import re
	filelist = np.array(glob.glob(folder+"/*.json"))
	filelist = np.sort(filelist)
	currentfno = -1

	dim = 0
	nnlmin = np.inf
	nnlmax = 0
	sstime = 0.
	ssfneg = 0.

	mstime = 0.
	msfneg = 0.
	msiter = 0

	sotime = 0.
	sofneg = 0.
	soiter = 0

	batime = 0.
	total = 0.

	threshold = 0

	minconsideredn = 1
	maxconsideredn = 3
	print("N considered in [%d, %d]"%(minconsideredn,maxconsideredn))

	if(format == 'table'):
		# print("\t\t\t\t\t\tSingle Start\t\t\tMulti Start\t\t\tSampling\t\t  Baron")
		# print("Function\tdim\tnnl range\t% False Neg\tAvg Time\t% False Neg\t#iter\tAvg Time\t% False Neg\t#iter\tAvg Time\tAvg Time")
		# fmt = "f%d\t\t%d\t(%d-%d)\t\t%.4f\t\t%.2f\t%.4f\t%.2f\t%.3f\t%.4f\t\t%.2f\t%.4f"
		fmt = "f%d\t\t%d\t(%d-%d)\t\t%.4f\t\t%.2f\t%.4f\t%.2f\t%.4f\t\t%.2f\t%.4f"
	elif(format == 'latex'):
		# print("&Single Start&Multi Start&Sampling&Baron")
		# print("Function&% False Neg&Avg Time&% False Neg&Avg Time&% False Neg&Avg Time")
		# fmt = "\\multicolumn{1}{|c|}{\\ref{fn:f%d}}&\\multicolumn{1}{|c|}{%d}&\\multicolumn{1}{|c|}{(%d-%d)}&%.4f&%.2f&%.4f&%.2f&%.3f&%.4f&%.2f&%.4f\\\\"
		fmt = "\\multicolumn{1}{|c|}{\\ref{fn:f%d}}&\\multicolumn{1}{|c|}{%d}&\\multicolumn{1}{|c|}{(%d-%d)}&%.4f&%.2f&%.4f&%.2f&%.4f&%.2f&%.4f\\\\"

	for file in filelist:
		digits = [float(s) for s in re.findall(r'-?\d+\.?\d*', file)]
		fno = int(digits[0])

		if(fno != currentfno and currentfno !=-1):
			# if(format == 'table'):
			print(fmt%(currentfno,dim,nnlmin,nnlmax,batime/total,(ssfneg/total)*100,sstime/total,(msfneg/total)*100,mstime/total,
														(sofneg/total)*100,sotime/total))
			# elif(format == 'latex'):
				# print(fmt%(currentfno,(ssfneg/total)*100,sstime/total,(msfneg/total)*100,mstime/total,
														# (sofneg/total)*100,sotime/total,batime/total))
			dim = 0
			sstime = 0.
			ssfneg = 0.

			mstime = 0.
			msfneg = 0.
			msiter = 0

			sotime = 0.
			sofneg = 0.
			soiter = 0

			batime = 0.
			total = 0.
			nnlmin = np.inf
			nnlmax = 0

		currentfno = fno
		if file:
			with open(file, 'r') as fn:
				datastore = json.load(fn)
		for key in datastore.keys():
			dim = datastore[key]['dim']
			ii = datastore[key]['iterationinfo']
			m = datastore[key]['m']
			n = datastore[key]['n']
			if((n < minconsideredn or n > maxconsideredn)):
				# print(m,n)
				continue
			nnl = calcualteNonLin(dim,n)
			if(nnl < nnlmin): nnlmin = nnl
			if(nnl > nnlmax): nnlmax = nnl
			for iter in ii:
				if iter['log']['status'] != 0 and iter['log']['status'] != 'ok':
					continue
				robOptInfo = iter['robOptInfo']
				batime += robOptInfo['info']['baInfo'][0]['log']['time']
				msinfo = robOptInfo['info']['msInfo']
				mstime += msinfo[len(msinfo)-1]['log']['time']
				msiter += msinfo[len(msinfo)-1]['log']['noRestarts']
				so1xInfo = robOptInfo['info']['so1xInfo']
				try:
					sotime += so1xInfo[len(so1xInfo)-1]['log']['time']
					soiter += so1xInfo[len(so1xInfo)-1]['log']['maxEvals']
				except:
					for s in so1xInfo:
						if('log' in s):
							sotime += s['log']['time']
							soiter += s['log']['maxEvals']
				sstime += robOptInfo['info']['ssInfo'][0]['log']['time']
				total += 1
				diff = robOptInfo['diff']
				ss = diff['ss']
				ba = diff['ba']
				ms = diff['ms']
				so1x = diff['so1x']

				if ba<=threshold and ss>threshold:
					ssfneg += 1

				if ba<=threshold and ms>threshold:
					msfneg += 1

				if ba<=threshold and so1x>threshold:
					sofneg += 1


	# if(format == 'table'):
	print(fmt%(currentfno,dim,nnlmin,nnlmax,batime/total,(ssfneg/total)*100,sstime/total,(msfneg/total)*100,mstime/total,
												(sofneg/total)*100,sotime/total))
	# elif(format == 'latex'):
		# print(fmt%(currentfno,(ssfneg/total)*100,sstime/total,(msfneg/total)*100,mstime/total,
												# (sofneg/total)*100,sotime/total,batime/total))




def printRobOdiff(jsonfile, runs, fno, trainingscale, e):
	import json
	if jsonfile:
		with open(jsonfile, 'r') as fn:
			datastore = json.load(fn)

	print("Function#: %d. training scale = %s. e = %s"%(fno, trainingscale,e))
	print("m\tn\titer#\tss\t\tms\t\tso1x\t\tso2x\t\tso3x\t\tso4x\t\tbaron")
	threshold = 0
	for r in runs:
		pdeg=r[0]
		qdeg=r[1]
		key = "p%s_q%s"%(str(pdeg),str(qdeg))
		iterationInfo = datastore[key]["iterationinfo"]
		for iter in iterationInfo:
			iterno = iter['iterationNo']
			diff = iter['robOptInfo']['diff']
			mi = min(diff.values())
			ma = max(diff.values())
			if(abs(mi-ma)>0.1):
				ss = diff['ss']
				ba = diff['ba']
				ms = diff['ms']
				so1x = diff['so1x']
				# so2x = diff['so2x']
				# so3x = diff['so3x']
				# so4x = diff['so4x']
				spl = "\t\t\t"
				if ba<=threshold and ss>threshold:
					spl += "********\t"
				else: spl += "\t\t"
				if ba<=threshold and ms>threshold:
					spl += "********\t"
				else: spl += "\t\t"
				if ba<=threshold and so1x>threshold:
					spl += "********\t"
				else: spl += "\t\t"
				# if ba<=threshold and so2x>threshold:
				# 	spl += "********\t"
				# else: spl += "\t\t"
				# if ba<=threshold and so3x>threshold:
				# 	spl += "********\t"
				# else: spl += "\t\t"
				# if ba<=threshold and so4x>threshold:
				# 	spl += "********\t"
				# else: spl += "\t\t"

				# print("%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n%s"%(pdeg,qdeg,iterno,ss,ms,so1x,so2x,so3x,so4x,ba,spl))
				print("%d\t%d\t%d\t%f\t%f\t%f\t%f\n%s"%(pdeg,qdeg,iterno,ss,ms,so1x,ba,spl))



from apprentice import tools
# tools.numCoeffsPoly(self.dim, self.n)
print(tools.numCoeffsPoly(4, 3))
print(tools.numCoeffsPoly(6, 3))
print(tools.numCoeffsPoly(7, 3))

print("----")
print(tools.numCoeffsPoly(4, 4))
print(tools.numCoeffsPoly(6, 4))
print(tools.numCoeffsPoly(7, 4))



# print(tools.numCoeffsPoly(23, 2) + tools.numCoeffsPoly(23, 2))
# createTable1("test",'table')
createTable1("test",'latex')
# createTable2structure()
exit(1)

# plotmntesterrperfile("test/f18_noisepct10-1_s0_out_1x.299445.json", "../f18_test.txt","f19", "test")
plotmntesterrperfile("test/f20_noisepct10-1_s0_out_2x.299445.json", "../f20_test.txt","f20", "test")
exit(1)
infilePath = "../f8_noisepct10-3.txt"

cvoutfile = "test/f8_noisepct10-3_cv_out.299445.json"
s0outfile = "test/f8_noisepct10-3_s0_out.299445.json"
testfile8 = "../f8_test.txt"

box = np.array([[-1,1],[-1,1]])
debug = 1
infilePathNN = "../f8.txt"
s0outfileNN = "test/f8_s0_out.299445.json"

infilePath10_10_1 = "../f10_noisepct10-1.txt"
infilePath10_10_3 = "../f10_noisepct10-3.txt"
s0outfile10 = "test/f10_noisepct10-1_s0_out.299445.json"
s0outfile10_1x_10_1 = "test/f10_noisepct10-1_s0_out_1x.299445.json"
s0outfile10_2x_10_1 = "test/f10_noisepct10-1_s0_out_2x.299445.json"
s0outfile10_1x_10_3 = "test/f10_noisepct10-3_s0_out_1x.299445.json"
s0outfile10_2x_10_3 = "test/f10_noisepct10-3_s0_out_2x.299445.json"
testfile10 = "../f10_test.txt"

infilePath12_10_1 = "../f12_noisepct10-1.txt"
infilePath12_10_3 = "../f12_noisepct10-3.txt"
s0outfile12 = "test/f12_noisepct10-1_s0_out.299445.json"
s0outfile12_1x_10_1 = "test/f12_noisepct10-1_s0_out_1x.299445.json"
s0outfile12_2x_10_1 = "test/f12_noisepct10-1_s0_out_2x.299445.json"
s0outfile12_1x_10_3 = "test/f12_noisepct10-3_s0_out_1x.299445.json"
s0outfile12_2x_10_3 = "test/f12_noisepct10-3_s0_out_2x.299445.json"
testfile12 = "../f12_test.txt"

infilePath13_10_1 = "../f13_noisepct10-1.txt"
infilePath13_10_3 = "../f13_noisepct10-3.txt"
s0outfile13 = "test/f13_noisepct10-1_s0_out.299445.json"
s0outfile13_1x_10_1 = "test/f13_noisepct10-1_s0_out_1x.299445.json"
s0outfile13_2x_10_1 = "test/f13_noisepct10-1_s0_out_2x.299445.json"
s0outfile13_1x_10_3 = "test/f13_noisepct10-3_s0_out_1x.299445.json"
s0outfile13_2x_10_3 = "test/f13_noisepct10-3_s0_out_2x.299445.json"
testfile13 = "../f13_test.txt"

infilePath14_10_1 = "../f14_noisepct10-1.txt"
infilePath14_10_3 = "../f14_noisepct10-3.txt"
s0outfile14 = "test/f14_noisepct10-1_s0_out.299445.json"
s0outfile14_1x_10_1 = "test/f14_noisepct10-1_s0_out_1x.299445.json"
s0outfile14_2x_10_1 = "test/f14_noisepct10-1_s0_out_2x.299445.json"
s0outfile14_1x_10_3 = "test/f14_noisepct10-3_s0_out_1x.299445.json"
s0outfile14_2x_10_3 = "test/f14_noisepct10-3_s0_out_2x.299445.json"
testfile14 = "../f14_test.txt"

infilePath15_10_1 = "../f15_noisepct10-1.txt"
infilePath15_10_3 = "../f15_noisepct10-3.txt"
s0outfile15 = "test/f15_noisepct10-1_s0_out.299445.json"
s0outfile15_1x_10_1 = "test/f15_noisepct10-1_s0_out_1x.299445.json"
s0outfile15_2x_10_1 = "test/f15_noisepct10-1_s0_out_2x.299445.json"
s0outfile15_1x_10_3 = "test/f15_noisepct10-3_s0_out_1x.299445.json"
s0outfile15_2x_10_3 = "test/f15_noisepct10-3_s0_out_2x.299445.json"
testfile15 = "../f15_test.txt"

infilePath16_10_1 = "../f16_noisepct10-1.txt"
infilePath16_10_3 = "../f16_noisepct10-3.txt"
s0outfile16 = "test/f16_noisepct10-1_s0_out.299445.json"
s0outfile16_1x_10_1 = "test/f16_noisepct10-1_s0_out_1x.299445.json"
s0outfile16_2x_10_1 = "test/f16_noisepct10-1_s0_out_2x.299445.json"
s0outfile16_1x_10_3 = "test/f16_noisepct10-3_s0_out_1x.299445.json"
s0outfile16_2x_10_3 = "test/f16_noisepct10-3_s0_out_2x.299445.json"
testfile16 = "../f16_test.txt"

infilePath17_10_1 = "../f17_noisepct10-1.txt"
infilePath17_10_3 = "../f17_noisepct10-3.txt"
s0outfile17 = "test/f17_noisepct10-1_s0_out.299445.json"
s0outfile17_1x_10_1 = "test/f17_noisepct10-1_s0_out_1x.299445.json"
s0outfile17_2x_10_1 = "test/f17_noisepct10-1_s0_out_2x.299445.json"
s0outfile17_1x_10_3 = "test/f17_noisepct10-3_s0_out_1x.299445.json"
s0outfile17_2x_10_3 = "test/f17_noisepct10-3_s0_out_2x.299445.json"
testfile17 = "../f17_test.txt"

infilePath18_10_1 = "../f18_noisepct10-1.txt"
infilePath18_10_3 = "../f18_noisepct10-3.txt"
s0outfile18 = "test/f18_noisepct10-1_s0_out.299445.json"
s0outfile18_1x_10_1 = "test/f18_noisepct10-1_s0_out_1x.299445.json"
s0outfile18_2x_10_1 = "test/f18_noisepct10-1_s0_out_2x.299445.json"
s0outfile18_1x_10_3 = "test/f18_noisepct10-3_s0_out_1x.299445.json"
s0outfile18_2x_10_3 = "test/f18_noisepct10-3_s0_out_2x.299445.json"
testfile18 = "../f18_test.txt"

infilePath19_10_1 = "../f19_noisepct10-1.txt"
infilePath19_10_3 = "../f19_noisepct10-3.txt"
s0outfile19 = "test/f19_noisepct10-1_s0_out.299445.json"
s0outfile19_1x_10_1 = "test/f19_noisepct10-1_s0_out_1x.299445.json"
s0outfile19_2x_10_1 = "test/f19_noisepct10-1_s0_out_2x.299445.json"
s0outfile19_1x_10_3 = "test/f19_noisepct10-3_s0_out_1x.299445.json"
s0outfile19_2x_10_3 = "test/f19_noisepct10-3_s0_out_2x.299445.json"
testfile19 = "../f19_test.txt"

infilePath20_10_1 = "../f20_noisepct10-1.txt"
infilePath20_10_3 = "../f20_noisepct10-3.txt"
s0outfile20 = "test/f20_noisepct10-1_s0_out.299445.json"
s0outfile20_1x_10_1 = "test/f20_noisepct10-1_s0_out_1x.299445.json"
s0outfile20_2x_10_1 = "test/f20_noisepct10-1_s0_out_2x.299445.json"
s0outfile20_1x_10_3 = "test/f20_noisepct10-3_s0_out_1x.299445.json"
s0outfile20_2x_10_3 = "test/f20_noisepct10-3_s0_out_2x.299445.json"
testfile20 = "../f20_test.txt"

# runs = [[2,2],[3,3],[4,4],[5,5]]
runs2D = []
for i in range(1,7):
	for j in range(1,7):
		runs2D.append([i,j])

runs3D = []
for i in range(1,7):
	for j in range(1,7):
		# constantpluslinear = 4
		# if(tools.numCoeffsPoly(3, j)-constantpluslinear<=50):
		runs3D.append([i,j])

runs4D = []
for i in range(1,6):
	for j in range(1,6):
		# constantpluslinear = 5
		# if(tools.numCoeffsPoly(4, j)-constantpluslinear<=50):
		runs4D.append([i,j])

runs7D = []
for i in range(0,5):
	for j in range(0,4):
		# constantpluslinear = 5
		# if(tools.numCoeffsPoly(4, j)-constantpluslinear<=50):
		runs7D.append([i,j])

runs6D = []
for i in range(0,5):
	for j in range(0,4):
		# constantpluslinear = 5
		# if(tools.numCoeffsPoly(4, j)-constantpluslinear<=50):
		runs6D.append([i,j])

box14 = np.array([[1,3],[1,3]])
box17 = np.array([[-1,1],[-1,1],[-1,1]])
box18 = np.array([[-0.95,0.95],[-0.95,0.95],[-0.95,0.95],[-0.95,0.95]])
box19 = np.array([[-1,1],[-1,1],[-1,1],[-1,1]])
a=-4*np.pi
b=4*np.pi
box20 = np.array([[a,b],[a,b],[a,b],[a,b],[a,b],[a,b],[a,b]])

larr = np.array([10**i for i in range(2,-8,-1)])

roboptstrategy = "ss_ms_so_ba"

# runs = [[6,3],[7,3],[7,4],[7,5],[7,6],[7,7]]
# larr = np.array([10**i for i in np.linspace(3,-8,23)])

##############################################
# runCrossValidation(infilePath,box,cvoutfile,debug)


# runRappsipBaseStrategy(infilePath12,runs2D, box,"1x",s0outfile12,debug)
# tableS0(s0outfile12,testfile12,runs2D)
#
# runRappsipBaseStrategy(infilePath13,runs2D, box,"1x",s0outfile13,debug)
# tableS0(s0outfile13,testfile13,runs2D)
#
# runRappsipBaseStrategy(infilePath14,runs2D, box,"1x",s0outfile14,debug)
# tableS0(s0outfile14,testfile14,runs2D)
#
#
# runRappsipBaseStrategy(infilePath15,runs2D, box,"1x",s0outfile15,debug)
# tableS0(s0outfile15,testfile15,runs2D)
#
# runRappsipBaseStrategy(infilePath16,runs2D, box,"1x",s0outfile16,debug)
# tableS0(s0outfile16,testfile16,runs2D)

# runRappsipStrategy2(infilePath12, runs2D, larr,"all_p_q", box,".5x",s2outfile12,debug)
# runRappsipStrategy2(infilePath13, runs2D, larr,"all_p_q", box,"0.5x",s2outfile13,debug)
# runRappsipStrategy2(infilePath12, runs2D, larr,"ho_p_q", box,"1x",s2outfile12,debug)
# tableS0(s2outfile12,testfile12,runs2D,larr)
# runRappsipStrategy2(infilePath13, runs2D, larr,"ho_p_q", box,"1x",s2outfile13,debug)
# tableS0(s2outfile13,testfile13,runs2D,larr)
# runRappsipStrategy2(infilePath14, runs2D, larr,"ho_p_q", box,"1x",s2outfile14,debug)
# runRappsipStrategy2(infilePath13, runs2D, larr,"all_p_q", box,"0.5x",s2outfile13,debug)
# tableS0(s2outfile13,testfile13,runs2D,larr)

# prettyPrint(cvoutfile,s2outfile12,testfile12)

##############################################

# runRappsipBaseStrategy(infilePath10_10_1, runs4D, box19, "1x", roboptstrategy, s0outfile10_1x_10_1,debug=1)

##############################################

# runRappsipBaseStrategy(infilePath12_10_1, runs2D, box, "1x", roboptstrategy, s0outfile12_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath12_10_1, runs2D, box, "2x", roboptstrategy, s0outfile12_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath12_10_3, runs2D, box, "1x", roboptstrategy, s0outfile12_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath12_10_3, runs2D, box, "2x", roboptstrategy, s0outfile12_2x_10_3,debug=1)
#
# runRappsipBaseStrategy(infilePath13_10_1, runs2D, box, "1x", roboptstrategy, s0outfile13_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath13_10_1, runs2D, box, "2x", roboptstrategy, s0outfile13_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath13_10_3, runs2D, box, "1x", roboptstrategy, s0outfile13_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath13_10_3, runs2D, box, "2x", roboptstrategy, s0outfile13_2x_10_3,debug=1)
#
# runRappsipBaseStrategy(infilePath14_10_1, runs2D, box14, "1x", roboptstrategy, s0outfile14_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath14_10_1, runs2D, box14, "2x", roboptstrategy, s0outfile14_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath14_10_3, runs2D, box14, "1x", roboptstrategy, s0outfile14_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath14_10_3, runs2D, box14, "2x", roboptstrategy, s0outfile14_2x_10_3,debug=1)
#
# runRappsipBaseStrategy(infilePath15_10_1, runs2D, box, "1x", roboptstrategy, s0outfile15_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath15_10_1, runs2D, box, "2x", roboptstrategy, s0outfile15_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath15_10_3, runs2D, box, "1x", roboptstrategy, s0outfile15_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath15_10_3, runs2D, box, "2x", roboptstrategy, s0outfile15_2x_10_3,debug=1)
#
# runRappsipBaseStrategy(infilePath16_10_1, runs2D, box, "1x", roboptstrategy, s0outfile16_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath16_10_1, runs2D, box, "2x", roboptstrategy, s0outfile16_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath16_10_3, runs2D, box, "1x", roboptstrategy, s0outfile16_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath16_10_3, runs2D, box, "2x", roboptstrategy, s0outfile16_2x_10_3,debug=1)
#
# runRappsipBaseStrategy(infilePath17_10_1, runs3D, box17, "1x", roboptstrategy, s0outfile17_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath17_10_1, runs3D, box17, "2x", roboptstrategy, s0outfile17_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath17_10_3, runs3D, box17, "1x", roboptstrategy, s0outfile17_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath17_10_3, runs3D, box17, "2x", roboptstrategy, s0outfile17_2x_10_3,debug=1)
#
# runRappsipBaseStrategy(infilePath18_10_1, runs4D, box18, "1x", roboptstrategy, s0outfile18_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath18_10_1, runs4D, box18, "2x", roboptstrategy, s0outfile18_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath18_10_3, runs4D, box18, "1x", roboptstrategy, s0outfile18_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath18_10_3, runs4D, box18, "2x", roboptstrategy, s0outfile18_2x_10_3,debug=1)
#
# runRappsipBaseStrategy(infilePath19_10_1, runs4D, box19, "1x", roboptstrategy, s0outfile19_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath19_10_1, runs4D, box19, "2x", roboptstrategy, s0outfile19_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath19_10_3, runs4D, box19, "1x", roboptstrategy, s0outfile19_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath19_10_3, runs4D, box19, "2x", roboptstrategy, s0outfile19_2x_10_3,debug=1)

# runRappsipBaseStrategy(infilePath20_10_1, runs7D, box20, "1x", roboptstrategy, s0outfile20_1x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath20_10_1, runs7D, box20, "2x", roboptstrategy, s0outfile20_2x_10_1,debug=1)
# runRappsipBaseStrategy(infilePath20_10_3, runs7D, box20, "1x", roboptstrategy, s0outfile20_1x_10_3,debug=1)
# runRappsipBaseStrategy(infilePath20_10_3, runs7D, box20, "2x", roboptstrategy, s0outfile20_2x_10_3,debug=1)

##############################################
# 6D function test
s0outfile6D_1x = "test/f6D_s0_out_1x.299445.json"
s0outfile6D_2x = "test/f6D_s0_out_2x.299445.json"
# s0outfile6D_1x = "test/f6D_s0_out_1x_unscaled.299445.json"
# s0outfile6D_2x = "test/f6D_s0_out_2x.unscaled.299445.json"
sixDdata = "../../workflow/data/DM_6D.h5"
import apprentice
try:
	X,Y = apprentice.tools.readData(sixDdata)
except:
	DATA = apprentice.tools.readH5(sixDdata, [0])
	X, Y= DATA[0]
a = -1
b = 1
box6D = np.array([[a,b],[a,b],[a,b],[a,b],[a,b],[a,b]])
box6D = np.array([[1.251664,2.999495],[-5.998021, -3.00295],[0.2001926,0.5998408],[478.044,609.9385],[170.088,289.9473],[0.5009486,3.497425]])
runRappsipBaseStrategyFromPoints(X,Y, runs6D, box6D, "1x", roboptstrategy, s0outfile6D_1x,debug=1,debugfile="/tmp/s0_6D_latest.json")
runRappsipBaseStrategyFromPoints(X,Y, runs6D, box6D, "2x", roboptstrategy, s0outfile6D_2x,debug=1,debugfile="/tmp/s0_6D_latest.json")
##############################################

# plotmntesterr([s0outfile12_1x_10_1,s0outfile12_2x_10_1,s0outfile12_1x_10_3,s0outfile12_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile12, runs2D, "f12","test")
# plotmntesterr([s0outfile13_1x_10_1,s0outfile13_2x_10_1,s0outfile13_1x_10_3,s0outfile13_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile13, runs2D, "f13","test")
# plotmntesterr([s0outfile14_1x_10_1,s0outfile14_2x_10_1,s0outfile14_1x_10_3,s0outfile14_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile14, runs2D, "f14","test")
# plotmntesterr([s0outfile15_1x_10_1,s0outfile15_2x_10_1,s0outfile15_1x_10_3,s0outfile15_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile15, runs2D, "f15","test")
# plotmntesterr([s0outfile16_1x_10_1,s0outfile16_2x_10_1,s0outfile16_1x_10_3,s0outfile16_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile16, runs2D, "f16","test")
# plotmntesterr([s0outfile17_1x_10_1,s0outfile17_2x_10_1,s0outfile17_1x_10_3,s0outfile17_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile17, runs3D, "f17","test")
# plotmntesterr([s0outfile18_1x_10_1,s0outfile18_2x_10_1,s0outfile18_1x_10_3,s0outfile18_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile18, runs4D, "f18","test")
# plotmntesterr([s0outfile19_1x_10_1,s0outfile19_2x_10_1,s0outfile19_1x_10_3,s0outfile19_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile19, runs4D, "f19","test")
# plotmntesterr([s0outfile20_1x_10_1,s0outfile20_2x_10_1,s0outfile20_1x_10_3,s0outfile20_2x_10_3], ["e=10-1, 1x","e=10-1, 2x","e=10-3, 1x","e=10-3, 2x"], testfile20, runs4D, "f20","test")

##############################################
# for fno in range(12,17):
# for fno in range(14,17):
# 	for e in ["10-1", "10-3"]:
# 		for scale in ["1x","2x"]:
# 			jsonfile = "test/f%d_noisepct%s_s0_out_%s.299445.json"%(fno,e,scale)
# 			printRobOdiff(jsonfile, runs2D, fno,scale,e)
# 			print("\n")
#
# for fno in range(17,18):
# 	for e in ["10-1", "10-3"]:
# 		for scale in ["1x","2x"]:
# 			jsonfile = "test/f%d_noisepct%s_s0_out_%s.299445.json"%(fno,e,scale)
# 			printRobOdiff(jsonfile, runs3D, fno,scale,e)
# 			print("\n")

# for fno in range(18,20):
# 	for e in ["10-1", "10-3"]:
# 		for scale in ["1x","2x"]:
# 			jsonfile = "test/f%d_noisepct%s_s0_out_%s.299445.json"%(fno,e,scale)
# 			printRobOdiff(jsonfile, runs4D, fno,scale,e)
# 			print("\n")

##############################################





#end
