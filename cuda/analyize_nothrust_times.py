import numpy as np
from scipy import loadtxt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

#nparticles = [4,128,256,512,1024,4096]
nparticles=[5000, 4500, 4000, 3500, 3000,2500, 2000, 1500, 1400, 1300 ,1200 ,1100, 1000 ,900 ,800 ,700, 600 ,500 ,400 ,300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
rep_nparticles=[100000, 50000,10000, 5000, 3500, 2000, 1000 , 500 ,200, 100, 50, 20, 10]
niter =[1]
nsteps = [200]
#nsteps = [10,100,1000, 5000]
nblocks = [1,4,16,32,64,128]
nthreads = [1,16,64,128,256,512]

make_avg_ptot = False
times= False
ptot = True
make_scale_t = False
make_scale_b = False
ptot_scale_b = True
make_all_tb = False
make_heat = False
make_heat_particle = False
ptot_heat_particle = False

if(times):
	#z = open('nothrust2_reset_times_copy.txt')
	z = open('nothrust2_reset_times-200-v1.txt')
	events =[]
	total_run =[]
	for line in z.readlines():
		col = line.rstrip().split(' ')
		if (col[0] is '0'):
			events.append({"iterations":col[1],"nparticles":col[2],"steps":col[3],"t_iter":col[4], "t_average":col[5],"t_reset":col[7].rstrip()})#"t_rms":col[6],"t_reset":col[7].rstrip()})
		elif (col[0] is '1'):
			total_run.append({"iterations":int(col[1]),"nparticles":int(col[2]),"steps":int(col[3]),"blocks":int(col[4]),"threads":int(col[5]),"t_full":float(col[6]),"t_perStep":float(col[6])/int(col[3]),"t_perPart":float(col[6])/(float(col[3])*int(col[2]))})#, "t_reset_total":col[7],"t_reset_avg":col[8]})
	
	
	
	df = pd.DataFrame(total_run)
	for b in nblocks:
		ax = plt.subplot(211)
		bx = plt.subplot(212)
		leg =[]
		for t in nthreads:
			df[(df["threads"]==t) & (df["blocks"]==b) &(df["steps"]==200) ].plot(x="nparticles", y="t_perStep", ax=ax)
			df[(df["threads"]==t) & (df["blocks"]==b) &(df["steps"]==200) ].plot(x="nparticles", y="t_perPart", ax=bx)
			leg.append("blocks=%s,thread=%s"%(b,t))
		ax.legend(leg)
		bx.legend(leg)
		plt.show()
	ax = plt.subplot(211)
	bx = plt.subplot(212)
	leg =[]
	for s in nsteps:
		df[(df["threads"]==128) & (df["blocks"]==64) &(df["steps"]==s) ].plot(x="nparticles", y="t_perStep", ax=ax)
		df[(df["threads"]==128) & (df["blocks"]==64) &(df["steps"]==s) ].plot(x="nparticles", y="t_perPart", ax=bx)
		leg.append("steps=%s"%(s))
	ax.legend(leg)
	bx.legend(leg)
	plt.show()


if(make_avg_ptot):

	z1= open('ptot_steptimes-200-v1.txt')
	z2= open('ptot_steptimes-200-v2.txt')
	z3= open('ptot_steptimes-200-v3.txt')
	events =[]
	for line in z1.readlines():
		col = line.rstrip().split(' ')
		#if( col[0] == "double4"):
		if( col[0] == "float4"):
			events.append({"nparticles":int(col[1]),"iter":(float(col[2])+1), "nblocks":int(col[3]),"nthreads":int(col[4]),"ptot":float(col[5]),"time":float(col[6]),"timePerStep":(float(col[6])/(float(col[2])+1)),"ptotPerStep":float(col[5])/(float(col[2])+1),"file":1})
			#events.append({"nparticles":int(col[1]),"iter":(float(col[2])+1), "nblocks":int(col[3]),"nthreads":int(col[4]),"ptot":float(col[5]),"time":float(col[6]),"timePerStep":(float(col[6])/(float(col[2])+1)),"ptotPerStep":float(col[5])/(float(col[2])+1),"file":1})

	for line in z2.readlines():
		col = line.rstrip().split(' ')
		#if( col[0] == "double4"):
		if( col[0] == "float4"):
			events.append({"nparticles":int(col[1]),"iter":(float(col[2])+1), "nblocks":int(col[3]),"nthreads":int(col[4]),"ptot":float(col[5]),"time":float(col[6]),"timePerStep":(float(col[6])/(float(col[2])+1)),"ptotPerStep":float(col[5])/(float(col[2])+1),"file":2})
			#events.append({"nparticles":int(col[1]),"iter":(float(col[2])+1), "nblocks":int(col[3]),"nthreads":int(col[4]),"ptot":float(col[5]),"time":float(col[6]),"timePerStep":(float(col[6])/(float(col[2])+1)),"ptotPerStep":float(col[5])/(float(col[2])+1),"file":2})
	for line in z3.readlines():
		col = line.rstrip().split(' ')
		#if( col[0] == "double4"):
		if( col[0] == "float4"):
			events.append({"nparticles":int(col[1]),"iter":(float(col[2])+1), "nblocks":int(col[3]),"nthreads":int(col[4]),"ptot":float(col[5]),"time":float(col[6]),"timePerStep":(float(col[6])/(float(col[2])+1)),"ptotPerStep":float(col[5])/(float(col[2])+1),"file":3})
			#events.append({"nparticles":int(col[1]),"iter":(float(col[2])+1), "nblocks":int(col[3]),"nthreads":int(col[4]),"ptot":float(col[5]),"time":float(col[6]),"timePerStep":(float(col[6])/(float(col[2])+1)),"ptotPerStep":float(col[5])/(float(col[2])+1),"file":3})
	pf = pd.DataFrame(events)

	def get_avg(row,x, local):
		t = row["nthreads"]
		b = row["nblocks"]
		p = row["nparticles"]
		l = row["iter"]
		if local:	
			return np.mean(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]==l) ][x])
		else:
			return np.mean(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]>5) ][x])
		#return np.mean(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]>5) ]['timePerStep'])

	def get_std(row,x, local):
		t = row["nthreads"]
		b = row["nblocks"]
		p = row["nparticles"]
		l = row["iter"]
		if local:	
			return np.std(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]==l) ][x])
		else:
			return np.std(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]>5) ][x])
		#return np.std(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]>5) ]['timePerStep'])

	pf_new = pf[pf["file"]==2].filter(['nblocks','nthreads','nparticles','iter'], axis=1)
	pf_new['ptot'] = pf.apply(lambda row: get_avg(row,'ptot',True), axis=1)
	pf_new['ptot std'] = pf.apply(lambda row: get_std(row,'ptot',True), axis=1)
	pf_new['ptot avg'] = pf.apply(lambda row: get_avg(row,'ptotPerStep',False ), axis=1)
	pf_new['ptot avg std'] = pf.apply(lambda row: get_std(row,'ptotPerStep',False), axis=1)
	pf_new['time'] = pf.apply(lambda row: get_avg(row,'time',True), axis=1)
	pf_new['time std'] = pf.apply(lambda row: get_std(row,'time',True), axis=1)
	pf_new['time avg'] = pf.apply(lambda row: get_avg(row,'timePerStep',False), axis=1)
	pf_new['time avg std'] = pf.apply(lambda row: get_std(row,'timePerStep',False), axis=1)
	pf_new['time part avg'] = pf_new['time avg']/pf_new['nparticles']
	pf_new['time part avg std'] = pf_new['time avg std']/pf_new['nparticles']
	pf_new['ptot part avg'] = pf_new['ptot avg']/pf_new['nparticles']
	pf_new['ptot part avg std'] = pf_new['ptot avg std']/pf_new['nparticles']

	print(pf_new)
	np.savetxt(r'ptot_all3.txt', pf_new.values, fmt='%g')

if(ptot):
	z= open('ptot_all3.txt')
	#z1= open('ptot_steptimes-200-v1.txt')
	#z2= open('ptot_steptimes-200-v2.txt')
	#z3= open('ptot_steptimes-200-v3.txt')
	events =[]
	for line in z.readlines():
		col = line.rstrip().split(' ')
		#if( col[0] == "double4"):
		#if( col[0] == "float4"):
		step = (float(col[3]))
		events.append({"nparticles":int(col[2]),"iter":float(col[3]), "nblocks":int(col[0]),"nthreads":int(col[1]),
		"ptot":float(col[4]),"ptot std":float(col[5]),"ptot avg":float(col[6]),"ptot avg std":float(col[7]),
		"ptot step":float(col[4])/step,"ptot std step":float(col[5])/step,#"ptot avgstep ":float(col[6])/step,"ptot avg std step":float(col[7])/step,
		"time":float(col[8]),"time std":float(col[9]),"time avg":float(col[10]),"time avg std":float(col[11]),
		"time step":float(col[8])/step,"time std step":float(col[9])/step,#"time avg step":float(col[10])/step,"time avg atd step":float(col[11])/step
		"time part avg":float(col[12]),"time part std":float(col[13]),
		"ptot part avg":float(col[14]),"ptot part std":float(col[14])
		})

	#for line in z2.readlines():
	#	col = line.rstrip().split(' ')
	#	#if( col[0] == "double4"):
	#	if( col[0] == "float4"):
	#		events.append({"nparticles":int(col[1]),"iter":(float(col[2])+1), "nblocks":int(col[3]),"nthreads":int(col[4]),"ptot":float(col[5]),"time":float(col[6]),"timePerStep":(float(col[6])/(float(col[2])+1)),"ptotPerStep":float(col[5])/(float(col[2])+1),"file":2})
	#for line in z3.readlines():
	#	col = line.rstrip().split(' ')
	#	#if( col[0] == "double4"):
	#	if( col[0] == "float4"):
	#		events.append({"nparticles":int(col[1]),"iter":(float(col[2])+1), "nblocks":int(col[3]),"nthreads":int(col[4]),"ptot":float(col[5]),"time":float(col[6]),"timePerStep":(float(col[6])/(float(col[2])+1)),"ptotPerStep":float(col[5])/(float(col[2])+1),"file":3})
	pf = pd.DataFrame(events)
	print(pf)
	if(ptot_scale_b):
		for t in nthreads:
			ax = plt.subplot(111)
			leg =[]
			for b in nblocks:
			#for p in rep_nparticles:
				#pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==1000)].plot(x="iter", y="time step", ax=ax)
				pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==1000)].plot(x="iter", y="ptot step",yerr="ptot std step", ax=ax)
				leg.append("blocks=%s"%(b))
			ax.legend(leg)
			ax.set_ylabel("ptot per step")
			ax.set_title("ptot per step; nthreads = %s, particles = %s"%(t,1000))
			plt.show()
	if(make_scale_b):
		for t in nthreads:
			ax = plt.subplot(111)
			leg =[]
			for b in nblocks:
			#for p in rep_nparticles:
				#pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==1000)].plot(x="iter", y="time step", ax=ax)
				pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==1000)].plot(x="iter", y="time step",yerr="time std step", ax=ax)
				leg.append("blocks=%s"%(b))
			ax.legend(leg)
			ax.set_ylabel("time per step (ms)")
			ax.set_title("time per step; nthreads = %s, particles = %s"%(t,1000))
			plt.show()
	if(make_scale_t):
		for b in nblocks:
			ax = plt.subplot(111)
			leg =[]
			for t in nthreads:
			#for p in rep_nparticles:
				pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==1000)].plot(x="iter", y="time step",yerr="time std step", ax=ax)
				leg.append("threads=%s"%(t))
			ax.set_ylabel("time per step (ms)")
			ax.legend(leg)
			ax.set_title("time per step; nblocks = %s, particles = %s"%(b,1000))
			plt.show()



	if(make_all_tb):
		ax = plt.subplot(111)
		leg =[]
		b_markers = ["o","^","v","s","*","D"]
		t_colors = ["b","g","R","c","m","k","y"]
		for bi,b in enumerate(nblocks):
		#	ax = plt.subplot(111)
		#	leg =[]
			for ti,t in enumerate(nthreads):
			#for p in rep_nparticles:
		#		if t*b <1000:
				pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==1000)].plot(x="iter", y="time step", yerr="time std step", ax=ax, marker=b_markers[bi], color=t_colors[ti])
				leg.append("t=%s;b=%s"%(t,b))
		ax.legend(leg)
		ax.set_ylabel("time per step (ms)")
		ax.set_title("time per step; particles = %s"%(1000))
		plt.show()
#	def get_mean(row):
#		t = row["nthreads"]
#		b = row["nblocks"]
#		p = row["nparticles"]
#		return np.mean(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]>5) ]['time step'])
#
#	def get_std(row):
#		t = row["nthreads"]
#		b = row["nblocks"]
#		p = row["nparticles"]
#		#return "%f +/- %f" %(np.mean(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]>5) ]['timePerStep']), np.std(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]>5) ]['timePerStep']))
#		return np.std(pf[(pf["nthreads"]==t) & (pf["nblocks"]==b) &(pf["nparticles"]==p) & (pf["iter"]>5) ]['timePerStep'])
#	pf['avg'] = pf.apply(lambda row: get_mean(row), axis=1)
#	pf['std'] = pf.apply(lambda row: get_std(row), axis=1)
#	print(pf[(pf["nparticles"]==200) & (pf["nthreads"]==16) & (pf["nblocks"]==16)])
	if(make_heat):
		for p in rep_nparticles:
			fig, (ax1,ax2) = plt.subplots(ncols=2) 
			#fig, ax1 = plt.subplots(ncols=1) 
			pf_avg = pf[(pf["iter"]==101) & (pf["nparticles"]==p)].pivot(index="nblocks",columns="nthreads",values="time avg")
			pf_std = pf[(pf["iter"]==101) & (pf["nparticles"]==p)].pivot(index="nblocks",columns="nthreads",values="time std")
			#print(pf_avg)
			#ax1.bar3d(x=pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["nblocks"], y =pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["nthreads"], z=[0 for p in pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["avg"]],dx=4,dy=4,dz=pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["avg"])
			ax = sns.heatmap(pf_avg,annot=True,norm=LogNorm(vmin=.001, vmax=7), ax=ax1)
			ax = sns.heatmap(pf_avg,annot=pf_std,norm=LogNorm(vmin=.001, vmax=7), ax=ax2)
			plt.suptitle("Particles = %s"%p)
			ax1.set_title("Average time per step (ms)")
			ax2.set_title("Standard deviation")
			plt.show()
	if(make_heat_particle):
		for p in rep_nparticles:
			fig, (ax1,ax2) = plt.subplots(ncols=2) 
			#fig, ax1 = plt.subplots(ncols=1) 
			pf_avg = pf[(pf["iter"]==101) & (pf["nparticles"]==p)].pivot(index="nblocks",columns="nthreads",values="time part avg")
			pf_std = pf[(pf["iter"]==101) & (pf["nparticles"]==p)].pivot(index="nblocks",columns="nthreads",values="time part std")
			#print(pf_avg)
			#ax1.bar3d(x=pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["nblocks"], y =pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["nthreads"], z=[0 for p in pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["avg"]],dx=4,dy=4,dz=pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["avg"])
			ax = sns.heatmap(pf_avg,annot=True,norm=LogNorm(vmin=.001, vmax=7), ax=ax1)
			ax = sns.heatmap(pf_avg,annot=pf_std,norm=LogNorm(vmin=.001, vmax=7), ax=ax2)
			plt.suptitle("Particles = %s"%p)
			ax1.set_title("Average time per step per particle (ms)")
			ax2.set_title("Standard deviation")
			plt.show()
	if(ptot_heat_particle):
		for p in rep_nparticles:
			fig, (ax1,ax2) = plt.subplots(ncols=2) 
			#fig, ax1 = plt.subplots(ncols=1) 
			pf_avg = pf[(pf["iter"]==101) & (pf["nparticles"]==p)].pivot(index="nblocks",columns="nthreads",values="ptot part avg")
			pf_std = pf[(pf["iter"]==101) & (pf["nparticles"]==p)].pivot(index="nblocks",columns="nthreads",values="ptot part std")
			#print(pf_avg)
			#ax1.bar3d(x=pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["nblocks"], y =pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["nthreads"], z=[0 for p in pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["avg"]],dx=4,dy=4,dz=pf[(pf["iter"]==101) & (pf["nparticles"]==p)]["avg"])
			ax = sns.heatmap(pf_avg,annot=True,norm=LogNorm(vmin=.001, vmax=7), ax=ax1)
			ax = sns.heatmap(pf_avg,annot=pf_std,norm=LogNorm(vmin=.001, vmax=7), ax=ax2)
			plt.suptitle("Particles = %s"%p)
			ax1.set_title("Average ptot per step per particle")
			ax2.set_title("Standard deviation")
			plt.show()

