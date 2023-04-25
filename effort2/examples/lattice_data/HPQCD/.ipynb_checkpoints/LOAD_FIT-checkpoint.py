import gvar
from numpy import *
import CHIRAL_LOGS
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

DATA_FILE=open('CORRELATIONS_HQET.txt',"r").readlines()
parameters = eval("dict("+DATA_FILE[0]+")")
params = {}
for order in range(4):
	for A in ['hA1','hA2','hA3','hV','hT1','hT2','hT3']:
		params['a^'+str(order)+"_"+A]=eval(parameters['a^'+str(order)+"_"+A])
for A in ['hA1','hA2','hA3','hV','hT1','hT2','hT3']:
    params["Mpi^2/Lambda_"+A]=eval(parameters["Mpi^2/Lambda_"+A])
    
params["gDDpi"]=eval(parameters["gDDpi"])

correlations = eval("dict("+DATA_FILE[1]+")")
params = gvar.correlate(params,correlations)
for X in params:
  print(X+"\t  =  \t"+str(params[X]))


lambdaqcdphys = 0.5

MDSPHYS = 2.010
MBPHYS = 5.27964

MBsPHYS = 5.3669
MDsSPHYS = 2.112

MAXORDER = 4

mKPHYS =0.493677
mPIPHYS=0.13957  
mETAPHYS = 0.547862  
fpi=0.130              
	
def fitprintA(p,A,QSQ):

        qsq=QSQ
        value = 0
        w=(MBPHYS**2.0+MDSPHYS**2.0-QSQ)/(2*MBPHYS*MDSPHYS)
        
        for order in range(MAXORDER):
                cumulator =  p['a^'+str(order)+'_'+A]
                if order ==0:value = value + cumulator
                else:value = value + cumulator*((w-1)**order)
        value=value
        
        return value

def fitprintADs(p,A,QSQ):
        qsq=QSQ
        value = 0
        w=(MBsPHYS**2.0+MDsSPHYS**2.0-QSQ)/(2*MBsPHYS*MDsSPHYS)
        if A == 'hV' :jint=1                                         
        if A == 'hA1':jint=1
        if A == 'hA2':jint=2
        if A == 'hA3':jint=3
                
        if A == 'hT1':jint=1
        if A == 'hT2':jint=0
        if A == 'hT3':jint=2
                        
        chilogphys = CHIRAL_LOGS.CHIRALLOGS(jint,gvar.mean(w),mPIPHYS,mKPHYS,mETAPHYS,'l','physical')
        chilogDs = CHIRAL_LOGS.CHIRALLOGS(jint,gvar.mean(w),mPIPHYS,mKPHYS,mETAPHYS,'s','physical')
             
        chilog = chilogDs*p['gDDpi']**2/(16.0*3.141592**2*fpi**2)+p['Mpi^2/Lambda_'+A]*(mKPHYS/CHIRAL_LOGS.LambdaChi)**2
        chilog=chilog-(chilogphys*p['gDDpi']**2/(16.0*3.141592**2*fpi**2)+p['Mpi^2/Lambda_'+A]*(mPIPHYS/CHIRAL_LOGS.LambdaChi)**2)

        for order in range(MAXORDER):
                        cumulator =  p['a^'+str(order)+'_'+A]
                        if order ==0:value = value + cumulator
                        else:value = value + cumulator*((w-1)**order)
        value=value+chilog
        return value

#PERFORM EXPLICIT CHECKS
Synthetic_datapoints = gvar.BufferDict()

print("CHECKS FOR B to D*")
CHECKFILE=open('CHECKS_HQET.txt',"r").readlines()
for iA in range(7):
	A = ['hA1','hA2','hA3','hV','hT1','hT2','hT3'][iA]
	fitcheck = eval(CHECKFILE[iA])
	fitprint=[[],[],[]]
	for q in range(5):
		QSQ = ((MBPHYS-MDSPHYS)**2.0)*q/4
		fitprint[0].append(QSQ)
		fitprint[1].append(fitprintA(params,A,QSQ).mean)
		fitprint[2].append(fitprintA(params,A,QSQ).sdev)
	for x in range(3):
		for y in range(5):
			if not fitprint[x][y] == fitcheck[x][y]:
				if not abs(fitprint[x][y]/fitcheck[x][y]-1.0)<=0.00000001:
					print("error in checks for form factor "+A)
					print( str(fitprint[x][y])+"=/="+str(fitcheck[x][y]) )
			else: print("OK")

		
print("CHECKS FOR Bs to Ds*")			
CHECKFILE=open('CHECKS_s_HQET.txt',"r").readlines()
for iA in range(7):
	A = ['hA1','hA2','hA3','hV','hT1','hT2','hT3'][iA]
	fitcheck = eval(CHECKFILE[iA])
	fitprint=[[],[],[]]
	for q in range(5):
		QSQ = ((MBsPHYS-MDsSPHYS)**2.0)*q/4
		fitprint[0].append(QSQ)
		fitprint[1].append(fitprintADs(params,A,QSQ).mean)
		fitprint[2].append(fitprintADs(params,A,QSQ).sdev)
	for x in range(3):
		for y in range(5):
			if not fitprint[x][y] == fitcheck[x][y]:
				if not abs(fitprint[x][y]/fitcheck[x][y]-1.0)<=0.00000001:
					print("error in checks for form factor "+A)
					print( str(fitprint[x][y])+"=/="+str(fitcheck[x][y]) )
			else: print("OK")
			
			
			
	
print("CHECKS FOR SYNTHETIC DATAPOINTS")	
loadcheck=gvar.gload('synthetic_data.pydat')		
synthetic_datapoints = gvar.BufferDict()

synthplotA={}
synthplotw={}
for iA in range(7):
	A = ['hA1','hA2','hA3','hV','hT1','hT2','hT3'][iA]
	synthplotA[A]=[]
	synthplotw[A]=[]
	synthplotA[A+"s"]=[]
	synthplotw[A+"s"]=[]
	for q in range(4):
		QSQ = ((MBPHYS-MDSPHYS)**2.0)*q/3
		w=(MBPHYS**2.0+MDSPHYS**2.0-QSQ)/(2*MBPHYS*MDSPHYS)
		synthplotw[A].append(w)
		synthetic_datapoints[A+"_"+str(QSQ)]=fitprintA(params,A,QSQ)
		synthplotA[A].append(synthetic_datapoints[A+"_"+str(QSQ)])
		print(synthetic_datapoints[A+"_"+str(QSQ)]==loadcheck[A+"_"+str(QSQ)])
		
		QSQ = ((MBsPHYS-MDsSPHYS)**2.0)*q/3
		w=(MBsPHYS**2.0+MDsSPHYS**2.0-QSQ)/(2*MBsPHYS*MDsSPHYS)
		synthplotw[A+"s"].append(w)
		synthetic_datapoints[A+"s_"+str(QSQ)]=fitprintADs(params,A,QSQ)
		synthplotA[A+"s"].append(synthetic_datapoints[A+"s_"+str(QSQ)])
		print(synthetic_datapoints[A+"s_"+str(QSQ)]==loadcheck[A+"s_"+str(QSQ)])

#Make some plots
if True:
	for iA in range(7):
		A =  ['hA1','hA2','hA3','hV','hT1','hT2','hT3'][iA]
		Alab = [r'$h_{A_1}$',r'$h_{A_2}$',r'$h_{A_3}$',r'$h_V$',r'$h_{T_1}$',r'$h_{T_2}$',r'$h_{T_3}$'][iA]
		fitprint=[[],[]]
		for q in range(100):
			QSQ = ((MBPHYS-MDSPHYS)**2.0)*(q)/99.0
			w=(MBPHYS**2.0+MDSPHYS**2.0-QSQ)/(2*MBPHYS*MDSPHYS)
			fitprint[0].append(fitprintA(params,A,QSQ))
			fitprint[1].append(w)
		plt.fill_between(fitprint[1], [x.mean+x.sdev for x in fitprint[0]], [x.mean-x.sdev for x in fitprint[0]], alpha=0.25,label=r'$B\to D^*$',color='blue')
		plt.plot(fitprint[1], [x.mean for x in fitprint[0]],color='blue')


		plt.errorbar(x = synthplotw[A],y = [z.mean for z in synthplotA[A]], yerr= [z.sdev for z in synthplotA[A]], color='blue',ls='none',marker='x', capsize=20)

		fitprint=[[],[]]
		for q in range(100):
			QSQ = ((MBsPHYS-MDsSPHYS)**2.0)*(q)/99.0
			w=(MBsPHYS**2.0+MDsSPHYS**2.0-QSQ)/(2*MBsPHYS*MDsSPHYS)
			fitprint[0].append(fitprintADs(params,A,QSQ))
			fitprint[1].append(w)
		plt.fill_between(fitprint[1], [x.mean+x.sdev for x in fitprint[0]], [x.mean-x.sdev for x in fitprint[0]], alpha=0.25,label=r'$B_s\to D_s^*$',color='green')
		plt.plot(fitprint[1], [x.mean for x in fitprint[0]],color='green')
		
		plt.errorbar(x = synthplotw[A+"s"],y = [z.mean for z in synthplotA[A+"s"]], yerr= [z.sdev for z in synthplotA[A+"s"]], color='green',ls='none',marker='x', capsize=20)
		
		plt.legend(loc='upper right')
		plt.ylabel(Alab)
		plt.xlabel(r'$w$')
		plt.show()	
			



