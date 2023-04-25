import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rcParams

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

rcParams['figure.figsize'] = 15, 15
rcParams.update({'font.size': 42})
CB_color_cycle = ['#332288', '#117733','#44aa99', '#88ccee'    , '#ddcc77','#cc6677', '#aa4499', '#882255']
CB_names =       ['blue'   ,'green'   ,'lbg'    ,'lightblue'   ,'yellow'  ,'mauve'   ,'purple' ,'deepprple'  ]

CBfriendly = {}
for i in range(len(CB_color_cycle)):
        CBfriendly[CB_names[i]]=CB_color_cycle[i]
        
LambdaChi = 1.0
deltachi = 0.142

def r(w):
  if not w>1.0: # To avoid issues with w dropping below 1, we set 0.9999999999999999999=1. Note that these functions should not be used for w<1.
      ret=1
  else:
      ret= np.log(w+np.sqrt(w**2-1))/np.sqrt(w**2-1)
  return ret

def E1(w):
  return np.pi/(w+1.0)

def E2(w):
  return -np.pi/(w+1.0)**2

def E3(w):
  return np.pi

def E(j,w):
  if j==1: ret = E1(w)
  if j==2: ret = E2(w)
  if j==3: ret = E3(w)
  return ret

def G1(w):
    if not w>1.0:
       retbit = -1.0/3.0
    else:
       retbit = -1.0/(2*(w**2-1))*(w-r(w))
    return retbit

def G2(w):
    if not w>1.0:
       retbit = 1.0/5.0
    else:
       retbit = (w**2+2-3*w*r(w))/(2*(w**2-1)**2)
    return retbit

def G3(w):
  return -1.0

def G(j,w):
  if j==1: ret = G1(w)
  if j==2: ret = G2(w)
  if j==3: ret = G3(w)
  return ret

def a(x,theta,w):
  return x*np.cos(theta)/np.sqrt(1.0+w*np.sin(2.0*theta)+0j)

def integratefromzerotopiby2(func,w,x):  # crude integrator only used for checking against the scipy integration routine. Can be used (with care) in place of the scipy integrator, but slower.
  RANGE=2000
  integral = 0.0
  dtheta = (np.pi/2.0)/float(RANGE)
  for i in range(RANGE):
    phi  = (np.pi/2.0)*(i  )/(float(RANGE))
    phip = (np.pi/2.0)*(i+1)/(float(RANGE))
    integral = integral +(func(w,x,phi)+func(w,x,phip))/2.0*dtheta
  return integral

def F1integrand(w,x,theta):
  ap = a(x,theta,w)
  return (ap/(1+w*np.sin(2*theta)))*(np.pi*(np.sqrt(1-ap**2+0j)-1) -np.sqrt(ap**2-1+0j)*np.log(1-2*ap*(ap+np.sqrt(ap**2-1+0j)))- 2*ap )

def F2integrand(w,x,theta):
  ap = a(x,theta,w)
  return (ap*np.sin(2*theta)/(1+w*np.sin(2*theta))**2)*(-3*np.pi/2.0*(np.sqrt(1-ap**2+0j)-1)    + (np.pi*ap**2/(2)  + ((3-4*ap**2)/(0.0-1.0j))*(-0.5*np.log(1-2*ap*(ap+np.sqrt(ap**2-1+0j))) ))/np.sqrt(1-ap**2+0j) -3*ap  )





def F1(w,x):
  return integratefromzerotopiby2(F1integrand,w,x)

def F2(w,x):
  return integratefromzerotopiby2(F2integrand,w,x)

def F3(w,x):
  return (x)*(np.pi*(np.sqrt(1-x**2+0j)-1)   -(np.sqrt(x**2-1+0j)*np.log(1-2*x*(x+np.sqrt(x**2-1+0j))) -2*x) )




def F1scipy(w,x):
  return integrate.quad(lambda theta: F1integrand(w,x,theta), 0, np.pi/2.0)#integratefromzerotopiby2(F2integrand,w,x)
def F2scipy(w,x):
  return integrate.quad(lambda theta: F2integrand(w,x,theta), 0, np.pi/2.0)#integratefromzerotopiby2(F2integrand,w,x)


STORED = {} #For more efficient computation, once the numerical values for a given combination of j, w and x have been computed, store them rather than repeating the integration.

def F(j,w,x):
  if str(j)+"_"+str(w)+"_"+str(x) in STORED:
   ret = STORED[str(j)+"_"+str(w)+"_"+str(x)]
  else:
   if j==1: ret = F1scipy(w,x)[0]
   if j==2: ret = F2scipy(w,x)[0]
   if j==3: ret = F3(w,x)
   STORED[str(j)+"_"+str(w)+"_"+str(x)]=ret
  return ret

def I(j,w,m,x):
  return -(m**2 *x * E(j,w) + m**2 *x**2 *np.log(m**2/LambdaChi**2)*G(j,w)+m**2*F(j,w,x))

def I2z(w,m,x):
  return (m**2 *x**2 *np.log(m**2/LambdaChi**2)/4.0)


def FA1(w,m,x):
  wt=w
  ret= -2*(I(1,wt,m,x)-0.5*I(3,wt,m,x)+(wt+1)*I(1,wt,m,0)+(wt**2-1)*I(2,wt,m,0) -2.5*I(3,wt,m,0))
  return ret

def FA2(w,m,x):
  wt=w
  return -2*(I(1,wt,m,x)+(wt+1)*I(2,wt,m,x)-I(1,wt,m,0)-(wt+1)*I(2,wt,m,0))

def FA3(w,m,x):
  wt=w
  return -2*(-(wt+1)*I(2,wt,m,x)-0.5*I(3,wt,m,x)+(wt+2)*I(1,wt,m,0) +wt*(wt+1)*I(2,wt,m,0)-2.5*I(3,wt,m,0))

def FBAR(j,w,m):
  if j==1: ret = FA1(w,m,-deltachi/m)
  if j==2: ret = FA2(w,m,-deltachi/m)
  if j==3: ret = FA3(w,m,-deltachi/m)
  if j==0: ret = 0
  return np.real(ret)


def CHIRALLOGS(j,wt,mpi,mkaon,meta,spectag,latt_char):
 w=wt
 if latt_char=='physical':
  if spectag=='l':
   LOGFULL =  (3.0/2.0)*FBAR(j,w,mpi)+FBAR(j,w,mkaon) +(1.0/6.0)*FBAR(j,w,meta)
   RETURN=LOGFULL
  if spectag=='s':
   LOGFULL =  2.0*FBAR(j,w,mkaon) +(2.0/3.0)*FBAR(j,w,meta)
   RETURN=LOGFULL  
 return RETURN



print(FBAR(1,1.2,0.3))
print(FBAR(2,1.2,0.3))
print(FBAR(3,1.2,0.3))






