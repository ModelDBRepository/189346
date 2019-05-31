from brian import *
import sys, subprocess, os
import numpy as np
import matplotlib
from matplotlib import lines
import pylab as pl


# The potassium reversal potential is recovered 
# from a pathological value (-60mV) toward physiological values:
# Eke for excitatory neurons and Eki for inhibitory neurons.
Eke = -70
Eki = -80

defaultclock.dt = 0.05 *ms

# simulation time
simt  = 2500 *ms    

# Number of neurons
M  = 3000
Ne = int(M)
Ni = int(0.25*M)
p  = 0.01

# Synaptic reversal potential
ve  = 25 *mV
vi  = -70 *mV

# Excitatory ML parameters
Ce  = 5 *nF 
gke = 22 *usiemens
gnae= 20 *usiemens
gle = 2 *usiemens

v1e = -7 *mV
v2e = 15 *mV
v3e = -8 *mV 
v4e = 15 *mV
v3te = 0 *mV
v4te = 15 *mV

#vke = part of the neuron model eqse (see below)
vnae= 60 *mV
vle = -60 *mV
phie = 0.55  ###

# Inhibitory ML parameters
Ci = 1.0 *nF  
gki = 7.0 *usiemens
gnai= 16.0 *usiemens
gli = 7.0 *usiemens 

v1i = -7.35 *mV
v2i = 23.0 *mV 
v3i = -12.0 *mV
v4i = 12.0 *mV 
v3ti = -15.0 *mV
v4ti = 10.0 *mV 

#vki = part of the neuron model eqsi (see below)
vnai= 60 *mV
vli = -60 *mV
phii = 1.0 

# synaptic decay time constants
tauext = 3.0 *ms 
taue = 3.0 *ms 
taui = 7.0 *ms 

# synaptic coupling strength
Jee  =  90.0 *nsiemens
Jie  = 190.0 *nsiemens
Jei  =  50.0 *nsiemens
Jii  =  15.0 *nsiemens
Jin  = 200 *nsiemens

# neuron threshold, refractory period
v_the = 15 *mV
v_thi = 10 *mV
rfe   = 2 *ms
rfi   = 1 *ms

# External input to excitatory neurons.
# Given a time vector tvec = [t0, t1, ..., tn] and a rate vector rvec = [r0, r1, ..., rn],
# the function stim(t) generates a piecewise linear function that connects (t0,r0), (t1,r1), ..., (tn,rn).
exc_base = 400 *Hz #500
stim = 900  *Hz

tvec  = [   0.*ms,  1*ms, 11*ms,  12*ms,      simt]
rvec  = [exc_base,  stim, stim, exc_base, exc_base]

def heavi(x):
	if x > 0:
	    return 1
	elif x <=0:
	    return 0
	    
def stim(t):
	r = 0.*Hz
	for j in range(len(tvec)-1):
		r = r + (rvec[j] + (rvec[j+1]-rvec[j])/(tvec[j+1]-tvec[j])*(t-tvec[j]))*(heavi(t-tvec[j]) - heavi(t-tvec[j+1]))
	return r

excinput = lambda t: stim(t)
inhinput = 200.*Hz

# neuron models

# excitatory ML neurons
eqse = Equations('''

dv/dt = ( - gnae*minf*(v - vnae) - gke*w*(v - vke) - gle*(v - vle) \
	  - gext*(v - ve) - ge*(v - ve) - gi*(v - vi) )/Ce : volt

dw/dt = phie*(winf - w)/tauw : 1

dgext/dt = -gext/tauext : siemens

dge/dt = -ge/taue : siemens

dgi/dt = -gi/taui : siemens

minf  = .5*(1 + tanh((v - v1e)/v2e)) : 1

winf  = .5*(1 + tanh((v - v3e)/v4e)) : 1

tauw  = 1/cosh((v - v3te)/(2*v4te))*ms : second

vke : volt
    
''')

# inhibitory ML neurons
eqsi = Equations('''

dv/dt = ( - gnai*minf*(v - vnai) - gki*w*(v - vki) - gli*(v - vli) \
	  - gext*(v - ve) - ge*(v - ve) - gi*(v - vi))/Ci : volt

dw/dt = phii*(winf - w)/tauw : 1

dgext/dt = -gext/tauext : siemens

dge/dt = -ge/taue : siemens

dgi/dt = -gi/taui : siemens

minf  = .5*(1 + tanh((v - v1i)/v2i)) : 1

winf  = .5*(1 + tanh((v - v3i)/v4i)) : 1

tauw  = 1/cosh((v - v3ti)/(2*v4ti))*ms : second

vki : volt

''')

# Create population of neurons
Pe = NeuronGroup(Ne, model=eqse,
      threshold='v>v_the',
      refractory = rfe)

Pi = NeuronGroup(Ni, model=eqsi,
      threshold='v>v_thi',
      refractory = rfi)

# Recurrent connections      
Cee = Connection(Pe, Pe, 'ge', weight=Jee, sparseness=p)
Cie = Connection(Pe, Pi, 'ge', weight=Jie, sparseness=p)
Cei = Connection(Pi, Pe, 'gi', weight=Jei, sparseness=p)
Cii = Connection(Pi, Pi, 'gi', weight=Jii, sparseness=p)


# External Poisson spikes
inpute = PoissonGroup(Ne, rates = excinput)
inputi = PoissonGroup(Ni, rates = inhinput)

# Connect the input and neurons.
input_co1 = IdentityConnection(inpute, Pe, 'gext', weight=Jin)    
input_co2 = IdentityConnection(inputi, Pi, 'gext', weight=Jin)

# Gradual decrease (1000 - 2000ms) of potassium reversal potentials.
# Excitatory neurons: from Ek=-60mV to Ek=-70mV
# Inhibitory neurons: from Ek=-60mV to Ek=-80mV
tm_e     = [ 0.*ms,  1000.*ms,  2000.*ms,   simt]
EKe_arr  = [-60*mV,  -60*mV,      Eke*mV,   Eke*mV]

tm_i     = [ 0.*ms,  1000.*ms,  2000.*ms,   simt]
EKi_arr  = [-60*mV,  -60*mV,      Eki*mV,   Eki*mV]

def EKe_updated(t):
    v = 0.*mV
    for j in range(len(tm_e)-1):
	    v = v + (EKe_arr[j] + (EKe_arr[j+1]-EKe_arr[j])/(tm_e[j+1]-tm_e[j])*(t-tm_e[j]))*(heavi(t-tm_e[j]) - heavi(t-tm_e[j+1]))
    return v

def EKi_updated(t):
    v = 0.*mV
    for j in range(len(tm_i)-1):
	    v = v + (EKi_arr[j] + (EKi_arr[j+1]-EKi_arr[j])/(tm_i[j+1]-tm_i[j])*(t-tm_i[j]))*(heavi(t-tm_i[j]) - heavi(t-tm_i[j+1]))
    return v

@network_operation
def VKe(v):
    Pe.vke = EKe_updated(v.t)

@network_operation
def VKi(v):
    Pi.vki = EKi_updated(v.t)
        
# Initialization
Pe.v = -40*mV 
Pe.w = 0.1 
Pi.v = -40*mV 
Pi.w = 0.1


# Record population activity
Me = MultiStateMonitor(Pe, record=True)
Mi = MultiStateMonitor(Pi, record=True)

Re = PopulationRateMonitor(Pe,bin=1*ms)
Ri = PopulationRateMonitor(Pi,bin=1*ms)

net = Network(VKi,VKe,Pe,Pi,inpute,inputi,input_co1,input_co2,Cee,Cei,Cie,Cii,Me,Mi,Re,Ri)

net.run(simt)

#==== excitatory and inhibitory firing rates ====#
pl.figure(1,figsize=(10,4))
axfig2 = pl.subplot(111)
pl.plot(Re.times[500:]/ms,Re.rate[500:],'b',linewidth=2,label='exc')
pl.plot(Re.times[500:]/ms,Ri.rate[500:],'r',linewidth=2,label='inh')
pl.ylabel('Hz',fontsize=30)
pl.xlabel('ms',fontsize=30)
pl.xlim([500,2500])
pl.ylim([0,400])
pl.yticks([0,200,400])

# remove boundary
axfig2.spines['top'].set_visible(False)
axfig2.spines['right'].set_visible(False)
axfig2.get_xaxis().tick_bottom()
axfig2.get_yaxis().tick_left()

# display the period Ek changes.
line = lines.Line2D([1000,2000],[400,400],lw=4,color='k',alpha=1)
line.set_clip_on(False)
axfig2.add_line(line)

# legend
pl.legend(loc=2,fontsize=24,frameon=False)
leg = pl.gca().get_legend()
llines=leg.get_lines()
pl.setp(llines,linewidth=3)


#==== voltage trace of sample neurons ====#
pl.figure(2,facecolor='white',figsize=(10,4))

# excitatory neuron
ax1 = subplot(211,frameon=False)
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)
pl.plot(Me['v'].times/ms,Me['v'][0],'b',linewidth=2)
pl.xlim([500.,2500.])


# inhibitory neuron
ax2 = subplot(212,frameon=False)
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)
pl.plot(Mi['v'].times/ms,Mi['v'][0],'r',linewidth=2)
pl.xlim([500.,2500.])

  
matplotlib.rcParams.update({'font.size':24})      

pl.show()







