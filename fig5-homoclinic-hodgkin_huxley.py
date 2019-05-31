from brian import *
import sys, subprocess, os
import numpy as np
import matplotlib
from matplotlib import lines
import pylab as pl


# Key simulation parameters.
wee_n = 35.			# exc-exc coupling strength (nS)
gki_list = [15.]		# inhibitory neuron's potassium conductance
ext_max = 8.			# maximum external input to exc neurons (kHz)
	
# Run simulations with different g(K)_inh values.	
for gki in gki_list:
  
	defaultclock.dt=0.02*ms

	# Number of neurons
	N = 3000 
	Ne = int(N)
	Ni = int(N*0.25)
	p  = 0.01

	# simulation time
	sim_time = 2200 *ms

	# Synaptic reversal potential
	ve  = 55 *mV
	vi  = -90 *mV

	# synaptic time constant
	taue = 3 *msecond
	taui = 10 *msecond
	tauext = 3 *msecond

	# synaptic coupling strength
	wee=wee_n *nsiemens*(cmeter**-2) 
	wie=40 *nsiemens*(cmeter**-2) 
	wei=70.*nsiemens*(cmeter**-2) 
	wii=5.*nsiemens*(cmeter**-2) 
	wext=20.*nsiemens*(cmeter**-2)

	# External inputs to exc and inh neurons.
	# Given a time vector tvec = [t0, t1, ..., tn] and a rate vector rvec = [r0, r1, ..., rn],
	# the function stim(t) generates a piecewise linear function that connects (t0,r0), (t1,r1), ..., (tn,rn).
	tvec  = [  0.*ms,  200.*ms, 1200.*ms,   sim_time] 
	rvec  = [0.*Hz,    0.*Hz,      ext_max *kHz,    0.*Hz] 

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

	exte = lambda t: stim(t)
	exti = 50 *Hz 

	# HH neuron parameters #
	E_na = 50 *mV
	E_ke = -70 *mV
	E_ki = -70 *mV
	E_cl = -82 *mV
	E_ca = 120 *mV

	C = 1. *nfarad*(cmeter**-2)
	g_na = 100. *usiemens * (cmeter**-2)
	g_ke = 40. *usiemens * (cmeter**-2)
	g_ki = gki *usiemens * (cmeter**-2)
	g_kl = 0.05 *usiemens * (cmeter**-2) 
	g_nal = 0.0175 *usiemens * (cmeter**-2)
	g_cl = 0.05 *usiemens * (cmeter**-2)
	phi = 3 

	# neuron models
	
	# excitatory HH neurons
	eqe = Equations('''
	dv/dt   = ( -Ina - Ik - Icl - ge*(v-ve) - gi*(v-vi) - gext*(v-ve))/C :volt

	dn/dt   = phi*(alpha_n*(1-n) - beta_n*n) :1

	dh/dt   = phi*(alpha_h*(1-h) - beta_h*h) :1

	Ina = g_na*(minf**3*h)*(v-E_na) + g_nal*(v-E_na) :amp/meter**2
	Ik  = g_ke*(n**4)*(v - E_ke) + g_kl*(v - E_ke) :amp/meter**2
	Icl = g_cl*(v - E_cl) :amp/meter**2

	minf = alpha_m/(alpha_m + beta_m) :1
	alpha_m = 0.1*(mV**-1)*(v + 30.*mV)/(1 - exp(-0.1*(mV**-1)*(v+30.*mV)))/ms :Hz
	beta_m  = 4.*exp(-(v+55.*mV)/(18.*mV))/ms :Hz

	alpha_n = 0.01*(mV**-1)*(v+34.*mV)/(1-exp(-0.1*(mV**-1)*(v+34.*mV)))/ms :Hz
	beta_n  = 0.125*exp(-(v+44.*mV)/(80.*mV))/ms :Hz

	alpha_h = 0.07*exp(-(v+44.*mV)/(20.*mV))/ms :Hz
	beta_h  = 1/(1 + exp(-0.1*(mV**-1)*(v+14.*mV)))/ms :Hz

	dge/dt = -ge/taue : siemens/meter**2
	dgi/dt = -gi/taui : siemens/meter**2
	dgext/dt = -gext/tauext : siemens/meter**2
	''')

	# inhibitory HH neurons
	eqi = Equations('''
	dv/dt   = ( -Ina - Ik - Icl - ge*(v-ve) - gi*(v-vi) - gext*(v-ve))/C :volt

	dn/dt   = phi*(alpha_n*(1-n) - beta_n*n) :1

	dh/dt   = phi*(alpha_h*(1-h) - beta_h*h) :1

	Ina = g_na*(minf**3*h)*(v-E_na) + g_nal*(v-E_na) :amp/meter**2
	Ik  = g_ki*(n**4)*(v - E_ki) + g_kl*(v - E_ki) :amp/meter**2
	Icl = g_cl*(v - E_cl) :amp/meter**2

	minf = alpha_m/(alpha_m + beta_m) :1
	alpha_m = 0.1*(mV**-1)*(v + 30.*mV)/(1 - exp(-0.1*(mV**-1)*(v+30.*mV)))/ms :Hz
	beta_m  = 4.*exp(-(v+55.*mV)/(18.*mV))/ms :Hz

	alpha_n = 0.01*(mV**-1)*(v+34.*mV)/(1-exp(-0.1*(mV**-1)*(v+34.*mV)))/ms :Hz
	beta_n  = 0.125*exp(-(v+44.*mV)/(80.*mV))/ms :Hz

	alpha_h = 0.07*exp(-(v+44.*mV)/(20.*mV))/ms :Hz
	beta_h  = 1/(1 + exp(-0.1*(mV**-1)*(v+14.*mV)))/ms :Hz

	dge/dt = -ge/taue : siemens/meter**2
	dgi/dt = -gi/taui : siemens/meter**2
	dgext/dt = -gext/taue : siemens/meter**2
	''')

	# Create population of neurons
	Pe = NeuronGroup(Ne, model=eqe,
	threshold=EmpiricalThreshold(threshold=10*mV,refractory=2.*ms),method='RK')
	Pi = NeuronGroup(Ni, model=eqi,
	threshold=EmpiricalThreshold(threshold=10*mV,refractory=2.*ms),method='RK')
	    
	# Recurrent connections
	Cee=Connection(Pe,Pe,'ge',weight=wee,sparseness=p)
	Cie=Connection(Pe,Pi,'ge',weight=wie,sparseness=p)
	Cei=Connection(Pi,Pe,'gi',weight=wei,sparseness=p)
	Cii=Connection(Pi,Pi,'gi',weight=wii,sparseness=p)
	    
	# Create external Poisson spikes
	poisson_e = PoissonGroup(Ne, rates = exte)
	poisson_i = PoissonGroup(Ni, rates = exti)

	# Connect external input to neurons.
	inpute = IdentityConnection(poisson_e, Pe, 'gext', weight=wext)    
	inputi = IdentityConnection(poisson_i, Pi, 'gext', weight=wext)

	# Initialization
	Pe.v = -80*mV
	Pe.ge=(randn(len(Pe))*1.5+4)*10.*nS
	Pi.v = -80*mV
	Pi.gi=(randn(len(Pi))*1.5+4)*10.*nS

	# Record network activity
	volte = StateMonitor(Pe,'v',record=[0])
	volti = StateMonitor(Pi,'v',record=[0])
	Re = PopulationRateMonitor(Pe,bin=1*ms)
	Ri = PopulationRateMonitor(Pi,bin=1*ms)


	# Run simulations
	net = Network(Pe,Pi,poisson_e,poisson_i,Cee,Cie,Cei,Cii,inpute,inputi,volte,volti,Re,Ri)

	net.run(sim_time)

	# Display results

	#==== excitatory and inhibitory rate ====#
	pl.figure(1,figsize=(10,4))
	axfig2 = pl.subplot(111)
	pl.plot(Re.times/ms,Re.rate,'b',linewidth=2,label='exc')
	pl.plot(Ri.times/ms,Ri.rate,'r',linewidth=2,label='inh')
	pl.ylabel('Hz',fontsize=30)
	pl.xlabel('ms',fontsize=30)
	pl.xlim([200,2200])
	pl.ylim([0,500])
	pl.yticks([0,200,400])

	# remove boundary
	axfig2.spines['top'].set_visible(False)
	axfig2.spines['right'].set_visible(False)
	axfig2.get_xaxis().tick_bottom()
	axfig2.get_yaxis().tick_left()

	# legend
	pl.legend(loc=4,fontsize=24,frameon=False)
	leg = pl.gca().get_legend()
	llines=leg.get_lines()
	pl.setp(llines,linewidth=3)


	#==== hysteresis loops ====#

	# external input to excitatory neurons
	sim_time = 2200.
	tvec  = [0.,  200.,       1200.,  sim_time] 
	rvec  = [0.,    0.,       ext_max,      0.] #max = 900

	def stime(t):
		r = 0.
		for j in range(len(tvec)-1):
			r = r + (rvec[j] + (rvec[j+1]-rvec[j])/(tvec[j+1]-tvec[j])*(t-tvec[j]))*(heavi(t-tvec[j]) - heavi(t-tvec[j+1]))
		return r

	extinput = []
	for i_input in range(len(Re.times)):
		extinput.append(stime(Re.times[i_input]*1000.))
	extinput = np.array(extinput)
	exc_rate = np.array(Re.rate)
	
	
	pl.figure(2,figsize=(10,8))
	axfig2 = pl.subplot(111)
	pl.plot(extinput[200:],exc_rate[200:],linewidth=2,label=r'$\mathregular{g_{k}^{inh}}$:'+str(int(gki)) + ' nS')

	# remove boundary
	axfig2.spines['top'].set_visible(False)
	axfig2.spines['right'].set_visible(False)
	axfig2.get_xaxis().tick_bottom()
	axfig2.get_yaxis().tick_left()

	# axis
	pl.xlim([0.,ext_max])
	pl.xticks([0,4,8])
	pl.yticks([0,200,400])
	pl.ylim([0,400])
	pl.xlabel('external input (kHz)',fontsize=30)
	pl.ylabel('excitatory rate (Hz)',fontsize=30)
			
	# legend
	pl.legend(loc=4,fontsize=24,frameon=False)
	leg = pl.gca().get_legend()
	llines=leg.get_lines()
	pl.setp(llines,linewidth=3)


	matplotlib.rcParams.update({'font.size':24})      

	reinit_default_clock(t=0*ms)
	clear(True)

pl.show()







