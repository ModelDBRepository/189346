from brian import *
import sys, subprocess, os
import numpy as np
import matplotlib
from matplotlib import lines
import pylab as pl


defaultclock.dt = 0.05 *ms

# Number of neurons
M  = 3000
Ne = int(M)
Ni = int(0.25*M)
p  = 0. # neurons are not connected.

# ext_max: maximal external input
# EK:	   potassium reversal potential
ext_max = 8 *kHz
EK = [-90*mV, -75*mV, -60*mV]

for vk_n in EK:

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
    
    vke = vk_n 
    vnae= 60 *mV
    vle = -60 *mV
    phie = 0.55
    
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

    vki = vk_n 
    vnai= 60 *mV
    vli = -60 *mV
    phii = 1.0

    # simulation time
    simt  = 2000 *ms    

    # synaptic decay time constants
    tauext = 3.0 *ms       
    taue = 3.0 *ms 
    taui = 7.0 *ms 

    # synaptic coupling strength (neurons are not connected.)
    Jee  =   0.0 *nsiemens  
    Jie  =   0.0 *nsiemens  
    Jei  =   0.0 *nsiemens  
    Jii  =   0.0 *nsiemens  
    # synaptic strength of external inputs
    win = 200 *nsiemens
    
    # spike threshold, refractory period
    v_the = 15 *mV
    v_thi = 10 *mV
    rfe   = 2 *ms
    rfi   = 1 *ms

    # External inputs to exc and inh neurons.
    # Given a time vector tvec = [t0, t1, ..., tn] and a rate vector rvec = [r0, r1, ..., rn],
    # the function stim(t) generates a piecewise linear function that connects (t0,r0), (t1,r1), ..., (tn,rn).
    tvec  = [0.*ms,  simt] 
    rvec  = [0.*Hz, ext_max]

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

    ratee = lambda t: stim(t)
    ratei = lambda t: stim(t)


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

    ''')

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
    inpute = PoissonGroup(Ne, rates = ratee)
    inputi = PoissonGroup(Ni, rates = ratei)

    # Connect the input to neurons.
    input_co1 = IdentityConnection(inpute, Pe, 'gext', weight=win)    
    input_co2 = IdentityConnection(inputi, Pi, 'gext', weight=win)
        
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

    # Run simulations
    net = Network(Pe,Pi,inpute,inputi,input_co1,input_co2,Cee,Cei,Cie,Cii,Me,Mi,Re,Ri)

    net.run(simt)

    # Display results.
    
    #===== inhibitory rate =====#
    pl.figure(1,figsize=(10,8))
    axfig2 = pl.subplot(111)
    plot(np.arange(0,8,8./len(Ri.times)),Ri.rate,label=r'$\mathregular{E_{K}: }$'+str(vk_n))


    # remove boundary
    axfig2.spines['top'].set_visible(False)
    axfig2.spines['right'].set_visible(False)
    axfig2.get_xaxis().tick_bottom()
    axfig2.get_yaxis().tick_left()

    # axis
    pl.xlim([0.,8.])
    pl.ylim([0,550])
    pl.xticks(np.arange(0,9,2))
    pl.yticks([100,200,300,400,500])
    pl.xlabel('input rate (kHz)',fontsize=30)
    pl.ylabel('inh rate (Hz)',fontsize=30)

    # legend
    pl.legend(loc='best',fontsize=24,frameon=False)
    leg = pl.gca().get_legend()
    llines=leg.get_lines()
    pl.setp(llines,linewidth=3)

    #===== excitatory rate =====#
    pl.figure(2,figsize=(10,8))
    axfig2 = pl.subplot(111)
    plot(np.arange(0,8,8./len(Re.times)),Re.rate,label=r'$\mathregular{E_{K}: }$'+str(vk_n))

    # remove boundary
    axfig2.spines['top'].set_visible(False)
    axfig2.spines['right'].set_visible(False)
    axfig2.get_xaxis().tick_bottom()
    axfig2.get_yaxis().tick_left()

    # axis
    pl.xlim([0.,8.])
    pl.ylim([0,550])
    pl.xticks(np.arange(0,9,2))
    pl.yticks([100,200,300,400,500])
    pl.xlabel('input rate (kHz)',fontsize=30)
    pl.ylabel('exc rate (Hz)',fontsize=30)

    # legend
    pl.legend(loc='best',fontsize=24,frameon=False)
    leg = pl.gca().get_legend()
    llines=leg.get_lines()
    pl.setp(llines,linewidth=3)

    matplotlib.rcParams.update({'font.size':24})


    
    reinit_default_clock(t=0*ms)
    clear(True)

pl.show()




