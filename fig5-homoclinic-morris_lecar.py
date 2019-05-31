from brian import *
import sys, subprocess, os
import numpy as np
import matplotlib
from matplotlib import lines
import pylab as pl

# Key simulation parameters.
wee_n = 90			# exc-exc coupling strength (nS)
Ek_list = [-70]			# potassium reversal potential (mV)
ext_max = 1.2			# maximum external input to exc neurons (kHz)
	
# Run simulations
for vk_n in Ek_list:
  
    defaultclock.dt = 0.05 *ms

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
    
    vke = vk_n *mV
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

    vki = vk_n *mV
    vnai= 60 *mV
    vli = -60 *mV
    phii = 1.0 

    # simt:  total simulation time
    # psimt: run the simulation to stabilize
    simt  = 2500. *ms    
    psimt =  500. *ms
    
    # synaptic time constant
    tauext = 3.0 *ms        
    taue = 3.0 *ms 
    taui = 7.0 *ms 

    # synaptic coupling strength
    Jee  = wee_n *nsiemens  
    Jie  = 190.0 *nsiemens  
    Jei  =  50.0 *nsiemens  
    Jii  =  15.0 *nsiemens  
    # synaptic strength of external input
    win = 200 *nsiemens	
    
    # spike threshold, refractory period
    v_the = 15 *mV
    v_thi = 10 *mV
    rfe   = 2 *ms
    rfi   = 1 *ms

    # External inputs to exc and inh neurons.
    # Given a time vector tvec = [t0, t1, ..., tn] and a rate vector rvec = [r0, r1, ..., rn],
    # the function stim(t) generates a piecewise linear function that connects (t0,r0), (t1,r1), ..., (tn,rn).
    tvec  = [  0.*ms,    psimt,   (psimt+simt)/2.,      simt] 
    rvec  = [0. *Hz,     0. *Hz,         ext_max*kHz,     0. *Hz]

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
    exti = 0.6*kHz

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
    inpute = PoissonGroup(Ne, rates = exte)
    inputi = PoissonGroup(Ni, rates = exti)

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

    # Run the simulation
    net = Network(Pe,Pi,inpute,inputi,input_co1,input_co2,Cee,Cei,Cie,Cii,Me,Mi,Re,Ri)

    net.run(simt)

    # Display results.    
    
    #==== excitatory and inhibitory rate ====#
    pl.figure(1,figsize=(10,4))
    axfig2 = pl.subplot(111)
    pl.plot(Re.times/ms,Re.rate,'b',linewidth=2,label='exc')
    pl.plot(Ri.times/ms,Ri.rate,'r',linewidth=2,label='inh')
    pl.ylabel('Hz',fontsize=30)
    pl.xlabel('ms',fontsize=30)
    pl.xlim([500,2500])
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
	
    #===== hysteresis loop =====#

    # Construct the external input to excitatory neurons.     
    tvecx  = [0.,   psimt/ms,   (psimt/ms+simt/ms)/2.,  simt/ms] 
    rvecx  = [0. ,    0. ,         ext_max,                 0. ]

    def stimx(t):
	    rx = 0.
	    for j in range(len(tvecx)-1):
		    rx = rx + (rvecx[j] + (rvecx[j+1]-rvecx[j])/(tvecx[j+1]-tvecx[j])*(t-tvecx[j]))*(heavi(t-tvecx[j]) - heavi(t-tvecx[j+1]))
	    return rx
    tm = np.linspace(0.,2500.,len(Re.rate))
    ext_input = []
    for i_input in range(len(tm)):
	    ext_input.append(stimx(tm[i_input]))
    ext_input = np.array(ext_input)    
  

    pl.figure(2,figsize=(10,8))
    axfig2 = pl.subplot(111)
    plot(ext_input,Re.rate,linewidth=2,label=r'$\mathregular{E_{K}:}$'+str(vk_n))

    # remove boundary
    axfig2.spines['top'].set_visible(False)
    axfig2.spines['right'].set_visible(False)
    axfig2.get_xaxis().tick_bottom()
    axfig2.get_yaxis().tick_left()

    # axis
    pl.xlim([0.,ext_max])
    pl.yticks(np.arange(0,255,50))
    pl.xticks([0.5,1.,1.5])
    pl.ylim([0,270])
    pl.xlabel('external input (kHz)',fontsize=30)
    pl.ylabel('excitatory rate (Hz)',fontsize=30)
    
    # legend
    pl.legend(loc=2,fontsize=24,frameon=False)
    leg = pl.gca().get_legend()
    llines=leg.get_lines()
    pl.setp(llines,linewidth=3)  

    matplotlib.rcParams.update({'font.size':24})  

    
    reinit_default_clock(t=0*ms)
    clear(True)

pl.show()




