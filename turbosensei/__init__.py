# FORCsensei module 
# compile using: python3 setup.py sdist bdist_wheel

import os
import numpy as np
import codecs as cd
import scipy as sp
import scipy.stats as sps
from scipy import linalg
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, Layout, VBox, HBox
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from dask.distributed import Client, LocalCluster, progress #needed for multiprocessing
from scipy.special import gammaln, logsumexp
import numba

#define jit-based matrix inverse
@numba.jit
def inv_jit(A):
  return np.linalg.inv(A)

##### BEGIN SECTION: FILE SELECTION #################################################
class FileBrowser(object):
    def __init__(self):
        self.path = os.getcwd()
        self._update_files()

    def _update_files(self):
        self.files = list()
        self.dirs = list()
        if(os.path.isdir(self.path)):
            for f in os.listdir(self.path):
                ff = os.path.join(self.path, f)
                if os.path.isdir(ff):
                    self.dirs.append(f)
                else:
                    self.files.append(f)

    def widget(self):
        box = widgets.VBox()
        self._update(box)
        return box

    def _update(self, box):

        def on_click(b):
            if b.description == '..':
                self.path = os.path.split(self.path)[0]
            else:
                self.path = os.path.join(self.path, b.description)
            self._update_files()
            self._update(box)

        buttons = []
        if self.files:
            button = widgets.Button(description='..', background_color='#d0d0ff')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.dirs:
            button = widgets.Button(description=f, background_color='#d0d0ff')
            button.on_click(on_click)
            buttons.append(button)
        for f in self.files:
            button = widgets.Button(description=f)
            button.on_click(on_click)
            buttons.append(button)
        box.children = tuple([widgets.HTML("<h4>%s</h4>" % (self.path,))] + buttons)

##### BEGIN SECTION: PREPROCESSING  #################################################
def preprocessing_options(X):

    style = {'description_width': 'initial'} #general style settings

    ### Define sample properties ###
    fn = X['fn']
    prop_title = widgets.HTML(value='<h3>Sample preprocessing options</h3>')
    mass_title = widgets.HTML(value='To disable mass normalization use a value of -1')

    sample, unit, mass = sample_details(fn)

    sample_widge = widgets.Text(value=sample,description='Sample name:',style=style)
    
    if mass == "N/A":
        mass_widge = widgets.FloatText(value=-1, description = 'Sample mass (g):',style=style)
    else:
        mass_widge = widgets.FloatText(value=mass, description = 'Sample mass (g):',style=style)

    mass_widge1 = HBox([mass_widge,mass_title])
    
    ### Define measurement corrections ###
    correct_title = widgets.HTML(value='<h3>Select preprocessing options:</h3>')
    
    slope_widge = widgets.FloatSlider(
        value=70,
        min=1,
        max=100.0,
        step=1,
        description='Slope correction [%]:',
        style=style,
        readout_format='.0f',
    )
    
    slope_title = widgets.HTML(value='To disable high-field slope correction use a value of 100%')
    slope_widge1 = HBox([slope_widge,slope_title])

    #process_button = widgets.Button(description="Preprocess Data")
    #process_button.on_click(process_button_clicked)
    
    drift_widge = widgets.Checkbox(value=False, description='Measurement drift correction')
    fpa_widge = widgets.Checkbox(value=False, description='Remove first point artifact')
    lpa_widge = widgets.Checkbox(value=False, description='Remove last point artifact')
    #outlier_widge = widgets.Checkbox(value=False, description='Remove measurement outliers')
    correct_widge = VBox([correct_title,sample_widge,mass_widge1,slope_widge1,drift_widge,fpa_widge,lpa_widge])

    preprocess_nest = widgets.Tab()
    preprocess_nest.children = [correct_widge]
    preprocess_nest.set_title(0, 'PREPROCESSING')
    display(preprocess_nest)

    X["sample"] = sample_widge
    X["mass"] = mass_widge
    X["unit"] = unit
    X["drift"] = drift_widge
    X["slope"] = slope_widge
    X["fpa"] = fpa_widge
    X["lpa"] = lpa_widge 
    #X["outlier"] = outlier_widge
    
    return X


def plot_delta_hysteresis(X,ax):

    #unpack 
    M = X["DM"]
    H = X["H"]
    Fk = X["Fk"]
    Fj = X["Fj"]

    hfont = {'fontname':'STIXGeneral'}

    for i in range(5,int(np.max(Fk)),5):
    
        if X["mass"].value > 0.0: #SI and mass normalized (T and Am2/kg)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value/1000.0),'-k')        
        else: #SI not mass normalized (T and Am2)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)],'-k') 
      
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=12,color='k')
    ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('k')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
  
    ylim=np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-ylim*0.1,ylim])
    yticks0 = ax.get_yticks()
    yticks = yticks0[yticks0 != 0]
    ax.set_yticks(yticks)
  
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('k')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    Xticks = ax.get_xticks()
    Xidx = np.argwhere(np.abs(Xticks)>0.01)
    ax.set_xticks(Xticks[Xidx])

    xmax = X["xmax"]
    ax.set_xlim([-xmax,xmax])
    
    #label x-axis according to unit system
    ax.set_xlabel('$\mu_0 H [T]$',horizontalalignment='right', position=(1,25), fontsize=12)

    #label y-axis according to unit system
    if X["mass"].value > 0.0:
        ax.set_ylabel('$M - M_{hys} [Am^2/kg]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
    else: 
        ax.set_ylabel('$M - M_{hys} [Am^2]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)

    
    return X

def data_preprocessing(X):
  
    #parse measurements
    H, Hr, M, Fk, Fj, Ft, dH = parse_measurements(X["fn"])
    Hcal, Mcal, tcal = parse_calibration(X["fn"])
  
    # make a data dictionary for passing large numbers of arguments
    # should unpack in functions for consistency
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["dH"] = dH
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["Ft"] = Ft
    X["Hcal"] = Hcal
    X["Mcal"] = Mcal
    X["tcal"] = tcal

    if X['unit']=='Cgs':
        X = CGS2SI(X)
    
    if X["drift"].value == True:
        X = drift_correction(X)   
  
    #if X["mass"].value > 0.0:
    #    X = mass_normalize(X)
  
    if X["slope"].value < 100:
        X = slope_correction(X)
  
    if X["fpa"].value == True:
        X = remove_fpa(X)
    
    if X["lpa"].value == True:
        X = remove_lpa(X)
    
    #if X["outlier"].value == True:
    #    data = remove_outliers(data)
  
    #extend FORCs
    X = FORC_extend(X)

    #perform lower branch subtraction
    X = lowerbranch_subtract(X)
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(121)
    X = plot_hysteresis(X,ax1)
    ax2 = fig.add_subplot(122)
    X = plot_delta_hysteresis(X,ax2)
    
    outputfile = X["sample"].value+'_HYS.eps'    
    plt.savefig(outputfile, bbox_inches="tight")
    plt.show()
    
    return X

def plot_hysteresis(X,ax):

  #unpack 
    M = X["M"]
    H = X["H"]
    Fk = X["Fk"]
    Fj = X["Fj"]

    #mpl.style.use('seaborn-whitegrid')
    hfont = {'fontname':'STIXGeneral'}

    for i in range(5,int(np.max(Fk)),5):
    
        if X["mass"].value > 0.0: #SI and mass normalized (T and Am2/kg)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value/1000.0),'-k')        
        else: #SI not mass normalized (T and Am2)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)],'-k')        

    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=12,color='k')
    ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('k')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ylim=np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-ylim,ylim])
  
    #ax.set_ylim([-1,1])
    yticks0 = ax.get_yticks()
    yticks = yticks0[yticks0 != 0]
    ax.set_yticks(yticks)
  
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('k')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    xmax = np.max(np.abs(ax.get_xlim()))
    ax.set_xlim([-xmax,xmax])

    #label x-axis
    ax.set_xlabel('$\mu_0 H [T]$',horizontalalignment='right', position=(1,25), fontsize=12)

    #label y-axis according to unit system
    if X["mass"].value > 0.0:
        ax.set_ylabel('$M [Am^2/kg]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
    else: 
        ax.set_ylabel('$M [Am^2]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)

    
    X["xmax"]=xmax
    
    return X

def sample_details(fn):

    sample = fn.split('/')[-1]
    sample = sample.split('.')
    
    if type(sample) is list:
        sample=sample[0]

    units=parse_units(fn)
    mass=parse_mass(fn)
  
    return sample, units, mass

def slope_correction(X):
  
    #unpack
    H = X["H"]
    M = X["M"]
  
    # high field slope correction
    Hidx = H > (X["slope"].value/100) * np.max(H)
    p = np.polyfit(H[Hidx],M[Hidx],1)
    M = M - H*p[0]
  
    #repack
    X["M"]=M
  
    return X

def mass_normalize(X):
  
    X["M"] = X["M"] / (X["mass"].value/1000.) #convert to AM^2/kg
    
    return X

def remove_outliers(X):
    
    """Function to replace "bad" measurements to zero.
    
    Inputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    
    Outputs:
    Fmask: mask, accepted points = 1, rejected points = 0
    R: residuals from fitting process
    Rcrit: critical residual threshold 
    
    """
    #unpack variables
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    
    SF=2 #half width of the smooth (full width = 2SF+1)
    Mst=np.zeros(M.size)*np.nan #initialize output of smoothed magnetizations
    for i in range(M.size): #loop through each measurement
        idx=((Fk==Fk[i]) & (Fj<=Fj[i]+SF) & (Fj>=Fj[i]-SF)) #finding smoothing window in terms of H
        Npts=np.sum(idx) #check enough points are available (may not be the case as edges)
        if Npts>3:
            #create centered quadratic design matrix WRT H
            A = np.concatenate((np.ones(Npts)[:,np.newaxis],\
                                (H[idx]-H[i])[:,np.newaxis],\
                                ((H[idx]-H[i])**2)[:,np.newaxis]),axis=1)
            Mst[i] = np.linalg.lstsq(A,M[idx],rcond=None)[0][0] #regression estimate of M
        else:
            Mst[i] = M[i] #not enough points, so used M

    Mstst=np.zeros(M.size)*np.nan
    for i in range(M.size):
        idx=((Fk<=Fk[i]+SF) & (Fk>=Fk[i]-SF)  & (Fk[i]-Fk+(Fj-Fj[i])==0))
        Npts=np.sum(idx)
        if Npts>3:
            #create centered quadratic design matrix WRT Hr
            A = np.concatenate((np.ones(Npts)[:,np.newaxis],\
                                (Hr[idx]-Hr[i])[:,np.newaxis],\
                                ((Hr[idx]-Hr[i])**2)[:,np.newaxis]),axis=1)
            Mstst[i] = np.linalg.lstsq(A,Mst[idx],rcond=None)[0][0] #regression estimate of Mst
        else: 
            Mstst[i] = Mst[i] #not enough points, so used Mst
            
    
    R = Mstst-Mst #estimated residuals
    Rcrit = np.std(R)*2.5 #set cut-off at 2.5 sigma
    Fmask=np.ones(M.size) #initialize mask
    Fmask[np.abs(R)>Rcrit]=0.0
    
    idx = (np.abs(R)<Rcrit) #points flagged as outliers
    
    #remove points deemed to be outliers
    H = H[idx]
    Hr = Hr[idx]
    M = M[idx]
    Fk = Fk[idx]
    Fj = Fj[idx]
      
    #reset indicies as required
    Fk = Fk - np.min(Fk)+1
  
    Nforc = int(np.max(Fk))
    for i in range(Nforc):
        idx = (Fk == i)
        idx0 = np.argsort(Fj[idx])
        for i in range(idx.size):
            Fj[idx[idx0[i]]] = i+1
    
    #repack variables
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Fj"] = Fj   
    
    return X

def remove_lpa(X):
    
    #unpack
    Fj = X["Fj"]
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Ft = X["Ft"]
    
    #remove last point artifact
    Nforc = int(np.max(Fk))
    W = np.ones(Fk.size)
    
    for i in range(Nforc):      
        Fj_max=np.sum((Fk==i))
        idx = ((Fk==i) & (Fj==Fj_max))
        W[idx]=0.0
    
    idx = (W > 0.5)
    H=H[idx]
    Hr=Hr[idx]
    M=M[idx]
    Fk=Fk[idx]
    Fj=Fj[idx]
    Ft=Ft[idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    
    #repack
    X["Fj"] = Fj
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def remove_fpa(X):
    
    #unpack
    Fj = X["Fj"]
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    Ft = X["Ft"]
    
    #remove first point artifact
    idx=((Fj==1.0))
    H=H[~idx]
    Hr=Hr[~idx]
    M=M[~idx]
    Fk=Fk[~idx]
    Fj=Fj[~idx]
    Ft=Ft[~idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    Fj=Fj-1.
    
    #repack
    X["Fj"] = Fj
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def drift_correction(X):
  
    #unpack
    M = X["M"]
    Mcal = X["Mcal"]    
    Ft = X["Ft"]
    tcal = X["tcal"]
  
    #perform drift correction
    M=M*Mcal[0]/np.interp(Ft,tcal,Mcal,left=np.nan) #drift correction
  
    #repack
    X["M"] = M
  
    return X

def CGS2SI(X):
    
    X["H"] = X["H"]/1E4 #convert Oe into T
    X["M"] = X["M"]/1E3 #convert emu to Am2
      
    return X

def FORC_extend(X):
    
    Ne = 20 #extend up to 20 measurement points backwards
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    for i in range(int(X['Fk'][-1])):
        M0 = M[Fk==i+1]
        H0 = H[Fk==i+1]
        Hr0 = Hr[Fk==i+1][0]
        
        M1 = M0[0] - (np.flip(M0)[1:]-M0[0])
        H1 = H0[0] - (np.flip(H0)[1:]-H0[0])
            
        if M1.size>Ne:
            H1 = H1[-Ne-1:-1]
            M1 = M1[-Ne-1:-1]
        
        if i==0:    
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H1,H0))
            M_new = np.concatenate((M1,M0))
            Hr_new = np.ones(N_new)*Hr0
            Fk_new = np.ones(N_new)
            Fj_new = np.arange(N_new)+1-M1.size
        else:
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H_new,H1,H0))
            M_new = np.concatenate((M_new,M1,M0))
            Hr_new = np.concatenate((Hr_new,np.ones(N_new)*Hr0))
            Fk_new = np.concatenate((Fk_new,np.ones(N_new)+i))
            Fj_new = np.concatenate((Fj_new,np.arange(N_new)+1-M1.size))
            
    #pack up variables
    X['H'] = H_new
    X['Hr'] = Hr_new
    X['M'] = M_new
    X['Fk'] = Fk_new
    X['Fj'] = Fj_new
    
    return X

def lowerbranch_subtract(X):
    """Function to subtract lower hysteresis branch from FORC magnetizations
    
    Inputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    
    Outputs:
    M: lower branch subtracted magnetization [float, SI units]
   
    
    """
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    Hmin = np.min(H)
    Hmax = np.max(H)


    Nbar = 10
    nH = int((Hmax - Hmin)/dH)
    Hi = np.linspace(Hmin,Hmax,nH*50+1)
    Mi = np.empty(Hi.size)
    
    #perform basic loess
    for i in range(Hi.size):
        idx = (H>=Hi[i]-2.5*dH) & (H<=Hi[i]+2.5*dH)
        Mbar = M[idx]
        Hbar = H[idx]
        Fbar = Fk[idx]
        F0 = np.sort(np.unique(Fbar))
        if F0.size>Nbar:
            F0=F0[-Nbar]
        else:
            F0=np.min(F0)
        idx = Fbar>=F0
        
        p = np.polyfit(Hbar[idx],Mbar[idx],2)
        Mi[i] = np.polyval(p,Hi[i])
    
    Hlower = Hi
    Mlower = Mi
    Mcorr=M-np.interp(H,Hlower,Mlower,left=np.nan,right=np.nan) #subtracted lower branch from FORCs via interpolation

    Fk=Fk[~np.isnan(Mcorr)] #remove any nan
    Fj=Fj[~np.isnan(Mcorr)] #remove any nan
    H=H[~np.isnan(Mcorr)] #remove any nan
    Hr=Hr[~np.isnan(Mcorr)] #remove any nan
    M=M[~np.isnan(Mcorr)] #remove any nan
    Mcorr = Mcorr[~np.isnan(Mcorr)] #remove any nan
    
    #repack
    X["H"] = H    
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["DM"] = Mcorr
    
    return X


def lowerbranch_subtract2(X):
    """Function to subtract lower hysteresis branch from FORC magnetizations
    
    Inputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    
    Outputs:
    M: lower branch subtracted magnetization [float, SI units]
   
    
    """
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    Hmin = np.min(H)
    Hmax = np.max(H)
    Hbar = np.zeros(1)
    Mbar = np.zeros(1)

    Nbar = 10
    nH = int((Hmax - Hmin)/dH)
    Hi = np.linspace(Hmin,Hmax,nH*50+1)

    for i in range(Hi.size):
        idx = (H>=Hi[i]-dH/2) & (H<=Hi[i]+dH/2)
    
        H0 = H[idx][-Nbar:]
        M0 = M[idx][-Nbar:]
    
        Hbar = np.concatenate((Hbar,H0))
        Mbar = np.concatenate((Mbar,M0))
    
    Hbar = Hbar[1:]
    Mbar = Mbar[1:]
    Mhat = np.zeros(Hi.size)

    #perform basic loess
    for i in range(Hi.size):
        idx = (Hbar>=Hi[i]-2.5*dH) & (Hbar<=Hi[i]+2.5*dH)
        p = np.polyfit(Hbar[idx],Mbar[idx],2)
        Mhat[i] = np.polyval(p,Hi[i])
    
    Hlower = Hi
    Mlower = Mhat
    Mcorr=M-np.interp(H,Hlower,Mlower,left=np.nan,right=np.nan) #subtracted lower branch from FORCs via interpolation

    Fk=Fk[~np.isnan(Mcorr)] #remove any nan
    Fj=Fj[~np.isnan(Mcorr)] #remove any nan
    H=H[~np.isnan(Mcorr)] #remove any nan
    Hr=Hr[~np.isnan(Mcorr)] #remove any nan
    M=M[~np.isnan(Mcorr)] #remove any nan
    Mcorr = Mcorr[~np.isnan(Mcorr)] #remove any nan
    
    #repack
    X["H"] = H    
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["DM"] = Mcorr
    
    return X
##### END SECTION: PREPROCESSING  #################################################

##### BEGIN SECTION: MODEL FUNCTIONS  #################################################
def model_options(X):
    
    style = {'description_width': 'initial'} #general style settings
    
    #horizontal line widget
    HL = widgets.HTML(value='<hr style="height:3px;border:none;color:#333;background-color:#333;" />')
    

    M_title = widgets.HTML(value='<h3>Select data type:</h3>')
    M_widge = widgets.RadioButtons(options=['Magnetisations', 'Lower branch subtracted'],
                                   value='Lower branch subtracted',
                                   style=style)


    ### Horizontal smoothing ###
    S_title = widgets.HTML(value='<h3>Set smoothing parameters:</h3>')
    
    #SC widgets
    Sc_widge = widgets.IntRangeSlider(
        value=[2,8],
        min=2,
        max=20,
        step=1,
        description='Select $s_c$ range:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.0f',
        style = style
    )

    Sb_widge = widgets.IntRangeSlider(
        value=[2,8],
        min=2,
        max=20,
        step=1,
        description='Select $s_u$ range:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.0f',
        style = style
    )
    
    
    lambda_widge = widgets.FloatRangeSlider(
        value=[0.0,0.08],
        min=0,
        max=0.2,
        step=0.04,
        description='Select $\lambda$ range:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        style = style
    )

    #lambdaSb_widge = widgets.FloatSlider(
    #    value=0.05,
    #    min=0,
    #    max=0.2,
    #    step=0.025,
    #    description='Select $\lambda_{u}$:',
    #    disabled=False,
    #    continuous_update=False,
    #    orientation='horizontal',
    #    readout=True,
    #    readout_format='.3f',
    #    style = style
    #)

    down_title = widgets.HTML(value='<h3>Specify downsampling:</h3>')
    down_widge = widgets.IntSlider(
        value=np.minimum(X['M'].size,5000),
        min=100,
        max=X['M'].size,
        step=1,
        description='Number of points:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style = style
    )
    
    #display number of models to compare
    model_widge = widgets.interactive_output(variforc_array_size, {'SC': Sc_widge, 'SB': Sb_widge, 'L': lambda_widge})


    #combined widget
    DS = VBox([down_title,down_widge])
    SC = VBox([M_title,M_widge,HL,S_title,Sc_widge,Sb_widge,lambda_widge,model_widge])
    
    ### Setup Multiprocessing tab ####################
    #start cluster to test for number of cores
    
    #if 'cluster' in X:
    #    X['cluster'].close()

    #X['cluster'] = LocalCluster()
    #X['ncore'] = len(X['cluster'].workers)
    #X['cluster'].close()
    X['ncore']=8
    
    #header
    dask_title = widgets.HTML(value='<h3>DASK multiprocessing:</h3>')

    #selection widget
    dask_widge=widgets.IntSlider(
        value=X['ncore'],
        min=1,
        max=X['ncore'],
        step=1,
        description='Number of DASK workers:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        style=style
    )
    
    #final multiprocessing widget
    mpl_widge = VBox([dask_title,dask_widge])
    
    ### CONSTRUCT TAB MENU #############
    method_nest = widgets.Tab()
    method_nest.children = [SC,DS,mpl_widge]
    method_nest.set_title(0, 'MODEL ENSEMBLE')
    method_nest.set_title(1, 'DOWNSAMPLING')
    method_nest.set_title(2, 'PROCESSING')
    
    display(method_nest)
    
    ### SETUP OUTPUT ####
    X['Mtype']=M_widge
    X['SC']=Sc_widge
    X['SB']=Sb_widge
    X['lambda']=lambda_widge
    #X['lambdaSB']=lambdaSb_widge
    X['Ndown']=down_widge
    X['workers']=dask_widge
    
    return X

def MK92_weighted_regress(X,y,w,alpha,beta):
    
    #PERFORM WEIGHTED REGRESSION FOLLOWING MACKAY 1992
    w = np.sqrt(w / np.linalg.norm(w))
    w = w/np.sum(w)*y.size
    W = np.diag(w)
    
    XTW = X.T @ W
    M = XTW @ X
    N = np.size(y)
    I = np.eye(np.size(M,axis=1))

    XT_y = np.dot(XTW, y)
    lamb,VhT = linalg.eigh(M)
    lamb = np.real(lamb)
    Vh = VhT.T
    
    for i in range(15):

        ab0=[alpha, beta]
    
        Wprec = alpha*I + beta*M # Bishop eq 3.54
        Wbar = np.dot(VhT,Vh / (lamb + alpha / beta)[:, np.newaxis]) # Bishop eq 3.53
        Wbar = np.dot(Wbar, XT_y) # Bishop eq 3.53 (cont.) 
        
        gamma = np.sum(lamb / (alpha + lamb)) # Bishop eq 3.91
        alpha = gamma / np.maximum(np.sum(np.square(Wbar)),1.0e-10) # Bishop eq 3.91 (avoid division by zero)
        beta = (N - gamma) / np.sum(np.square(y - X @ Wbar)) # Bishop eq 3.95
    
        if np.allclose(ab0,[alpha,beta]):
            break
    
    #Once model is optimized estimate the log-evidence (Bishop equ 3.86)
    M = X.shape[1]
    ln_p = 0.5*M*np.log(alpha)
    ln_p +=0.5*N*np.log(beta)
    ln_p -=0.5*beta*np.sum(np.square(y - X @ Wbar))
    ln_p -=0.5*alpha*np.sum(Wbar**2)
    ln_p -= 0.5*np.linalg.slogdet(Wprec)[1]
    ln_p -= 0.5*N*np.log(2.0*np.pi)
    
    return Wbar, Wprec, ln_p, alpha, beta

def variforc_regression_evidence(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,M,Hc0,Hb0,X):

    # function to test all models
    H0 = Hc0+Hb0
    Hr0 = Hb0-Hc0
    
    rho = np.zeros(Hc0.size)
    Midx = np.zeros(Hc0.size)
       
    for i in range(Hc0.size):
        ln_p = np.zeros(5)
        
        w, idx = vari_weights(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0[i],Hb0[i])
        
        #perform 2nd-order least squares to estimate magnitude of beta
        Aw = X[idx,0:6] * np.sqrt(w[:,np.newaxis])
        Bw = M[idx] * np.sqrt(w)
        p=np.linalg.lstsq(Aw, Bw, rcond=0)[0]
        hat = np.dot(X[idx,0:6],p)
        beta = 1/np.mean((M[idx]-hat)**2)
        alpha = beta*0.0002
        
        #perform 1st regression for model selection    
        _, _, ln_p[0], _, _ = MK92_weighted_regress(X[idx,0:3],M[idx],w,alpha=alpha,beta=beta)
        
        _, _, ln_p[1], _, _ = MK92_weighted_regress(X[idx,0:5],M[idx],w,alpha=alpha,beta=beta)        
        
        Wbar, _, ln_p[2], _, _ = MK92_weighted_regress(X[idx,0:6],M[idx],w,alpha=alpha,beta=beta)
        rho2 = -0.5*Wbar[5]
        
        Wbar, _, ln_p[3], _, _ = MK92_weighted_regress(X[idx,0:10],M[idx],w,alpha=alpha,beta=beta)
        rho3 = -0.5*Wbar[5]-Wbar[8]*H0[i]-Wbar[9]*Hr0[i]

        _, _, ln_p[4], _, _ = MK92_weighted_regress(X[idx,:],M[idx],w,alpha=alpha,beta=beta)
        #rho4[i] = -0.5*Wbar[5]-Wbar[8]*H0[i]-Wbar[9]*Hr0[i]-(Wbar[11]*3*H[i]**2)/2-(Wbar[13]*3*Hr[i]**2)/2-Wbar[12]*2*H[i]*Hr[i]

        Midx[i]=np.argmax(ln_p)        
    
        if Midx[i]==2:
            rho[i]=rho2
        elif Midx[i]>2:
            rho[i]=rho3
            
    return np.column_stack((Midx,rho))



def triangulate_rho(X):

    se = X['se']
    rho = X['rho']
    Hc = X['Hc']
    Hb = X['Hb']
    dH = X['dH']
    
    #PERFORM GRIDDING AND INTERPOLATION FOR FORC PLOT
    X['Hc1'], X['Hc2'], X['Hb1'], X['Hb2'] = measurement_limts(X)
    Hc1 = 0-3*dH
    Hc2 = X['Hc2']
    Hb1 = X['Hb1']-X['Hc2']
    Hb2 = X['Hb2']

    #create grid for interpolation
    Nx = np.ceil((Hc2-Hc1)/dH)+1 #number of points along x
    Ny = np.ceil((Hb2-Hb1)/dH)+1 #number of points along y
    xi = np.linspace(Hc1,Hc2,int(Nx))
    yi = np.linspace(Hb1,Hb2,int(Ny))

    #perform triangluation and interpolation
    triang = tri.Triangulation(Hc, Hb)
    interpolator = tri.LinearTriInterpolator(triang, rho)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = interpolator(Xi, Yi)

    interpolator1 = tri.LinearTriInterpolator(triang, se)
    SEi = interpolator(Xi, Yi)

    X['Hc1'] = Hc1
    X['Xi']=Xi
    X['Yi']=Yi
    X['Zi']=Zi
    X['SEint']=interpolator1
    X['Zint']=interpolator
    
    return X

def plot_model_results_down(X):

    #PLOT FULL MODEL
    ## UNPACK VARIABLE ##
    Midx = X['Midx']
    Hb1 = X['Hb1']-X['Hc2']
    Hb2 = X['Hb2']
    Hc1 = X['Hc1']
    Hc2 = X['Hc2']
    Hc = X['Hci']
    Hb = X['Hbi']
    

    fig = plt.figure(figsize=(12,4.75))

    ##################### BOOTSTRAP PSI ######################
    np.random.seed(999)
    MC_psi = np.zeros(int(2E3))
    for i in range(MC_psi.size):
        bs = np.random.randint(0,Hc.size,Hc.size)
        MC_psi[i] = np.sum((Midx[bs]>0) & (Midx[bs]<4)) / Hc.size
    
    ax2 = fig.add_subplot(1,3,2)
    ax2.hist(MC_psi,bins=25,density=True)
    ax2.set_xlabel('$\psi$',fontsize=12)
    ax2.set_ylabel('Proportion of cases [0-1]',fontsize=12)
    ylim2 = ax2.get_ylim()
    xlim2 = ax2.get_xlim()

    clow = np.percentile(MC_psi,2.5)
    cupp = np.percentile(MC_psi,97.5)
    ax2.plot((clow,clow),ylim2,'-r')
    ax2.plot((cupp,cupp),ylim2,'-r')
    ax2.text(clow-(xlim2[1]-xlim2[0])/12,ylim2[1]/1.1,'$\psi$ (2.5) = {:.3f}'.format(clow),fontsize=12, rotation=90,color='r')
    ax2.text(cupp+(xlim2[1]-xlim2[0])/30,ylim2[1]/1.1,'$\psi$ (97.5) = {:.3f}'.format(cupp),fontsize=12, rotation=90,color='r')

    ax2.set_ylim(ylim2)
    ax2.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')

    ##################### PLOT MODEL ORDER ######################

    #DEFINE COLORMAP
    cseq=[]
    cseq.append((68/255,119/255,170/255,1))
    cseq.append((102/255,204/255,238/255,1))
    cseq.append((34/255,136/255,51/255,1))
    cseq.append((204/255,187/255,68/255,1))
    cseq.append((238/255,102/255,119/255,1))

    ax1 = fig.add_subplot(1,3,1)
    ax1.plot(Hc[Midx==0],Hb[Midx==0],'.',label='$H_1$',markeredgecolor=cseq[0],markerfacecolor=cseq[0],markersize=3)
    ax1.plot(Hc[Midx==1],Hb[Midx==1],'.',label='$H_{2a}$',markeredgecolor=cseq[1],markerfacecolor=cseq[1],markersize=3)
    ax1.plot(Hc[Midx==2],Hb[Midx==2],'.',label='$H_{2b}$',markeredgecolor=cseq[2],markerfacecolor=cseq[2],markersize=3)
    ax1.plot(Hc[Midx==3],Hb[Midx==3],'.',label='$H_3$',markeredgecolor=cseq[3],markerfacecolor=cseq[3],markersize=3)
    ax1.plot(Hc[Midx==4],Hb[Midx==4],'.',label='$H_4$',markeredgecolor=cseq[4],markerfacecolor=cseq[4],markersize=3)
    ax1.set_xlim((0,Hc2))
    ax1.set_ylim((Hb1,Hb2))
    ax1.set_xlabel('$\mu_0H_c$ [T]',fontsize=12)
    ax1.set_ylabel('$\mu_0H_u$ [T]',fontsize=12)
    ax1.set_aspect('equal')
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')
    ax1.tick_params(axis='both',which='minor',direction='out',length=3.5,width=1,color='k')
    ax1.legend(fontsize=12,labelspacing=0,handletextpad=-0.6,loc=4,bbox_to_anchor=(1.035,-0.02),frameon=False,markerscale=2.5)

    ########## PLOT HISTOGRAM #############

    ax3 = fig.add_subplot(1,3,3)
    N, bins, patches = ax3.hist(Midx,bins=(-0.5,0.5,1.5,2.5,3.5,4.5),rwidth=0.8,density=True)

    for i in range(5):
        patches[i].set_facecolor(cseq[i])

    ax3.set_xticks(range(5))    
    ax3.set_xticklabels(('$H_1$', '$H_{2a}$', '$H_{2b}$', '$H_3$', '$H_4$'),size=12)

    ax3.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')
    ax3.set_xlabel('Selected model',fontsize=12)
    ax3.set_ylabel('Proportion of cases [0-1]: $\psi$ = {:.3f}'.format(np.sum((Midx>0) & (Midx<4)) / Hc.size),fontsize=12)
    ax3.set_xlim((-0.5,4.5))

    ##################### OUTPUT PLOTS ######################
    outputfile = X["sample"].value+'_model.eps'
    plt.tight_layout()
    plt.savefig(outputfile)
    plt.show()

    return X

def plot_model_results_full(X):

    #PLOT FULL MODEL
    ## UNPACK VARIABLE ##
    Xi = X['Xi']
    Yi = X['Yi']    
    Zi = X['Zi']
    Midx = X['Midx']
    Hb1 = X['Hb1']-X['Hc2']
    Hb2 = X['Hb2']
    Hc1 = X['Hc1']
    Hc2 = X['Hc2']
    Hc = X['Hc']
    Hb = X['Hb']
    
    fig = plt.figure(figsize=(12,4.75))

    ##################### PLOT FORC ######################
    cmap,vmin,vmax = FORCinel_colormap(Zi)
    ax2 = fig.add_subplot(1,3,2)
    CS = ax2.contourf(Xi, Yi, Zi, 50, cmap = cmap, vmin=vmin, vmax=vmax)       
    cbar2 = fig.colorbar(CS,fraction=0.055, pad=0.05,label='$Am^2$ $T^{-2}$')
    cbar2.ax.tick_params(labelsize=10)
    #cbar2.ax.set_title('$Am^2 T^{-2} (x10^{-6})$',fontsize=10)
    ax2.set_xlim((0,Hc2))
    ax2.set_ylim((Hb1,Hb2))
    ax2.set_ylabel('$\mu_0H_u$ [T]',fontsize=12)
    ax2.set_xlabel('$\mu_0H_c$ [T]',fontsize=12)
    ax2.set_aspect('equal')
    ax2.minorticks_on()
    ax2.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')
    ax2.tick_params(axis='both',which='minor',direction='out',length=3.5,width=1,color='k')
    ax2.plot((0,Hc2),(Hb1,X['Hb1']),'--k')

    ##################### PLOT MODEL ORDER ######################

    #DEFINE COLORMAP
    cseq=[]
    cseq.append((68/255,119/255,170/255,1))
    cseq.append((102/255,204/255,238/255,1))
    cseq.append((34/255,136/255,51/255,1))
    cseq.append((204/255,187/255,68/255,1))
    cseq.append((238/255,102/255,119/255,1))

    ax1 = fig.add_subplot(1,3,1)
    ax1.plot(Hc[Midx==0],Hb[Midx==0],'.',label='$H_1$',markeredgecolor=cseq[0],markerfacecolor=cseq[0],markersize=3)
    ax1.plot(Hc[Midx==1],Hb[Midx==1],'.',label='$H_{2a}$',markeredgecolor=cseq[1],markerfacecolor=cseq[1],markersize=3)
    ax1.plot(Hc[Midx==2],Hb[Midx==2],'.',label='$H_{2b}$',markeredgecolor=cseq[2],markerfacecolor=cseq[2],markersize=3)
    ax1.plot(Hc[Midx==3],Hb[Midx==3],'.',label='$H_3$',markeredgecolor=cseq[3],markerfacecolor=cseq[3],markersize=3)
    ax1.plot(Hc[Midx==4],Hb[Midx==4],'.',label='$H_4$',markeredgecolor=cseq[4],markerfacecolor=cseq[4],markersize=3)
    ax1.set_xlim((0,Hc2))
    ax1.set_ylim((Hb1,Hb2))
    ax1.set_xlabel('$\mu_0H_c$ [T]',fontsize=12)
    ax1.set_ylabel('$\mu_0H_u$ [T]',fontsize=12)
    ax1.set_aspect('equal')
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')
    ax1.tick_params(axis='both',which='minor',direction='out',length=3.5,width=1,color='k')
    ax1.legend(fontsize=12,labelspacing=0,handletextpad=-0.6,loc=4,bbox_to_anchor=(1.035,-0.02),frameon=False,markerscale=2.5)
    cbar1 = fig.colorbar(CS,fraction=0.055, pad=0.05)
    cbar1.ax.tick_params(labelsize=10)
    cbar1.remove()

    ########## PLOT HISTOGRAM #############

    ax3 = fig.add_subplot(1,3,3)
    N, bins, patches = ax3.hist(Midx,bins=(-0.5,0.5,1.5,2.5,3.5,4.5),rwidth=0.8,density=True)

    for i in range(5):
        patches[i].set_facecolor(cseq[i])

    ax3.set_xticks(range(5))    
    ax3.set_xticklabels(('$H_1$', '$H_{2a}$', '$H_{2b}$', '$H_3$', '$H_4$'),size=12)

    ax3.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')
    ax3.set_xlabel('Selected model',fontsize=12)
    ax3.set_ylabel('Proportion of cases [0-1]: $\psi$ = {:.3f}'.format(np.sum((Midx>0) & (Midx<4)) / Hc.size),fontsize=12)
    ax3.set_xlim((-0.5,4.5))

    ##################### OUTPUT PLOTS ######################
    outputfile = X["sample"].value+'_model.eps'
    plt.tight_layout()
    plt.savefig(outputfile)
    plt.show()

    return X

def measurement_limts(X):
    """Function to find measurement limits and conver units if required

    Inputs:
    file: name of data file (string)    


    Outputs:
    Hc1: minimum Hc
    Hc2: maximum Hc
    Hb1: minimum Hb
    Hb2: maximum Hb
    """    
    
    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(X["fn"],string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(X["fn"],string)

    string='Hc2' #upper Hc value for the FORC box
    Hc2=parse_header(X["fn"],string)

    string='Hc1' #lower Hc value for the FORC box
    Hc1=parse_header(X["fn"],string)

    if X['unit']=='Cgs': #convert CGS to SI
        Hc2=Hc2/1E4 #convert from Oe to T
        Hc1=Hc1/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T  

    return Hc1, Hc2, Hb1, Hb2

def FORCinel_colormap(Z):

    #setup initial colormap assuming that negative range does not require extension
    cdict = {'red':     ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         #(0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                         (0.3193,  102/255, 102/255),
                       (0.563,  204/255, 204/255),
                       (0.6975,  204/255, 204/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255)),

            'green':   ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         #(0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  178/255, 178/255),
                        (0.563,  204/255, 204/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  102/255, 102/255),
                       (0.9748,  25/255, 25/255),
                       (1.0, 25/255, 25/255)),

             'blue':   ((0.0,  255/255, 255/255),
                         (0.1387,  255/255, 255/255),
                         #(0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  102/255, 102/255),
                        (0.563,  76/255, 76/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255))}

    if np.abs(np.min(Z))<=np.max(Z)*0.19: #negative extension is not required
        #cmap = LinearSegmentedColormap('forc_cmap', cdict)
        vmin = -np.max(Z)*0.19
        vmax = np.max(Z)
    else: #negative extension is required
        vmin=np.min(Z)
        vmax=np.max(Z)        
    
    anchors = np.zeros(9)
    anchors[1]=(-0.015*vmax-vmin)/(vmax-vmin)
    anchors[2]=(0.015*vmax-vmin)/(vmax-vmin)
    anchors[3]=(0.19*vmax-vmin)/(vmax-vmin)
    anchors[4]=(0.48*vmax-vmin)/(vmax-vmin)
    anchors[5]=(0.64*vmax-vmin)/(vmax-vmin)
    anchors[6]=(0.80*vmax-vmin)/(vmax-vmin)
    anchors[7]=(0.97*vmax-vmin)/(vmax-vmin)
    anchors[8]=1.0

    Rlst = list(cdict['red'])
    Glst = list(cdict['green'])
    Blst = list(cdict['blue'])

    for i in range(9):
        Rlst[i] = tuple((anchors[i],Rlst[i][1],Rlst[i][2]))
        Glst[i] = tuple((anchors[i],Glst[i][1],Glst[i][2]))
        Blst[i] = tuple((anchors[i],Blst[i][1],Blst[i][2]))
        
    cdict['red'] = tuple(Rlst)
    cdict['green'] = tuple(Glst)
    cdict['blue'] = tuple(Blst)

    cmap = LinearSegmentedColormap('forc_cmap', cdict)

    return cmap, vmin, vmax

def variforc_array_size(SC,SB,L): #array of variforc smoothing parameter

    Sc_min = SC[0]
    Sc_max = SC[1]
    Sb_min = SB[0]
    Sb_max = SB[1]
    Lambda_min = L[0]
    Lambda_max = L[1]
    num = 8

    Sc = np.unique(np.round(np.geomspace(Sc_min, Sc_max, num=num)))
    Sb = np.unique(np.round(np.geomspace(Sb_min, Sb_max, num=num)))
    #Lambda = np.unique(np.linspace(Lambda_min, Lambda_max, num=int(num*0.5)))
    Lambda = np.arange(Lambda_min, Lambda_max+0.001,0.04)

    if (Lambda_max > 0):
        [Sc0,Sc1,Sb0,Sb1,L]=np.meshgrid(Sc,Sc,Sb,Sb,Lambda)
        Sc0 = np.matrix.flatten(Sc0)
        Sc1 = np.matrix.flatten(Sc1)
        Sb0 = np.matrix.flatten(Sb0)
        Sb1 = np.matrix.flatten(Sb1)
        L = np.matrix.flatten(L)
    else:
        [Sc0,Sc1,Sb0,Sb1]=np.meshgrid(Sc,Sc,Sb,Sb)
        Sc0 = np.matrix.flatten(Sc0)
        Sc1 = np.matrix.flatten(Sc1)
        Sb0 = np.matrix.flatten(Sb0)
        Sb1 = np.matrix.flatten(Sb1)
        L = np.zeros((Sb1.size,3))    

    idx = ((Sc1>=Sc0) & (Sb1>=Sb0))

    results = widgets.HTML(value='<h5>Number of VARIFORC models to compare = {:}</h5>'.format(int(np.sum(idx))))
    display(results)

def profile_options(X):
    Hb1 = X['Hb1']-X['Hc2']
    Hb2 = X['Hb2']
    Hc1 = np.maximum(X['Hc1'],0)
    Hc2 = X['Hc2']
    style = {'description_width': 'initial'} #general style settings
    
    HL = widgets.HTML(value='<hr style="height:3px;border:none;color:#333;background-color:#333;" />')
    
    P_title = widgets.HTML(value='<h3>Select profile type:</h3>')
    P_widge = widgets.RadioButtons(options=[('Horizontal profile',0), ('Vertical profile',1)],
                                       value=0,
                                       style=style)
    
    H_title = widgets.HTML(value='<h4>Horizontal profile specification:</h4>')
    x_Hb_widge = widgets.FloatSlider(
        value=0.0,
        min=Hb1,
        max=Hb2,
        step=0.001,
        description='$\mu_0H_u$ [T]',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
        layout={'width': '350px'},
        style = style
    )
    
    x_Hc_widge = widgets.FloatRangeSlider(
        value=[Hc1,Hc2],
        min=Hc1,
        max=Hc2,
        step=0.001,
        description='$\mu_0H_c$ [T]',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
        layout={'width': '350px'},
        style = style
    )
    
    V_title = widgets.HTML(value='<h4>Vertical profile specification:</h4>')
    y_Hc_widge = widgets.FloatSlider(
        value=(Hc1+Hc2)/2.0,
        min=Hc1,
        max=Hc2,
        step=0.001,
        description='$\mu_0H_c$ [T]',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
        layout={'width': '350px'},
        style = style
    )
    
    y_Hb_widge = widgets.FloatRangeSlider(
        value=[Hb1,Hb2],
        min=Hb1,
        max=Hb2,
        step=0.001,
        description='$\mu_0H_u$ [T]',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
        layout={'width': '350px'},
        style = style
    )
    
    profile_widge = VBox([P_title,P_widge,HL,H_title,x_Hb_widge,x_Hc_widge, \
                         HL,V_title,y_Hc_widge,y_Hb_widge])
    
    profile_nest = widgets.Tab()
    profile_nest.children = [profile_widge]
    profile_nest.set_title(0, 'PLOT PROFILES')
    display(profile_nest)   
    
    X['P_widge'] = P_widge
    X['x_Hb_widge'] = x_Hb_widge
    X['x_Hc_widge'] = x_Hc_widge
    X['y_Hc_widge'] = y_Hc_widge
    X['y_Hb_widge'] = y_Hb_widge

    return X

def profile_plot(X):

    if X['P_widge'].value==0:
        X = x_profile(X,X['x_Hc_widge'].value,X['x_Hb_widge'].value)
    else:
        X = y_profile(X,X['y_Hc_widge'].value,X['y_Hb_widge'].value)
    
    return X

def x_profile(X,Hc,Hb):

    Hc1, Hc2 = Hc[0], Hc[1]

    dH = X['dH']
    NH = int(np.sqrt((Hc2-Hc1)**2)/dH)
    Hc0 = np.linspace(Hc1,Hc2,NH)
    Hb0 = np.linspace(Hb,Hb,NH)
    
    rho_int = X['Zint'](Hc0,Hb0)
    coef = sps.norm.ppf(0.025/np.sum(rho_int.mask==False))
    CI_int = X['SEint'](Hc0,Hb0)*coef

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)
    
    if X['mass'].value>0.0:
        ax1.plot(Hc0,rho_int/(X['mass'].value/1000.0),color='k')
        ax1.fill_between(Hc0, (rho_int-CI_int)/(X['mass'].value/1000.0), (rho_int+CI_int)/(X['mass'].value/1000.0),color='lightgrey')
        ax1.set_ylabel('$Am^2$ $T^{-2} kg^{-1}$',fontsize=12)
    else:
        ax1.plot(Hc0,rho_int,color='k')
        ax1.fill_between(Hc0, (rho_int-CI_int), (rho_int+CI_int),color='lightgrey')
        ax1.set_ylabel('$Am^2$ $T^{-2}$',fontsize=12)

    ax1.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')
    ax1.tick_params(axis='both',which='minor',direction='out',length=3.5,width=1,color='k')

    ax1.set_xlabel('$\mu_0H_c$ [T]',fontsize=12)
    ax1.minorticks_on()
    
    outputfile = X['sample'].value+'_Hc_PROFILE.eps'
    plt.savefig(outputfile, dpi=300, bbox_inches="tight")
    plt.show
    
    return X

def y_profile(X,Hc,Hb):

    Hb1, Hb2 = Hb[0], Hb[1]

    dH = X['dH']
    NH = int(np.sqrt((Hb2-Hb1)**2)/dH)
    Hc0 = np.linspace(Hc,Hc,NH)
    Hb0 = np.linspace(Hb1,Hb2,NH)
    
    rho_int = X['Zint'](Hc0,Hb0)
    coef = sps.norm.ppf(0.025/np.sum(rho_int.mask==False))
    CI_int = X['SEint'](Hc0,Hb0)*coef

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1,1,1)

    if X['mass'].value>0.0:
        ax1.plot(Hb0,rho_int/(X['mass'].value/1000.0),color='k')
        ax1.fill_between(Hb0, (rho_int-CI_int)/(X['mass'].value/1000.0), (rho_int+CI_int)/(X['mass'].value/1000.0),color='lightgrey')
        ax1.set_ylabel('$Am^2$ $T^{-2} kg^{-1}$',fontsize=12)
    else:
        ax1.plot(Hb0,rho_int,color='k')
        ax1.fill_between(Hb0, (rho_int-CI_int), (rho_int+CI_int),color='lightgrey')
        ax1.set_ylabel('$Am^2$ $T^{-2}$',fontsize=12)
    
    ax1.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')
    ax1.tick_params(axis='both',which='minor',direction='out',length=3.5,width=1,color='k')

    ax1.set_xlabel('$\mu_0H_b$ [T]',fontsize=12)
    ax1.minorticks_on()
    
    outputfile = X['sample'].value+'_Hu_PROFILE.eps'
    plt.savefig(outputfile, dpi=300, bbox_inches="tight")
    plt.show
    
    return X

def variforc_array(X): #array of variforc smoothing parameter

    Sc_min = X['SC'].value[0]
    Sc_max = X['SC'].value[1]
    Sb_min = X['SB'].value[0]
    Sb_max = X['SB'].value[1]
    Lambda_min = X['lambda'].value[0]
    Lambda_max = X['lambda'].value[1]
    num = 8

    Sc = np.unique(np.round(np.geomspace(Sc_min, Sc_max, num=num)))
    Sb = np.unique(np.round(np.geomspace(Sb_min, Sb_max, num=num)))
    #Lambda = np.linspace(Lambda_min, Lambda_max, num=int(num*0.5))
    Lambda = np.arange(Lambda_min, Lambda_max+0.001,0.04)


    if (Lambda_max > 0):
        [Sc0,Sc1,Sb0,Sb1,L]=np.meshgrid(Sc,Sc,Sb,Sb,Lambda)
        Sc0 = np.matrix.flatten(Sc0)
        Sc1 = np.matrix.flatten(Sc1)
        Sb0 = np.matrix.flatten(Sb0)
        Sb1 = np.matrix.flatten(Sb1)
        L = np.matrix.flatten(L)
    else:
        [Sc0,Sc1,Sb0,Sb1]=np.meshgrid(Sc,Sc,Sb,Sb)
        Sc0 = np.matrix.flatten(Sc0)
        Sc1 = np.matrix.flatten(Sc1)
        Sb0 = np.matrix.flatten(Sb0)
        Sb1 = np.matrix.flatten(Sb1)
        L = np.zeros((Sb1.size,3))    

    idx = ((Sc1>=Sc0) & (Sb1>=Sb0))
    Sc0 = Sc0[idx]
    Sc1 = Sc1[idx]
    Sb0 = Sb0[idx]
    Sb1 = Sb1[idx]
    L = L[idx]

    Sp = np.column_stack((Sc0,Sc1,Sb0,Sb1,L))
    Nsplit = 30
    Sp_i = np.array_split(Sp,Nsplit)
    
    X['Sp'] = Sp
    X['Sp_i'] = Sp_i
    
    return X

def in_window(P,Hc_i,Hb_i):
    return (Hc_i>=P['Hc1']) & (Hc_i<=P['Hc2']) & (Hb_i<=P['Hb2']) & (Hb_i>=(P['Hb1']-(P['Hc2']-P['Hc1'])+(Hc_i-P['Hc1'])))

def FORC_regress0(X,y):

    N = y.size    
    XT = X.T
    XTX = XT @ X
    ydev = np.sum((y-np.mean(y))**2)

    B1 = inv_jit(XTX[0:3,0:3]) @ XT[0:3,:] @ y 
    ssr1 = np.sum((X[:,0:3] @ B1 - y)**2)
    r2_1 = 1 - ssr1 / ydev
    BF1, B = BayesFactor3_i(N,r2_1)
    
    if (BF1>1.0):
        B2 = inv_jit(XTX[0:6,0:6]) @ XT[0:6,:] @ y
        ssr2 = np.sum((X[:,0:6] @ B2 - y)**2)
        r2_2 = 1 - ssr2 / ydev
        BF2, _ = BayesFactor3_i(N,r2_2,2,B)    
    else:
        return 0.0
    
    if (BF2>1.0*BF1):
        B3 = inv_jit(XTX) @ XT @ y
        ssr3 = np.sum((X @ B3 - y)**2)
        r2_3 = 1 - ssr3 / ydev    
        BF3, _ = BayesFactor3_i(N,r2_3,3,B)
    else:
        return 1.0
    
    if (BF3>1.0*BF2):
        return 3.0
    else:
        return 2.0

def FORC_regress0_full(X,y):

    N = y.size    
    XT = X.T
    XTX = XT @ X
    ydev = np.sum((y-np.mean(y))**2)

    # Test 1st order model
    status = True
    BF1 = 0.0
    try:
        iv = inv_jit(XTX[0:3,0:3])
        #B1 = np.linalg.inv(XTX[0:3,0:3]) @ XT[0:3,:] @ y
        #ssr1 = np.sum((X[:,0:3] @ B1 - y)**2) 
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        ssr1 = np.sum((X[:,0:3] @ (iv @ XT[0:3,:] @ y) - y)**2) 
        r2_1 = 1 - ssr1 / ydev
        BF1, B = BayesFactor3_i(N,r2_1)
        
    # Test 2nd order model    
    status = True
    BF2 = 0.0
    try:
        iv = inv_jit(XTX[0:6,0:6])
        #B2 = np.linalg.inv(XTX[0:6,0:6]) @ XT[0:6,:] @ y
        #ssr2 = np.sum((X[:,0:6] @ (np.linalg.inv(XTX[0:6,0:6]) @ XT[0:6,:] @ y) - y)**2)
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        ssr2 = np.sum((X[:,0:6] @ (iv @ XT[0:6,:] @ y) - y)**2)
        r2_2 = 1 - ssr2 / ydev
        BF2, _ = BayesFactor3_i(N,r2_2,2,B)    
    
    # Test 3rd order model
    status = True
    BF3 = 0.0      
    try:
        iv = inv_jit(XTX)
        #B3 = np.linalg.inv(XTX) @ XT @ y
        #ssr3 = np.sum((X @ (np.linalg.inv(XTX) @ XT @ y) - y)**2)
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        ssr3 = np.sum((X @ (iv @ XT @ y) - y)**2)
        r2_3 = 1 - ssr3 / ydev   
        BF3, _ = BayesFactor3_i(N,r2_3,3,B)
    
    return np.array((0,BF1,BF2,BF3))

def BayesFactor3_i(N,R2,order=1,B=0):
    
    a = -3/4
    if (order==1):
        A = -1.3862943611198906
        B = gammaln((N-1)/2)
        BF = A+gammaln((N-2-1)/2)-B+(-(N-2-1)/2+a+1)*np.log(1-R2)
    elif (order==2):
        A = -0.8128078577831402
        BF = A+gammaln((N-5-1)/2)-B+(-(N-5-1)/2+a+1)*np.log(1-R2)
    else:
        A = 1.5205488938776595
        BF = A+gammaln((N-9-1)/2)-B+(-(N-9-1)/2+a+1)*np.log(1-R2)

    
    return BF, B

def variforc_BF0_full(Xlsq,M,Hc,Hb,dH,Hc0,Hb0,sc0,sc1,lamb_sc,sb0,sb1,lamb_sb):
    
    Npts = Hc0.size
    BF = np.zeros((Npts,4))

    for i in range(Npts):
        idx = vari_pts(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0[i],Hb0[i])
        BF[i,:] = FORC_regress0_full(Xlsq[idx,:],M[idx])

    return BF
    
def dask_BF0(X,S,Didx,Mswitch):
    
    Xlsq = X['Xlsq']
    if (Mswitch<0.5):
        M = X['M']
    else:
        M = X['DM']

    Hc = X['Hc']
    Hb = X['Hb']
    dH = X['dH']
    Hc0 = X['Hc0']
    Hc0 = Hc0[Didx]
    Hb0 = X['Hb0']
    Hb0 = Hb0[Didx]
    
    sc0=S[:,0]
    sc1=S[:,1]
    sb0=S[:,2]
    sb1=S[:,3]
    lamb=S[:,4]

    Npts = len(S)
    L = np.zeros((Npts,5))    
    
    for i in range(Npts):
        BF = variforc_BF0_full(Xlsq,M,Hc,Hb,dH,Hc0,Hb0,sc0[i],sc1[i],lamb[i],sb0[i],sb1[i],lamb[i])
        Li = np.argmax(BF,axis=1) 
        L[i,0] = np.sum(Li==0)
        L[i,1] = np.sum(Li==1)
        L[i,2] = np.sum(Li==2) 
        L[i,3] = np.sum(Li==3)

        BFn = BF - np.array([0,0,np.log(3.2),0])
        Li = np.argmax(BFn,axis=1)
        L[i,4] = np.sum(Li==2) 

    return L

def weighted_regression(X,L,Mswitch):
    
    i0 = np.argmax(L[:,2])
    sc0 = X['Sp'][i0,0]
    sc1 = X['Sp'][i0,1]
    sb0 = X['Sp'][i0,2]
    sb1 = X['Sp'][i0,3] 
    lamb = X['Sp'][i0,4]
       
    Hc = X['Hc']
    Hb = X['Hb']
    H = X['H']
    Hr = X['Hr']   
    
    dH = X['dH']
    if (Mswitch<0.5):
        M = X['Mnorm']
    else:
        M = X['DMnorm']    
    
    Xlsq = X['Xlsq']
    
    rho = np.zeros(Hc.size) 
    se = np.zeros(Hc.size)

    for i in range(Hc.size):
        
        w, idx = vari_weights(sc0,sc1,lamb,sb0,sb1,lamb,Hc,Hb,dH,Hc[i],Hb[i])
        
        #perform 2nd-order least squares to estimate rho and variance-covariance matrix
        Aw = Xlsq[idx,0:6] * np.sqrt(w[:,np.newaxis])
        Bw = M[idx] * np.sqrt(w)
        p=np.linalg.lstsq(Aw, Bw, rcond=0)
        if p[1].size==1:
            rho2 = (p[0][3]-p[0][4])/4
            sigma2 = p[1]/(Bw.size-6)
            S2 = sigma2 * inv_jit(np.dot(Aw.T, Aw))
            A2=np.zeros(6)[:,np.newaxis]
            A2[3]=0.25
            A2[4]=-0.25
            se2 = np.sqrt(A2.T @ S2 @ A2)
 
        #perform 3rd-order least squares to estimate rho and variance-covariance matrix
        Aw = Xlsq[idx,:] * np.sqrt(w[:,np.newaxis])
        p=np.linalg.lstsq(Aw, Bw, rcond=0)
        rho3 = p[0][3]/4 - p[0][4]/4 + (3*p[0][6]*Hc[i])/4 - (3*p[0][7]*Hb[i])/4 + (p[0][8]*Hb[i])/4 - (p[0][9]*Hc[i])/4
        if p[1].size==1:
            sigma2 = p[1]/(Bw.size-10)
            S3 = sigma2 * inv_jit(np.dot(Aw.T, Aw))
            A3=np.zeros(10)[:,np.newaxis]
            A3[3]=0.25
            A3[4]=-0.25
            A3[6]=3*Hc[i]/4
            A3[7]=-3*Hb[i]/4
            A3[8]=Hb[i]/4
            A3[9]=-Hc[i]/4
            se3 = np.sqrt(A3.T @ S3 @ A3)

        rho[i] = rho2*X['Pr'][i,2]+rho3*X['Pr'][i,3]
        se[i] = se2*X['Pr'][i,2]+se3*X['Pr'][i,3]

    X['rho'] = rho
    X['se'] = se

    return X


def calculate_model(X):

    if ('client' in X) == False: #start DASK if required
        c = LocalCluster(n_workers=X['workers'].value)
        X['client'] = Client(c)

    if X['Mtype'].value=='Magnetisations':
        Mswitch = 0
    else:
        Mswitch = 1
    
    X['Mswitch'] = Mswitch
    
    # Create variables
    M = X['M']
    DM = X['DM']
    X['Hc'] = 0.5*(X['H']-X['Hr'])
    X['Hb'] = 0.5*(X['H']+X['Hr'])
    X['Mnorm'] = M/np.max(M)
    X['DMnorm'] = DM/np.max(DM)
    X['Xlsq'] = np.column_stack((np.ones((X['Hc'].size,1)),X['Hc'],X['Hb'],X['Hc']**2,X['Hb']**2,X['Hc']*X['Hb'],X['Hc']**3,X['Hb']**3,X['Hc']**2*X['Hb'],X['Hc']*X['Hb']**2))

    Hc1, Hc2, Hb1, Hb2 = measurement_limts(X)
    P = {}
    P['Hc1'] = Hc1
    P['Hc2'] = Hc2
    P['Hb1'] = Hb1
    P['Hb2'] = Hb2

    idx = np.argwhere(in_window(P,X['Hc'],X['Hb'])==True)
    X['Hc0'] = X['Hc'][idx]
    X['Hb0'] = X['Hb'][idx]

    #scatter variables
    D = {}
    D['Xlsq'] = X['Xlsq']
    D['M'] = X['Mnorm']
    D['DM'] = X['DMnorm']
    D['Hc'] = X['Hc']
    D['Hb'] = X['Hb']
    D['dH'] = X['dH']
    D['Hc0'] = X['Hc0']
    D['Hb0'] = X['Hb0']
    X['Ds'] = X['client'].scatter(D,broadcast=True)

    Ntot = np.size(X['Hc0'])
    np.random.seed(999)
    Didx = np.sort(np.random.choice(Ntot, X['Ndown'].value, replace=False)) #downsampled indicies

    X = variforc_array(X) #get smoothing parameter

    jobs = []
    for i in range(len(X['Sp_i'])):
        job = X['client'].submit(dask_BF0,X['Ds'],X['Sp_i'][i],Didx,Mswitch)
        jobs.append(job)
    
    results = X['client'].gather(jobs)

    L = results[0]
    for i in range(len(results)-1):
        L=np.concatenate((L,results[i+1]))
    
    X['L'] = L
    
    #Make results plots
    i0 = np.argmax(L[:,2])
    if (Mswitch<0.5):
        BF = variforc_BF0_full(X['Xlsq'],X['Mnorm'],X['Hc'],X['Hb'],X['dH'],X['Hc'],X['Hb'],X['Sp'][i0,0],X['Sp'][i0,1],X['Sp'][i0,4],X['Sp'][i0,2],X['Sp'][i0,3],X['Sp'][i0,4])
    else:
        BF = variforc_BF0_full(X['Xlsq'],X['DMnorm'],X['Hc'],X['Hb'],X['dH'],X['Hc'],X['Hb'],X['Sp'][i0,0],X['Sp'][i0,1],X['Sp'][i0,4],X['Sp'][i0,2],X['Sp'][i0,3],X['Sp'][i0,4])
    
    #BF[BF<0.0]=0.0
    X['BF']=BF  
    #X['Pr']=BF/np.sum(BF,axis=1)[:,np.newaxis]
    X['Pr']=np.exp(BF-logsumexp(BF,axis=1)[:,np.newaxis])
    Lpt = np.argmax(BF,axis=1)
    Lpt[np.max(X['BF'],axis=1)<1]=0
    
    X = plot_model_selection(X,Lpt[idx],i0)
    X = weighted_regression(X,L,Mswitch) #calculate rho for optimal smooth
    X = triangulate_rho(X) #triangulate rho for plotting

    return X

def plot_model_selection(X,Lpt,i0):
    
    R_out = widgets.HTML(value='<h3>Model Comparison Results</h3>')
    HL = widgets.HTML(value='<hr style="height:3px;border:none;color:#333;background-color:#333;" />')
    H_out = widgets.HTML(value='<h4>Optimal VARIFORC Smoothing Factors</h4>')
    sc0_out = widgets.Label(value='Optimal $Sc_0$ = {:}'.format(int(X['Sp'][i0,0])))
    sc1_out = widgets.Label(value='Optimal $Sc_1$ = {:}'.format(int(X['Sp'][i0,1])))
    sb0_out = widgets.Label(value='Optimal $Sb_0$ = {:}'.format(int(X['Sp'][i0,2])))
    sb1_out = widgets.Label(value='Optimal $Sb_1$ = {:}'.format(int(X['Sp'][i0,3])))
    lam_out = widgets.Label(value='Optimal $\lambda$ = {:.2f}'.format(X['Sp'][i0,4]))
    T_out = widgets.HTML(value='<h4>Distribution of model performance</h4>')
    
    display(VBox([R_out,HL,H_out,sc0_out,sc1_out,sb0_out,sb1_out,lam_out,HL,T_out]))

    L = X['L']
    i0 = np.argmax(L[:,2])
    a = L[:,0]+L[:,1]
    b = L[:,2]
    c = L[:,3]
    
    fig, ax = plt.subplots()
    ax.plot(np.array((0,1,0.5,0)),np.array((0,0,np.sqrt(3)/2,0)),'k')
    ax.set_aspect(1.0)
    ax.set_axis_off()
    ax.set_axis_off()
    
    a0 = np.arange(0,1.05,0.2)
    for i in a0:
        a1 = np.array((i,i))
        b1 = np.array((0,1-i))
        c1 = np.array((1-i,0))
        ax.plot(0.5*(2*b1+c1)/(a1+b1+c1),np.sqrt(3)/2*c1/(a1+b1+c1),'--k',linewidth=0.5)
        
    b0 = np.arange(0,1.05,0.2)
    for i in b0:
        b1 = np.array((i,i))
        a1 = np.array((0,1-i))
        c1 = np.array((1-i,0)) 
        ax.plot(0.5*(2*b1+c1)/(a1+b1+c1),np.sqrt(3)/2*c1/(a1+b1+c1),'--k',linewidth=0.5)
        
    c0 = np.arange(0,1.05,0.2)
    for i in c0:
        c1 = np.array((i,i))
        a1 = np.array((0,1-i))
        b1 = np.array((1-i,0)) 
        ax.plot(0.5*(2*b1+c1)/(a1+b1+c1),np.sqrt(3)/2*c1/(a1+b1+c1),'--k',linewidth=0.5)
            
    ax.plot(0.5*(2*b+c)/(a+b+c),np.sqrt(3)/2*c/(a+b+c),'ok')
    ax.plot(0.5*(2*b[i0]+c[i0])/(a[i0]+b[i0]+c[i0]),np.sqrt(3)/2*c[i0]/(a[i0]+b[i0]+c[i0]),'or')
    
    a = np.array((1))
    b = np.array((0))
    c = np.array((0))
    ax.text(0.5*(2*b+c)/(a+b+c)-0.175,np.sqrt(3)/2*c/(a+b+c)-0.06,'Overfitting',fontsize=12)
       
    a = np.array((0))
    b = np.array((1))
    c = np.array((0))    
    ax.text(0.5*(2*b+c)/(a+b+c)-0.1,np.sqrt(3)/2*c/(a+b+c)-0.06,'Optimal',fontsize=12)
    
    a = np.array((0))
    b = np.array((0))
    c = np.array((1)) 
    ax.text(0.5*(2*b+c)/(a+b+c)-0.15,np.sqrt(3)/2*c/(a+b+c)+0.03,'Underfitting',fontsize=12)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,np.sqrt(3)/2))
    outputfile = X['sample'].value+'_TERNARY.eps'
    plt.savefig(outputfile, dpi=300, bbox_inches="tight")
    plt.show()
    
    F_out = widgets.HTML(value='<h4>Optimal VARIFORC model</h4>')
    display(VBox([HL,F_out]))

    fig, ax = plt.subplots()
    cseq=[]
    cseq.append((0/255,0/255,0/255,1))
    cseq.append((86/255,180/255,233/255,1))
    cseq.append((213/255,94/255,0/255,1))
    
    ax.plot(X['Hc0'][Lpt<=1],X['Hb0'][Lpt<=1],'.',label='$Overfit$',markeredgecolor=cseq[0],markerfacecolor=cseq[0],markersize=3)
    ax.plot(X['Hc0'][Lpt==2],X['Hb0'][Lpt==2],'.',label='$Optimal$',markeredgecolor=cseq[1],markerfacecolor=cseq[1],markersize=3)
    ax.plot(X['Hc0'][Lpt==3],X['Hb0'][Lpt==3],'.',label='$Underfit$',markeredgecolor=cseq[2],markerfacecolor=cseq[2],markersize=3)
    
    ax.set_aspect(1.0)    
    Hc1, Hc2, Hb1, Hb2 = measurement_limts(X)    
    Hb1 = Hb1-Hc2

    ax.set_xlim((np.maximum(0,Hc1),Hc2))
    ax.set_ylim((Hb1,Hb2))
    ax.set_xlabel('$\mu_0H_c$ [T]',fontsize=12)
    ax.set_ylabel('$\mu_0H_u$ [T]',fontsize=12)
    ax.set_aspect('equal')
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,color='k',labelsize='12')
    ax.tick_params(axis='both',which='minor',direction='out',length=3.5,width=1,color='k')   
    ax.legend(fontsize=10,labelspacing=0,handletextpad=-0.6,loc=4,bbox_to_anchor=(1.05,-0.02),frameon=False,markerscale=2.5);
    ax.set_title('$\psi$ = {:.2f}'.format(np.sum(Lpt==2)/np.size(Lpt)));
    outputfile = X['sample'].value+'_ORDER.eps'
    plt.savefig(outputfile, dpi=150, bbox_inches="tight")
    plt.show()

    return X

""" def calculate_model2(X):

    if ('client' in X) == False: #start DASK if required
        c = LocalCluster(n_workers=X['workers'].value)
        X['client'] = Client(c)
    
    H = X['H']
    Hr = X['Hr']
    dH = X['dH']

    #form top-level design matrix
    X0 = np.column_stack((np.ones(H.size),H,Hr,H**2,Hr**2,H*Hr,H**3,Hr**3,H**2*Hr,H*Hr**2,
                    H**4,H**3*Hr,H**2*Hr**2,H*Hr**3,Hr**4))
    H = X['H']
    Hr = X['Hr']
    Hc = (H-Hr)/2
    Hb = (H+Hr)/2
    X['Hc'] = Hc
    X['Hb'] = Hb

    if X['Mtype'].value=='Magnetisations':
        M = X['M']
    else:
        M = X['DM']

    D_Hc = X['client'].scatter(Hc,broadcast=True)
    D_Hb = X['client'].scatter(Hb,broadcast=True)
    D_M = X['client'].scatter(M,broadcast=True)
    D_X = X['client'].scatter(X0,broadcast=True)

    #Down-sample and Split arrays for DASK
    Nsplit = 30

    if X['Ndown'].value<Hc.size:
        X['Hc1'], X['Hc2'], X['Hb1'], X['Hb2'] = measurement_limts(X)
        gradient = (X['Hb1']-(X['Hb1']-X['Hc2']))/(X['Hc2']-X['Hc1'])
        intercept = X['Hb1']-gradient*X['Hc2']
   
        #generate random points to down-sample
        np.random.seed(999)
        #Hci = np.random.rand(X['Ndown'].value*3)*(X['Hc2']-X['Hc1'])+X['Hc1']
        #Hbi = np.random.rand(X['Ndown'].value*3)*(X['Hb2']-(X['Hb1']-X['Hc2']))+(X['Hb1']-X['Hc2'])
        #Hidx = np.argwhere(Hbi>=Hci*gradient+intercept)
        #X['Hci'] = Hci[Hidx[0:X['Ndown'].value]]
        #X['Hbi'] = Hbi[Hidx[0:X['Ndown'].value]]
        
        Ridx = np.random.choice(M.size, size=X['Ndown'].value, replace=False)
        X['Hci']=Hc[Ridx]
        X['Hbi']=Hb[Ridx]
        
        Hc0 = np.array_split(X['Hci'],Nsplit)
        Hb0 = np.array_split(X['Hbi'],Nsplit)
    else:
        Hc0 = np.array_split(Hc,Nsplit)
        Hb0 = np.array_split(Hb,Nsplit)

    sc0 = X['SC'].value[0]
    sc1 = X['SC'].value[1]
    sb0 = X['SB'].value[0]
    sb1 = X['SB'].value[1]
    lamb_sc = X['lambdaSC'].value
    lamb_sb = X['lambdaSB'].value

    #Split jobs over DASK
    jobs = []
    for i in range(len(Hc0)):
        job = X['client'].submit(variforc_regression_evidence,sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,D_Hc,D_Hb,dH*np.sqrt(2),D_M,Hc0[i],Hb0[i],D_X)
        jobs.append(job)

    results = X['client'].gather(jobs)

    Midx = results[0][:,0]
    rho = results[0][:,1]

    for i in range(len(results)-1):
        Midx=np.concatenate((Midx,results[i+1][:,0]))
        rho=np.concatenate((rho,results[i+1][:,1]))
    
    X['rho'] = rho
    X['Midx'] = Midx

    if X['Ndown'].value<Hc.size:
        X['Xi']='NA'
        X = plot_model_results_down(X)
    else:
        X = triangulate_rho(X)
        X = plot_model_results_full(X)
    
    
    return X """
##### END SECTION: MODEL FUNCTIONS  #################################################

##### BEGIN SECTION: FORC plotting  #################################################
def FORC_plot(X):

    #unpack data
    Xi = X['Xi']
    
    #if type(Xi) is str:
    #    print('Error: The current model is based on downsampled data. Run a full model using "calculate_model"') #return error
    #    return X
    
    Yi = X['Yi']
    Zi = X['Zi']
    Hc1 = X['Hc1']
    Hc2 = X['Hc2']
    Hb1 = X['Hb1']
    Hb2 = X['Hb2']

    #Set up widgets for interactive plot
    style = {'description_width': 'initial'} #general style settings
    
    #DEFINE INTERACTIVE WIDGETS
    
    #should a colorbar be included
    colorbar_widge = widgets.Checkbox(value=False, description = 'Include color scalebar',style=style) 
    
    #Frequency for contour lines to be included in plot
    contour_widge = widgets.Select(
        options=[['Select contour frequency',-1],
                 ['Every level',1],
                 ['Every 2nd level',2],
                 ['Every 3rd level',3],
                 ['Every 4th level',4],
                 ['Every 5th level',5],
                 ['Every 10th level',10],
                 ['Every 20th level',20],
                 ['Every 50th level',50],
                ],
        value=-1,
        rows=1,
        description='Plot contours',style=style)
    
    contourpts_widge = widgets.FloatSlider(value=1.0,min=0.5,max=3.0,step=0.5, description = 'Contour line width [pts]',style=style)

    #check box for plot download
    download_widge = widgets.Checkbox(value=False, description = 'Download plot',style=style) 
    
    #How many contour levels should be included
    level_widge = widgets.Select(
        options=[['20',20],['30',30],['50',50],['75',75],['100',100],['200',200],['500',500]],
        value=50,
        rows=1,
        description='Number of color levels',style=style)

    #X-axis minimum value
    xmin_widge = widgets.FloatText(value=0,description='Minimum $\mu_0H_c$ [T]',style=style,step=0.001)    
    xmax_widge = widgets.FloatText(value=np.round(Hc2*1000)/1000,description='Maximum $\mu_0H_c$ [T]',style=style,step=0.001)
    ymin_widge = widgets.FloatText(value=np.round((Hb1-Hc2)*1000)/1000,description='Minimum $\mu_0H_u$ [T]',style=style,step=0.001)
    ymax_widge = widgets.FloatText(value=np.round(Hb2*1000)/1000,description='Maximum $\mu_0H_u$ [T]',style=style,step=0.001)

    #launch the interactive FORC plot
    x = interactive(forcplot,
             Xi=fixed(Xi), #X point grid
             Yi=fixed(Yi), #Y point grid
             Zi=fixed(Zi), #interpolated Z values
             fn=fixed(X['sample']), #File information
             mass=fixed(X['mass']), #Preprocessing information
             colorbar=colorbar_widge, #Include colorbar             
             level=level_widge, #Number of levels to plot 
             contour=contour_widge, #Contour levels to plot
             contourpts=contourpts_widge, #Contour line width
             xmin=xmin_widge, #X-minimum
             xmax=xmax_widge, #X-maximum
             ymin=ymin_widge, #Y-minimum
             ymax=ymax_widge, #Y-maximum
             download = download_widge #download plot
            )
    
    #create tabs
    tab_nest = widgets.Tab()
    # tab_nest.children = [tab_visualise]
    tab_nest.set_title(0, 'FORC PLOTTING')

    #interact function in isolation
    tab_nest.children = [VBox(children = x.children)]
    display(tab_nest)
    
    #display(x) #display the interactive plot

def forcplot(Xi,Yi,Zi,fn,mass,colorbar,level,contour,contourpts,xmin,xmax,ymin,ymax,download):
    

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    
    if mass.value<0.0:        
        Xi_new = Xi
        Yi_new = Yi
        Zi_new = Zi
        xlabel_text = '$\mu_0 H_c [T]$' #label Hc axis [SI units]
        ylabel_text = '$\mu_0 H_u [T]$' #label Hu axis [SI units]
        cbar_text = '$Am^2 T^{-2}$'
    else:
        Xi_new = Xi
        Yi_new = Yi
        Zi_new = Zi / (mass.value/1000.0)
        xlabel_text = '$\mu_0 H_c [T]$' #label Hc axis [SI units]
        ylabel_text = '$\mu_0 H_u [T]$' #label Hu axis [SI units]
        cbar_text = '$Am^2 T^{-2} kg^{-1}$'
        

    #define colormaps
    idx=(Xi_new>=xmin) & (Xi_new<=xmax) & (Yi_new>=ymin) & (Yi_new<=ymax) #find points currently in view
    cmap,vmin,vmax = FORCinel_colormap(Zi_new[idx])

    CS = ax.contourf(Xi_new, Yi_new, Zi_new, level, cmap = cmap, vmin=vmin, vmax=vmax)       
    if (contour>0) & (contour<level):
        CS2 = ax.contour(CS, levels=CS.levels[::contour], colors='k',linewidths=contourpts)

    ax.set_xlabel(xlabel_text,fontsize=14) #label Hc axis [SI units]
    ax.set_ylabel(ylabel_text,fontsize=14) #label Hu axis [SI units]  

    # Set plot Xlimits
    xlimits = np.sort((xmin,xmax))
    ax.set_xlim(xlimits)
    
    #Set plot Ylimits
    ylimits = np.sort((ymin,ymax))
    ax.set_ylim(ylimits)
    
    #Set ticks and plot aspect ratio
    ax.tick_params(labelsize=14)
    ax.set_aspect('equal') #set 1:1 aspect ratio
    ax.minorticks_on() #add minor ticks
    
    #Add colorbar
    if colorbar == True:    
        cbar = fig.colorbar(CS,fraction=0.04, pad=0.08)
        cbar.ax.tick_params(labelsize=14)
        #cbar.ax.set_title(cbar_text,fontsize=14)
        cbar.set_label(cbar_text,fontsize=14)
    
    #Activate download to same folder as data file
    if download==True:
        outputfile = fn.value+'_FORC.eps'
        plt.savefig(outputfile, dpi=300, bbox_inches="tight")
    
    #show the final plot
    plt.show()

##### END SECTION: FORC plotting  #################################################


##### BEGIN SECTION: HELPER FUNCTIONS  #################################################

  
# define function which will look for lines in the header that start with certain strings
def find_data_lines(fp):
    """Helper function to identify measurement lines in a FORC data file.
    
    Given the various FORC file formats, measurements lines are considered to be those which:
    Start with a '+' or,
    Start with a '-' or,
    Are blank (i.e. lines between FORCs and calibration points) or,
    Contain a ','

    Inputs:
    fp: file identifier

    Outputs:
    line: string corresponding to data line that meets the above conditions
    """
    return [line for line in fp if ((line.startswith('+')) or (line.startswith('-')) or (line.strip()=='') or line.find(',')>-1.)]

#function to parse calibration points and provide time stamps
def lines_that_start_with(string, fp):
    """Helper function to lines in a FORC data file that start with a given string
    
    Inputs:
    string: string to compare lines to 
    fp: file identifier

    Outputs:
    line: string corresponding to data line that meets the above conditions
    """
    return [line for line in fp if line.startswith(string)]

def parse_header(file,string):
    """Function to extract instrument settings from FORC data file header
    
    Inputs:
    file: name of data file (string)    
    string: instrument setting to be extracted (string)

    Outputs:
    output: value of instrument setting [-1 if setting doesn't exist] (float)
    """
    output=-1 #default output (-1 corresponds to no result, i.e. setting doesn't exist)
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idx = line.find('=') #Some file formats may contain an '='
            if idx>-1.: #if '=' found
                output=float(line[idx+1:]) #value taken as everything to right of '='
            else: # '=' not found
                idx = len(string) #length of the setting string 
                output=float(line[idx+1:])  #value taken as everything to right of the setting name 

    return output

def parse_units(file):
    """Function to extract instrument unit settings ('') from FORC data file header
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    CGS [Cgs setting] or SI [Hybrid SI] (string)
    """
    string = 'Units of measure' #header definition of units
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idxSI = line.find('Hybrid SI') #will return location if string is found, otherwise returns -1
            idxCGS = line.find('Cgs') #will return location if string is found, otherwise returns -1
    
    if idxSI>idxCGS: #determine which unit string was found in the headerline and output
        return 'SI'
    else:
        return 'Cgs'

def parse_mass(file):
    """Function to extract sample from FORC data file header
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    Mass in g or N/A
    """
    output = 'N/A'
    string = 'Mass' #header definition of units
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idx = line.find('=') #Some file formats may contain an '='
            if idx>-1.: #if '=' found
                output=(line[idx+1:]) #value taken as everything to right of '='
            else: # '=' not found
                idx = len(string) #length of the setting string 
                output=(line[idx+1:])  #value taken as everything to right of the setting name
        
            if output.find('N/A') > -1:
                output = 'N/A'
            else:
                output = float(output)

    return output

def calibration_times(file, Npts):
    """Function to estimate the time at which calibration points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.

    Inputs:
    file: name of data file (string)    
    Npts: number of calibration points (int)

    Outputs:
    tcal_k: Estimated times at which the calibration points were measured (float)
    """    
    unit=parse_units(file) #determine measurement system (CGS or SI)

    string='PauseRvrsl' #Pause at reversal field (new file format, -1 if not available)
    tr0=parse_header(file,string)
    
    string='PauseNtl' #Pause at reversal field (old file format, -1 if not available)
    tr1=parse_header(file,string)

    tr=np.max((tr0,tr1)) #select Pause value depending on file format
    
    string='Averaging time' #Measurement averaging time 
    tau=parse_header(file,string)

    string='PauseCal' #Pause at calibration point
    tcal=parse_header(file,string)

    string='PauseSat' #Pause at saturation field
    ts=parse_header(file,string)

    string='SlewRate' #Field slewrate
    alpha=parse_header(file,string)

    string='HSat' #Satuation field
    Hs=parse_header(file,string)

    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(file,string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(file,string)

    string='Hc2' #upper Hc value for the FORC box (n.b. Hc1 is assumed to be 0)
    Hc2=parse_header(file,string)

    string='NForc' # Numer of measured FORCs (new file format, -1 if not available)
    N0=parse_header(file,string)

    string='NCrv'  # Numer of measured FORCs (old file format, -1 if not available)
    N1=parse_header(file,string)

    N=np.max((N0,N1)) #select Number of FORCs depending on file format

    if unit=='Cgs':
        alpha=alpha/1E4 #convert from Oe to T
        Hs=Hs/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T
    
    dH = (Hc2-Hb1+Hb2)/N #estimated field spacing
    
    #now following Elgi's estimate of the measurement time
    nc2 = Hc2/dH
    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt2 = tr + tau + (Hc2-Hb2-dH)/alpha

    Npts=int(Npts)
    tcal_k=np.zeros(Npts)
    
    for k in range(1,Npts+1):
        if k<=1+nc2:
            tcal_k[k-1]=k*Dt1-Dt2+dH/alpha*k**2+(tau-dH/alpha)*(k-1)**2
        else:
            tcal_k[k-1]=k*Dt1-Dt2+dH/alpha*k**2+(tau-dH/alpha)*((k-1)*(1+nc2)-nc2)

    return tcal_k

def measurement_times(file,Fk,Fj):
    """Function to estimate the time at which magnetization points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.

    Inputs:
    file: name of data file (string)    
    Fk: FORC indicies (int)
    Fj: Measurement indicies within given FORC

    Outputs:
    Ft: Estimated times at which the magnetization points were measured (float)
    """    
    unit=parse_units(file) #determine measurement system (CGS or SI)

    string='PauseRvrsl' #Pause at reversal field (new file format, -1 if not available)
    tr0=parse_header(file,string)
    
    string='PauseNtl' #Pause at reversal field (old file format, -1 if not available)
    tr1=parse_header(file,string)

    tr=np.max((tr0,tr1)) #select Pause value depending on file format
    
    string='Averaging time' #Measurement averaging time 
    tau=parse_header(file,string)

    string='PauseCal' #Pause at calibration point
    tcal=parse_header(file,string)

    string='PauseSat' #Pause at saturation field
    ts=parse_header(file,string)

    string='SlewRate' #Field slewrate
    alpha=parse_header(file,string)

    string='HSat' #Satuation field
    Hs=parse_header(file,string)

    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(file,string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(file,string)

    string='Hc2' #upper Hc value for the FORC box (n.b. Hc1 is assumed to be 0)
    Hc2=parse_header(file,string)

    string='NForc' # Numer of measured FORCs (new file format, -1 if not available)
    N0=parse_header(file,string)

    string='NCrv'  # Numer of measured FORCs (old file format, -1 if not available)
    N1=parse_header(file,string)

    N=np.max((N0,N1)) #select Number of FORCs depending on file format

    if unit=='Cgs':
        alpha=alpha/1E4 #convert from Oe to T
        Hs=Hs/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T

    dH = (Hc2-Hb1+Hb2)/N #estimated field spacing
    
    #now following Elgi's estimate of the measurement time
    nc2 = Hc2/dH

    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt3 = Hb2/alpha

    Npts=int(Fk.size)
    Ft=np.zeros(Npts)
    
    for i in range(Npts):
        if Fk[i]<=1+nc2:
            Ft[i]=Fk[i]*Dt1+Dt3+Fj[i]*tau+dH/alpha*(Fk[i]*(Fk[i]-1))+(tau-dH/alpha)*(Fk[i]-1)**2
        else:
            Ft[i]=Fk[i]*Dt1+Dt3+Fj[i]*tau+dH/alpha*(Fk[i]*(Fk[i]-1))+(tau-dH/alpha)*((Fk[i]-1)*(1+nc2)-nc2)

    return Ft

def parse_calibration(file):
    """Function to extract measured calibration points from a FORC sequence
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    Hcal: sequence of calibration fields [float, SI units]
    Mcal: sequence of calibration magnetizations [float, SI units]
    tcal: Estimated times at which the calibration points were measured (float, seconds)
    """ 

    dum=-9999.99 #dum value to indicate break in measurement seqence between FORCs and calibration points
    N0=int(1E6) #assume that any file will have less than 1E6 measurements
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in find_data_lines(fp): #does the current line contain measurement data
            count=count+1 #increase counter
            idx = line.find(',') #no comma indicates a blank linw
            if idx>-1: #line contains a comma
                H0[count]=float(line[0:idx]) #assign field value (1st column)
                line=line[idx+1:] #remove the leading part of the line (only characters after the first comma remain)
                idx = line.find(',') #find next comman
                if idx>-1: #comma found in line
                    M0[count]=float(line[0:idx]) #read values up to next comma (assumes 2nd column is magnetizations)
                else: #comma wasn't found   
                    M0[count]=float(line) # magnetization value is just the remainder of the line 
            else:
                H0[count]=dum #line is blank, so fill with dummy value
                M0[count]=dum #line is blank, so fill with dummy value

    idx_start=np.argmax(H0!=dum) #find the first line that contains data            
    M0=M0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector           
    M0=M0[~np.isnan(M0)] #remove any NaNs at the end of the array
    H0=H0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector
    H0=H0[~np.isnan(H0)] #remove any NaNs at the end of the array

    ## now need to pull out the calibration points, will be after alternate -9999.99 entries
    idxSAT = np.array(np.where(np.isin(H0, dum))) #location of dummy values
    idxSAT = np.ndarray.squeeze(idxSAT) #squeeze into 1D
    idxSAT = idxSAT[0::2]+1 #every second index+1 should be calibration points

    Hcal=H0[idxSAT[0:-1]] #calibration fields
    Mcal=M0[idxSAT[0:-1]] #calibration magnetizations
    tcal=calibration_times(file,Hcal.size) #estimate the time of each calibratio measurement

    unit = parse_units(file)
    
    if unit=='Cgs': #ensure SI units
        Hcal=Hcal/1E4 #convert from Oe to T
        Mcal=Mcal/1E3 #convert from emu to Am^2

    return Hcal, Mcal, tcal

def parse_measurements(file):
    """Function to extract measurement points from a FORC sequence
    
    Inputs:
    file: name of data file (string)    

    Outputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    Ft: Estimated times at which the points were measured (float, seconds)
    dH: Measurement field spacing [float SI units]
    """ 

    dum=-9999.99 #dum value to indicate break in measurement seqence between FORCs and calibration points
    N0=int(1E6) #assume that any file will have less than 1E6 measurements
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in find_data_lines(fp): #does the current line contain measurement data
            count=count+1 #increase counter
            idx = line.find(',') #no comma indicates a blank linw
            if idx>-1: #line contains a comma
                H0[count]=float(line[0:idx]) #assign field value (1st column)
                line=line[idx+1:] #remove the leading part of the line (only characters after the first comma remain)
                idx = line.find(',') #find next comman
                if idx>-1: #comma found in line
                    M0[count]=float(line[0:idx]) #read values up to next comma (assumes 2nd column is magnetizations)
                else: #comma wasn't found   
                    M0[count]=float(line) # magnetization value is just the remainder of the line 
            else:
                H0[count]=dum #line is blank, so fill with dummy value
                M0[count]=dum #line is blank, so fill with dummy value

    idx_start=np.argmax(H0!=dum) #find the first line that contains data            
    M0=M0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector           
    M0=M0[~np.isnan(M0)] #remove any NaNs at the end of the array
    H0=H0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector
    H0=H0[~np.isnan(H0)] #remove any NaNs at the end of the array

    ## determine indicies of each FORC
    idxSAT = np.array(np.where(np.isin(H0, dum))) #find start address of each blank line
    idxSAT = np.ndarray.squeeze(idxSAT) #squeeze into 1D
    idxSTART = idxSAT[1::2]+1 #find start address of each FORC
    idxEND = idxSAT[2::2]-1 ##find end address of each FORC

    
    #Extract first FORC to initialize arrays 
    M=M0[idxSTART[0]:idxEND[0]+1] #Magnetization values
    H=H0[idxSTART[0]:idxEND[0]+1] #Field values
    Hr=np.ones(idxEND[0]+1-idxSTART[0])*H0[idxSTART[0]] #Reversal field values
    Fk=np.ones(idxEND[0]+1-idxSTART[0]) #index number of FORC
    Fj=np.arange(1,1+idxEND[0]+1-idxSTART[0])# measurement index within given FORC

    #Extract remaining FORCs one by one into into a long-vector
    for i in range(1,idxSTART.size):
        M=np.concatenate((M,M0[idxSTART[i]:idxEND[i]+1]))
        H=np.concatenate((H,H0[idxSTART[i]:idxEND[i]+1]))
        Hr=np.concatenate((Hr,np.ones(idxEND[i]+1-idxSTART[i])*H0[idxSTART[i]]))
        Fk=np.concatenate((Fk,np.ones(idxEND[i]+1-idxSTART[i])+i))
        Fj=np.concatenate((Fj,np.arange(1,1+idxEND[i]+1-idxSTART[i])))
    
    unit = parse_units(file) #Ensure use of SI units
    
    if unit=='Cgs':
        H=H/1E4 #Convert Oe into T
        Hr=Hr/1E4 #Convert Oe into T
        M=M/1E3 #Convert emu to Am^2

    dH = np.mean(np.diff(H[Fk==np.max(Fk)])) #mean field spacing

    Ft=measurement_times(file,Fk,Fj) #estimated time of each measurement point

    return H, Hr, M, Fk, Fj, Ft, dH

####### Define VARIFORC window functions #######
def vari_pts(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0,Hb0):
       
    Sc=vari_s(sc0,sc1,lamb_sc,Hc0,dH)
    Sb=vari_s(sb0,sb1,lamb_sb,Hb0,dH)
        
    idx=((np.abs(Hc-Hc0)/dH<Sc) & (np.abs(Hb-Hb0)/dH<Sb))
    
    return idx

def vari_T(u,s):
    
    T=np.zeros(u.shape) #initialize array

    absu=np.abs(u)
    absu_s=absu-s

    idx=(absu<=s)
    T[idx]= 2.*absu_s[idx]**2 #3rd condition

    idx=(absu<=s-0.5)
    T[idx]=1.-2.*(absu_s[idx]+1.)**2 #2nd condition

    idx=(absu<=s-1.)
    T[idx]=1.0 #first condition
  
    return T


def vari_W(Hc,Hc0,Hb,Hb0,dH,Sc,Sb):
    # calculate grid of weights
    #Hc = Hc grid
    #Hb = Hb grid
    #Hc0,Hb0 = center of weighting function
    #dH = field spacing
    #Sc = Hc-axis smoothing factor
    #Sb = Hb-axis smoothing factor
    
    x=Hc-Hc0
    y=Hb-Hb0
        
    return vari_T(x/dH,Sc)*vari_T(y/dH,Sb)


def vari_s(s0,s1,lamb,H,dH):
    
    #calculate local smoothing factor
    RH = np.maximum(s0,np.abs(H)/dH)
    LH = (1-lamb)*s1+lamb*np.abs(H)/dH
    
    return np.min((LH,RH))


def vari_weights(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0,Hb0):
       
    Sc=vari_s(sc0,sc1,lamb_sc,Hc0,dH)
    Sb=vari_s(sb0,sb1,lamb_sb,Hb0,dH)
        
    idx=((np.abs(Hc-Hc0)/dH<Sc) & (np.abs(Hb-Hb0)/dH<Sb))
    weights=vari_W(Hc[idx],Hc0,Hb[idx],Hb0,dH,Sc,Sb)
    
    return weights, idx