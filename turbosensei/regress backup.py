import numpy as np
import ipywidgets as widgets
from ipywidgets import VBox, HBox
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.special import gammaln, logsumexp
from dask.distributed import Client, LocalCluster, progress #needed for multiprocessing
import codecs as cd
import turbosensei.utils as ut
import numba

#### define jit-based matrix inverse ####
@numba.jit
def inv_jit(A):
  return np.linalg.inv(A)

#### REGRESS OPTIONS ####
def options(X):
    
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
    X['ncore']=4
    
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
    X['Ndown']=down_widge
    X['workers']=dask_widge
    
    return X

    #### REGRESSION COMPARISON

def compare(X):

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

    idx = np.argwhere(in_window(X,X['Hc'],X['Hb'])==True)
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
        job = X['client'].submit(process_split,X['Ds'],X['Sp_i'][i],Didx,Mswitch)
        jobs.append(job)
    
    results = X['client'].gather(jobs)

    L = results[0]
    for i in range(len(results)-1):
        L=np.concatenate((L,results[i+1]))
    
    X['L'] = L
    
    #Make results plots
    i0 = np.argmax(L[:,2])
    if (Mswitch<0.5):
        BF = regress_split(X['Xlsq'],X['Mnorm'],X['Hc'],X['Hb'],X['dH'],X['Hc'],X['Hb'],X['Sp'][i0,0],X['Sp'][i0,1],X['Sp'][i0,4],X['Sp'][i0,2],X['Sp'][i0,3],X['Sp'][i0,4])
    else:
        BF = regress_split(X['Xlsq'],X['DMnorm'],X['Hc'],X['Hb'],X['dH'],X['Hc'],X['Hb'],X['Sp'][i0,0],X['Sp'][i0,1],X['Sp'][i0,4],X['Sp'][i0,2],X['Sp'][i0,3],X['Sp'][i0,4])
    
    BF[np.isinf(BF)]=1E200
    X['BF']=BF  
    X['Pr']=np.exp(BF-logsumexp(BF,axis=1)[:,np.newaxis])
    Lpt = np.argmax(BF,axis=1)
    Lpt[np.max(X['BF'],axis=1)<1]=0
    
    X = plot_model_selection(X,Lpt[idx],i0)

    return X

def process_split(X,S,Didx,Mswitch):
    
    Xlsq = X['Xlsq']
    if (Mswitch<0.5):
        M = X['M']
    else:
        M = X['DM']

    Hc = X['Hc']
    Hb = X['Hb']
    #Hc = Xlsq[:,1]
    #Hb = Xlsq[:,2]
    
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
        BF = regress_split(Xlsq,M,Hc,Hb,dH,Hc0,Hb0,sc0[i],sc1[i],lamb[i],sb0[i],sb1[i],lamb[i])
        Li = np.argmax(BF,axis=1) 
        L[i,0] = np.sum(Li==0)
        L[i,1] = np.sum(Li==1)
        L[i,2] = np.sum(Li==2) 
        L[i,3] = np.sum(Li==3)

        Li = np.argmax(BF,axis=1)
        L[i,4] = np.sum(Li==2) 

    return L

def regress_split(Xlsq,M,Hc,Hb,dH,Hc0,Hb0,sc0,sc1,lamb_sc,sb0,sb1,lamb_sb):
    
    Npts = Hc0.size
    BF = np.zeros((Npts,4))

    for i in range(Npts):
        idx = OLS_pts(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0[i],Hb0[i])
        BF[i,:] = OLS2BF(Xlsq[idx,:],M[idx])

    return BF

def execute(X):
    
    L = X['L']
    Mswitch = X['Mswitch']

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

    X = triangulate_rho(X) #triangulate rho for plotting

    return X

def triangulate_rho(X):

    se = X['se']
    rho = X['rho']
    Hc = X['Hc']
    Hb = X['Hb']
    dH = X['dH']
    
    #PERFORM GRIDDING AND INTERPOLATION FOR FORC PLOT
    X['Hc1'], X['Hc2'], X['Hb1'], X['Hb2'] = ut.measurement_limts(X)
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

#### PLOTTING FUNCTIONS ####
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
    Hc1 = X['Hc1']
    Hc2 = X['Hc2']
    Hb1 = X['Hb1']
    Hb2 = X['Hb2']
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

#### HELPER FUNCTIONS ####
def OLS2BF(X,y):

    N = y.size    
    XT = X.T
    XTX = XT @ X
    ydev = np.sum((y-np.mean(y))**2)

    # Test 1st order model
    status = True
    BF1 = 0.0
    try:
        iv = inv_jit(XTX[0:3,0:3])
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        ssr1 = np.sum((X[:,0:3] @ (iv @ XT[0:3,:] @ y) - y)**2) 
        r2_1 = 1 - ssr1 / ydev
        BF1, B = Bayes_factor(N,r2_1)
        
    # Test 2nd order model    
    status = True
    BF2 = 0.0
    try:
        iv = inv_jit(XTX[0:6,0:6])
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        ssr2 = np.sum((X[:,0:6] @ (iv @ XT[0:6,:] @ y) - y)**2)
        r2_2 = 1 - ssr2 / ydev
        BF2, _ = Bayes_factor(N,r2_2,2,B)    
    
    # Test 3rd order model
    status = True
    BF3 = 0.0      
    try:
        iv = inv_jit(XTX)
    except np.linalg.LinAlgError as err:
        status = False
    
    if status:
        ssr3 = np.sum((X @ (iv @ XT @ y) - y)**2)
        r2_3 = 1 - ssr3 / ydev   
        BF3, _ = Bayes_factor(N,r2_3,3,B)
    
    return np.array((0,BF1,BF2,BF3))

def Bayes_factor(N,R2,order=1,B=0):
    
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

    #### HELPER FUNCTIONS ####
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
def in_window(X,Hc_i,Hb_i):
    return (Hc_i>=X['Hc1']) & (Hc_i<=X['Hc2']) & (Hb_i<=X['Hb2']) & (Hb_i>=(X['Hb1']-(X['Hc2']-X['Hc1'])+(Hc_i-X['Hc1'])))

@numba.jit
def OLS_pts(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0,Hb0):
       
    Sc=vari_s(sc0,sc1,lamb_sc,Hc0,dH)
    Sb=vari_s(sb0,sb1,lamb_sb,Hb0,dH)
    idx = np.logical_and(np.abs(Hc-Hc0)/dH<Sc,np.abs(Hb-Hb0)/dH<Sb)

    return idx    

@numba.jit
def vari_s(s0,s1,lamb,H,dH):
    
    #calculate local smoothing factor
    RH = np.maximum(s0,np.abs(H)/dH)
    LH = (1-lamb)*s1+lamb*np.abs(H)/dH
    
    return np.minimum(LH,RH)

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
def vari_weights(sc0,sc1,lamb_sc,sb0,sb1,lamb_sb,Hc,Hb,dH,Hc0,Hb0):
       
    Sc=vari_s(sc0,sc1,lamb_sc,Hc0,dH)
    Sb=vari_s(sb0,sb1,lamb_sb,Hb0,dH)
        
    idx=((np.abs(Hc-Hc0)/dH<Sc) & (np.abs(Hb-Hb0)/dH<Sb))
    weights=vari_W(Hc[idx],Hc0,Hb[idx],Hb0,dH,Sc,Sb)
    
    return weights, idx
