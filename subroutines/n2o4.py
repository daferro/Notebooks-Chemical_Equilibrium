# ============================================== #
import numpy                as np                #
# ---------------------------------------------- #
import matplotlib.pyplot    as plt               #
import matplotlib.ticker    as mticker           #
import plotly.graph_objects as go                #
from   plotly.subplots      import make_subplots #
# ---------------------------------------------- #
from   constants            import m_u,m_e,q_e,h,k_B,c_0,eps0,NA
from   constants            import P_o,c_o,R,hbar,a_0,Eh,Hz_au
from   constants            import NPOINTST,NPOINTSXI,REL_XI_EQ
from   constants            import ZERO1,ZERO2,ZERO3,ZERO4
from   constants            import FONTSIZE
# ---------------------------------------------- #
import general
# ---------------------------------------------- #
# ============================================== #

NPOINTSYB   = 51   # number of yB points (thermo & kinetics)
last_info   = None
last_fig    = None
ARRHENIUS_A = None
ARRHENIUS_B = None

# ============================================= #
# ---- Specific for N2O4 <-> 2NO2 (PART 1) ---- #
# ============================================= #
def load_n2o4_2no2():
    # from Atkins' Physical Chemistry
    TREF     = 298
    DHo_ref  =  57.20 * 1E3
    DSo_ref  = 175.83
    DCPo_ref =  -2.88
    #-------------------------------------------
    refdata = (DHo_ref,DSo_ref,DCPo_ref,TREF)
    #-------------------------------------------
    molecules     = ["N2O4","NO2"]
    nus           = np.array([-1,2])
    #-------------------------------------------
    GEOMINFO      = {"N2O4":[(1,5),(1,5,3),(4,5)] , "NO2":[(0,1),(1,0,2)]}
    #-------------------------------------------
    n_0           = np.array([1.00, 0.00])
    #-------------------------------------------
    return refdata,molecules,nus,n_0,GEOMINFO
# --------------------------------------------- #
def get_xieq_PT_N2O4(T,P,nA0,nB0):
    refdata = load_n2o4_2no2()[0]
    dG0     = general.get_DGo(T,refdata)
    Kp      = np.exp(-dG0/R/T)
    alpha   = Kp * P_o/P
    # solution for second-order equation, knowing that (-nB0/2<= xi <= nA0)
    xi_eq   = (-nB0+(2*nA0+nB0)*np.sqrt(alpha/(alpha+4)))/2
    return xi_eq
# --------------------------------------------- #
def get_xieq_VT_N2O4(T,V,nA0,nB0):
    refdata = load_n2o4_2no2()[0]
    dG0     = general.get_DGo(T,refdata)
    Kp      = np.exp(-dG0/R/T)
    beta    = Kp * (P_o*V/R/T)
    # solution for second-order equation, knowing that (-nB0/2<= xi <= nA0)
    xi_eq   = (-4*nB0-beta+np.sqrt((8*(2*nA0+nB0)+beta)*beta))/8
    return xi_eq
# --------------------------------------------- #
def yB_to_xi_N2O4(yB):
    refdata,molecules,nus,n_0,GEOMINFO = load_n2o4_2no2()
    xi = (yB*n_0.sum() - n_0[1])/(2-yB)
    return xi
# --------------------------------------------- #
def intercept_getGm_N2O4(T,P,yB):
    refdata,molecules,nus,n_0 = load_n2o4_2no2()[0:4]
    xi     = yB_to_xi_N2O4(yB)
    ntot   = n_0.sum() + xi
    Gtot   = general.get_G_PT(xi,P,T,n_0,nus,refdata)
    Gm     = Gtot/ntot
    return Gm, Gtot, ntot
# --------------------------------------------- #
def intercept_getline_N2O4(T,P,yB_i):
    refdata,molecules,nus,n_0 = load_n2o4_2no2()[0:4]
    #partial pressures and quotient of reaction
    yA_i  = 1-yB_i
    pB_i  = P*yB_i
    pA_i  = P*yA_i
    Qp    = (pB_i/P_o)**2 / (pA_i/P_o)
    # value for xi_i
    xi_i  = yB_to_xi_N2O4(yB_i)
    # delta_r G^o (T)
    dGo_T = general.get_DGo(T,refdata)
    # G_tot
    Gm_i, Gtot_i, ntot_i = intercept_getGm_N2O4(T,P,yB_i)
    # get slope (m)
    dxidyB = (2*n_0.sum() - n_0[1]) / (2-yB_i)**2
    dGdxi  = dGo_T + R*T*np.log(Qp)
    m      = 1/ntot_i * dxidyB * (dGdxi - Gm_i)
    # Point of the line
    xx     = yB_i
    yy     = (Gm_i - intercept_getGm_N2O4(T,P,0)[0])/(R*T)
    # get intercept (b)
    m      = m / (R*T)
    b      = yy - m*xx
    return m,b, (xx,yy)
# --------------------------------------------- #
def plot_intercept_N2O4(T,P,yB):
    nA0,nB0 = load_n2o4_2no2()[3]
    n0      = nA0+nB0
    # equilibrium
    xi_eq   = get_xieq_PT_N2O4(T,P,nA0,nB0)
    yB_eq   = (nB0+2*xi_eq)/(n0 + xi_eq)
    Gtot_eq = intercept_getGm_N2O4(T,P,yB_eq)[1]
    # Get data in terms of yB_i
    lGm,lGtot,lntot = [],[],[]
    list_yB         = np.linspace(0.0,1.0,NPOINTSYB)
    for yB_i in list_yB:
        Gm_i, Gtot_i, ntot_i = intercept_getGm_N2O4(T,P,yB_i)
        lntot.append(ntot_i)
        lGtot.append(Gtot_i)
    # control point
    if yB == 0.0: yB = ZERO4
    if yB == 1.0: yB = 1 - ZERO4
    # equation for tangent of Gm
    m,b,(xx1,yy1) = intercept_getline_N2O4(T,P,yB)
    # values at the selected point
    Gm, Gtot, ntot = intercept_getGm_N2O4(T,P,yB)
    # apply reference
    Gtot    =   (Gtot    - lGtot[0])/(R*T)
    Gtot_eq =   (Gtot_eq - lGtot[0])/(R*T)
    lGtot   = [(lGtot[i] - lGtot[0])/(R*T) for i in range(len(lGtot))]
    lGm     = [lGtot[i]/lntot[i]           for i in range(len(lGtot))]
    # --- Create side-by-side axes ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    # -- left plot --
    ax1.plot(list_yB,lGtot,'k-')
    ax1.plot(yB_eq,Gtot_eq,'ko',zorder=1)
    ax1.plot(yB   ,Gtot   ,'ro',zorder=2)
    for ii in ["x","y"]: ax1.tick_params(axis=ii,labelsize=FONTSIZE[1])
    ax1.set_xlim( 0.0 , 1.0 )
    ax1.set_xlabel(r"$y_{\rm NO_2}$",fontsize=FONTSIZE[2])
    ax1.set_ylabel(r"$[G(y_{\rm NO_2})-G(0)]\cdot (RT)^{-1} \;\; \mathrm{[mol]}$",fontsize=FONTSIZE[2])
    # -- right plot --
    xx2 = [ZERO4,1-ZERO4]
    yy2 = [m*xx_i + b for xx_i in xx2]
    ylim1 = min(lGm)
    ylim2 = max(lGm)
    delta = ylim2-ylim1
    ylim1 = ylim1 - delta*0.1
    ylim2 = ylim2 + delta*0.1
    ax2.plot(list_yB,lGm,'k-')
    ax2.plot(xx1,yy1,'ro')
    ax2.plot(xx2,yy2,'r--',label=rf"${m:+.5f} \cdot y_{{\rm NO_2}} {b:+.5f}$")
    for ii in ["x","y"]: ax2.tick_params(axis=ii,labelsize=FONTSIZE[1])
    ax2.set_xlim( 0 , 1 )
    ax2.set_ylim(ylim1,ylim2)
    ax2.set_xlabel(r"$y_{\rm NO_2}$",fontsize=FONTSIZE[2])
    ax2.set_ylabel(r"$[G_{\rm m}(y_{\rm NO_2})- G_{\rm m}(0)] \cdot (RT)^{-1}$",fontsize=FONTSIZE[2])
    ax2.legend(loc="upper center",fontsize=FONTSIZE[0])
    # --- update global variable: last_fig ---
    global last_fig
    last_fig = plt.gcf()
    # --- Show and close figure ---
    plt.show()
    plt.close()
    print(rf"Equilibrium at y(NO2) = {yB_eq:.7f}")
# --------------------------------------------- #
def plot_3DeqPT_N2O4(T,P):

    refdata,molecules,nus,n_0 = load_n2o4_2no2()[0:4]

    # calculate data for each plot
    Z1 = get_xieq_PT_N2O4(T,P,n_0[0],n_0[1])
    Z2 = general.get_G_PT(Z1,P,T,n_0,nus,refdata)/(R*T)

    # Create figure with two subplots
    # title1 = r'$(a) \;\; z: \;\; \xi_{\mathrm{eq}} \;\; \text{(mol)}$'
    # title2 = r'$(b) \;\; z: \;\; [G(\xi_{\mathrm{eq}}) - G(0)] / RT  \;\; \text{(mol)}$'
    title1 = "(a)  z:  <i>ξ</i><sub>eq</sub> (mol)"
    title2 = "(b)  z:  [<i>G</i>(<i>ξ</i><sub>eq</sub>) − <i>G</i>(0)] / <i>RT</i>  (mol)"
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=(title1,title2))

    fig.add_trace(go.Surface(x=P/1E5, y=T, z=Z1, colorscale='Viridis', showscale=False),row=1, col=1)
    fig.add_trace(go.Surface(x=P/1E5, y=T, z=Z2, colorscale='Plasma' , showscale=False),row=1, col=2)

    # formating
    fig.update_layout(
        width=1000, height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
          xaxis=dict(title=dict(text='p (bar)',font=dict(size=18))),
          yaxis=dict(title=dict(text='T (K)'  ,font=dict(size=18))),
          zaxis=dict(title=dict(text=''       ,font=dict(size=18))),
          domain=dict(x=[0.00, 0.50], y=[0, 1]),
          camera=dict(eye=dict(x=1.6, y=1.6, z=0.9))),
        scene2=dict(
          xaxis=dict(title=dict(text='p (bar)',font=dict(size=18))),
          yaxis=dict(title=dict(text='T (K)'  ,font=dict(size=18))),
          zaxis=dict(title=dict(text=''       ,font=dict(size=18))),
          domain=dict(x=[0.50, 1.00], y=[0, 1]),
          camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)))
    )

    # Ajust position of plot titles
    for ann in fig['layout']['annotations']:
        ann['y'] -= 0.15
        ann['font'] = dict(size=16)

    fig.show(config={"toImageButtonOptions": {"format": "svg","filename": "equilibrium_surfaces","scale": 2}})
# ============================================= #

# ============================================= #
# ---- Specific for N2O4 <-> 2NO2 (PART 2) ---- #
# ============================================= #
def optimize_and_freqs_n2o4(molecule,UNPAIREDS,CHARGES,key,DFTGRID,GEOMINFO):
    print(rf" * Molecule: {molecule:s}")
    dftdata = general.optimize_and_freqs(molecule,UNPAIREDS[molecule],CHARGES[molecule],key[0],key[1],DFTGRID,bsym=True)
    print("")
    # visualize molecule
    xyz_opt = general.files_of_interest(molecule,key[0],key[1],DFTGRID)[1]
    view    = general.create_visualization_xyz(xyz_opt)
    general.show_indented(view, indent_px=50)
    # download button
    general.pyscf_download(molecule,key[0],key[1],DFTGRID,[1])
    # print geometric info
    geominfo = general.geometric_info_xyz(xyz_opt,GEOMINFO[molecule])
    print(geominfo)
    # print information for partition functions
    general.pyscf_printdata(dftdata)
    print("")
    return dftdata
# ============================================= #


# ============================================= #
# ---- Specific for N2O4 <-> 2NO2 (PART 3) ---- #
# ============================================= #
def get_constants_N2O4(T):
    # Gibbs free energy from experimental data
    DGo   = general.get_DGo(T,load_n2o4_2no2()[0])
    # Equilibrium constant(s)
    Kp_o  = np.exp(-DGo/R/T)
    Kc_o  = Kp_o * P_o/(c_o*R*T) # adimensional
    Kc    = Kc_o * c_o           # mol/m^3
    # Forward rate constant
    kfw   = ARRHENIUS_A * np.exp(-ARRHENIUS_B/T)
    # Backward rate constant
    kbw  = kfw/Kc
    # String with data
    string  = rf"   * Constants for the reaction at {T:.2f} K" + "\n"
    string += "\n"
    string += rf"     equilibrium constant   Kp^o = {Kp_o:.3E}" + "\n"
    string += rf"     equilibrium constant   Kc^o = {Kc_o:.3E}" + "\n"
    string += rf"     forward  rate constant kfw  = {kfw:.3E} 1/s" + "\n"
    string += rf"     backward rate constant kbw  = {kbw:.3E} m^3 / mol / s" + "\n"
    string += "\n"
    # return data
    return DGo,Kp_o,Kc_o,kfw,kbw,string
# --------------------------------------------- #
def xi_to_data_N2O4(xi,T0,P0,V0,yA0,scenario):
    # values at xi=0
    n0 = P0*V0/(R*T0)
    nA0 = yA0*n0
    nB0 = n0-nA0
    # values at xi
    nA = nA0 -   xi
    nB = nB0 + 2*xi
    n  = n0  +   xi
    yA = nA/n
    yB = nB/n
    if   "VT" in scenario: P,V = n*R*T0/V0, V0
    elif "PT" in scenario: P,V = P0 , n*R*T0/P0
    else: raise Exception
    cA = nA/V
    cB = nB/V
    pA = yA*P
    pB = yB*P
    # quotient of reaction
    Qp = np.where(pA == 0,np.inf,(pB/P_o)**2 / (pA/P_o))
    # energy of interest
    if "VT" in scenario: energy = get_A_VT_N2O4(xi,V0,T0,nA0,nB0)
    if "PT" in scenario: energy = get_G_PT_N2O4(xi,P0,T0,nA0,nB0)
    # refdata = load_n2o4_2no2()[0]
    # if "PT" in scenario: energy = get_G_PT(xi,P0,T0,np.array([nA0,nB0]),np.array([-1,2]),refdata)
    # if "VT" in scenario: energy = get_A_VT(xi,V0,T0,np.array([nA0,nB0]),np.array([-1,2]),refdata)
    # return data
    return (nA,nB),(pA,pB),(yA,yB),(cA,cB),(n,P,V),Qp,energy
# --------------------------------------------- #
def xi2time_PT_N2O4(xi,xi1,xi2,kfw,alpha):
    beta  = kfw + 4*alpha
    s     = np.sqrt(kfw/beta)
    term1 = (xi1-xi)/xi1
    term2 = (xi2-xi)/xi2
    texp  = term1**(1+s) / term2**(1-s)
    t     = -np.log(texp)/(2*s*beta)
    return t
# --------------------------------------------- #
def xi2time_VT_N2O4(xi,xi1,xi2,kbw,V0):
    t  = np.log( abs(xi1/xi2 * (xi-xi2)/(xi-xi1))  )
    t *= V0/(4*kbw*(xi1-xi2))
    return t
# --------------------------------------------- #
def get_G_PT_N2O4(xi,P,T,nA0,nB0):
    #Delta_r{G}^o(T)
    DGo_T   = get_constants_N2O4(T)[0]
    #Delta_r{G}^*(T)
    DGast   = DGo_T + R*T*np.log(P/P_o)
    # mix term
    n0      = nA0 + nB0
    nA      = nA0 -   xi
    nB      = nB0 + 2*xi
    n       = nA  + nB
    DDGmix  = 0.0
    if nA  != 0.0: DDGmix += nA *np.log(nA /n )
    if nB  != 0.0: DDGmix += nB *np.log(nB /n )
    if nA0 != 0.0: DDGmix -= nA0*np.log(nA0/n0)
    if nB0 != 0.0: DDGmix -= nB0*np.log(nB0/n0)
    DDGmix *= R*T
    # return G(xi) - G(0)
    DGtot   = DGast * xi + DDGmix
    return DGtot
# --------------------------------------------- #
def get_A_VT_N2O4(xi,V,T,nA0,nB0):
    # initial conditions
    n0      = nA0 + nB0
    p0      = n0*R*T/V
    # current conditions
    nA      = nA0 -   xi
    nB      = nB0 + 2*xi
    n       = nA  + nB
    p       = n *R*T/V
    #Delta_r{G}^o(T)
    DGo_T   = get_constants_N2O4(T)[0]
    # pressure term
    termP   = p *np.log(p/P_o/np.e)
    termP  -= p0*np.log(p0/P_o/np.e)
    # mix term
    DDGmix  = 0.0
    if nA  != 0.0: DDGmix += nA *np.log(nA /n )
    if nB  != 0.0: DDGmix += nB *np.log(nB /n )
    if nA0 != 0.0: DDGmix -= nA0*np.log(nA0/n0)
    if nB0 != 0.0: DDGmix -= nB0*np.log(nB0/n0)
    DDGmix *= R*T
    # return A(xi) - A(0)
    DAtot   = DGo_T * xi + V*termP + DDGmix
    return DAtot
# --------------------------------------------- #
def datatoinfo_N2O4(T0,p0,V0,yA0,xi,scenario):
    # constants
    Kp_o = get_constants_N2O4(T0)[1]
    # calculate more magnitudes
    (nA,nB),(pA,pB),(yA,yB),(cA,cB),(n,P,V),Qp,E = xi_to_data_N2O4(xi,T0,p0,V0,yA0,scenario)
    # string with information
    string  = rf"       (P , V , T) = ({P*1E-5:6.2f} bar , {V*1E3:6.2f} L , {T0:6.2f} K)"+"\n"
    string += "\n"
    string += rf"       num. moles  = {n:6.3f} mol"+"\n"
    string += rf"       extent (xi) = {xi:6.3f} mol"+"\n"
    string += "\n"
    if "VT" in scenario: string += rf"       A(xi)-A(0)  = {E/(R*T0):8.2E}*(RT) mol"+"\n"
    if "PT" in scenario: string += rf"       G(xi)-G(0)  = {E/(R*T0):8.2E}*(RT) mol"+"\n"
    string += "\n"
    string += rf"       data for N2O4"+"\n"
    string += rf"         - number of moles  = {nA:6.3f} mol"+"\n"
    string += rf"         - mole fraction    = {yA:6.3f} mol"+"\n"
    string += rf"         - partial pressure = {pA*1E-5:6.3f} bar"+"\n"
    string += rf"         - concentration    = {cA/1000:8.2E} M"+"\n"
    string += "\n"
    string += rf"       data for NO2"+"\n"
    string += rf"         - number of moles  = {nB:6.3f} mol"+"\n"
    string += rf"         - mole fraction    = {yB:6.3f} mol"+"\n"
    string += rf"         - partial pressure = {pB*1E-5:6.3f} bar"+"\n"
    string += rf"         - concentration    = {cB/1000:8.2E} M"+"\n"
    string += "\n"
    if pA == 0.0: string += rf"       ==> Qp^o = infinity"+"\n"
    else        : string += rf"       ==> Qp^o = {(pB*pB)/(pA*P_o):.3E}"+"\n"
    string += rf"       ==> Kp^o = {Kp_o:.3E}"+"\n"
    string += "\n"
    return string
# --------------------------------------------- #
def kinetics_N2O4(T0,P0,V0,yA0,scenario,arrhenius):
    # --- global variable to update ---
    global last_info
    global ARRHENIUS_A
    global ARRHENIUS_B
    ARRHENIUS_A,ARRHENIUS_B = arrhenius
    # which scenario
    if   "PT" in scenario: scenario = "PT"
    elif "VT" in scenario: scenario = "VT"
    else: raise Exception
    # Get data for this reaction
    DGo,Kp_o,Kc_o,kfw,kbw,STRING = get_constants_N2O4(T0)
    # Determine xieq, xi1 and xi2
    (nA0,nB0) = xi_to_data_N2O4(0,T0,P0,V0,yA0,scenario)[0]
    if scenario == "VT":
       xieq = get_xieq_VT_N2O4(T0,V0,nA0,nB0)
       Kc   = kfw/kbw
       lamb = 4*(kbw/V0)
       a0   = (nB0**2/4 - nA0*V0*Kc/4)
       a1   = (nB0      +     V0*Kc/4)
       xi1  = (-a1 + np.sqrt(a1**2-4*a0))/2
       xi2  = (-a1 - np.sqrt(a1**2-4*a0))/2
    if scenario == "PT":
       xieq  = get_xieq_PT_N2O4(T0,P0,nA0,nB0)
       alpha = kbw*P0/(R*T0)
       beta  = kfw + 4*alpha
       s     = np.sqrt(kfw/beta)
       xi1   = (-nB0+(2*nA0+nB0)*s)/2
       xi2   = (-nB0-(2*nA0+nB0)*s)/2
    # Calculate from xi = 0 to xieq
    xis    = np.linspace(0,REL_XI_EQ*xieq,NPOINTSXI)
    if scenario == "VT": times = [xi2time_VT_N2O4(xi,xi1,xi2,kbw,V0   ) for xi in xis]
    if scenario == "PT": times = [xi2time_PT_N2O4(xi,xi1,xi2,kfw,alpha) for xi in xis]
    # for t_i,xi_i in zip(times,xis): print(t_i,xi_i)
    # String with information
    STRING += rf"   * Initial conditions:"+"\n\n"
    STRING += datatoinfo_N2O4(T0,P0,V0,yA0,0   ,scenario)
    STRING += rf"   * At equilibrium:"+"\n\n"
    STRING += datatoinfo_N2O4(T0,P0,V0,yA0,xieq,scenario)
    last_info = STRING
    # plot data
    plot_kinetics_N2O4(np.array(times),xis,xieq,T0,P0,V0,yA0,scenario)
# ============================================= #
def plot_kinetics_N2O4(times,xis,xieq,T0,P0,V0,yA0,scenario):

    # select good units for time (among secs, milisecs, microsecs and nanosecs)
    for unitst,factor in [("s",1E0) , ("ms",1E3) , ("$\\mu$s",1E6) , ("ns",1E9)]:
        last_t = times[-1]*factor
        if last_t > 0.5: break

    dataeq = xi_to_data_N2O4(xieq,T0,P0,V0,yA0,scenario)
    yA,yB = [],[]
    Qp    = []
    AG    = []
    V     = []
    P     = []
    nA    = []
    for xi_i in xis:
        tni,tpi,tyi,tci,(n_i,P_i,V_i),Qp_i,E_i = xi_to_data_N2O4(xi_i,T0,P0,V0,yA0,scenario)
        nA.append(tni[0])
        yA.append(tyi[0])
        yB.append(tyi[1])
        Qp.append(Qp_i)
        AG.append(E_i)
        V.append(V_i)
        P.append(P_i)

    plt.rcParams['text.usetex'] = True
    fig, axs = plt.subplots(2, 3, figsize=(12,6))
    fig.suptitle(rf'$(P,V,T)_0 = ({P0*1E-5:.2f} \; {{\rm bar}},{V0*1E3:.2f} \; {{\rm L}},{T0:.2f} \; {{\rm K}})$; $y_{{\rm N_2O_4}}(0)={yA0:.2f}$', fontsize=FONTSIZE[2])
    # -------------------------------------
    # (a) Population
    # -------------------------------------
    axs[0, 0].plot(times*factor,yA,color='k',label=r'i=N$_2$O$_4$')
    axs[0, 0].axhline(y=dataeq[2][0],color="k",ls=":",zorder=1)

    axs[0, 0].plot(times*factor,yB,color='r',label=r'i=NO$_2$')
    axs[0, 0].axhline(y=dataeq[2][1],color="r",ls=":",zorder=1)

    axs[0, 0].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    axs[0, 0].set_ylabel(r'$y_i$',fontsize=FONTSIZE[2])
    axs[0, 0].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[0, 0].set_title('(a)')
    axs[0, 0].legend(frameon=False)

    # -------------------------------------
    # (b) ξ vs time
    # -------------------------------------
    nA0,nB0 = xi_to_data_N2O4(0,T0,P0,V0,yA0,scenario)[0]
    xlim1   = -nB0/2
    xlim2   = +nA0
    axs[0, 1].plot(times*factor,xis,ls='-',color='k')
    axs[0, 1].axhline(y=xieq,ls=":",color="k",zorder=1)

    axs[0, 1].set_ylabel('$\\xi$ (mol)',fontsize=FONTSIZE[2])
    # axs[0, 1].set_ylim(xlim1,xlim2)
    axs[0, 1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    axs[0, 1].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[0, 1].set_title('(b)')

    # -------------------------------------
    # (c) P(or V) vs time
    # -------------------------------------
    if "PT" in scenario:
        axs[0, 2].plot(times*factor,np.array(V)*1E3 ,ls='-',color='k')
        axs[0, 2].set_ylabel('$V$ (L)',fontsize=14)
    if "VT" in scenario:
        axs[0, 2].plot(times*factor,np.array(P)*1E-5,ls='-',color='k')
        axs[0, 2].set_ylabel('$P$ (bar)',fontsize=FONTSIZE[2])
    axs[0, 2].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[0, 2].set_title('(c)')

    # -------------------------------------
    # (d) Q vs time
    # -------------------------------------
    xx,yy=factor*times[1:], Qp[1:]
    axs[1, 0].plot(xx,yy,ls='-',color='k',zorder=2)
    axs[1, 0].axhline(y=dataeq[5],ls=":" ,color="k",zorder=1)
    axs[1, 0].set_ylabel('$Q_p^\\circ$',fontsize=FONTSIZE[2])
    axs[1, 0].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[1, 0].set_title('(d)')

    # -------------------------------------
    # (e) dξ/dt vs time
    # -------------------------------------
    Kpo,Kco,kfw = get_constants_N2O4(T0)[1:4]
    dxidt_ana       = kfw*np.array(nA)*(1-Qp/Kpo)
    axs[1, 1].plot(times*factor,dxidt_ana/factor,ls='-',color='k')
    # dxidt_num       = np.gradient(xis,times)
    # axs[1, 1].plot(times*factor,dxidt_num/factor,'rx')
    axs[1, 1].set_ylabel(rf'd$\xi$/d$t$ (mol/{unitst:s})',fontsize=FONTSIZE[2])
    axs[1, 1].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[1, 1].set_title('(e)')

    # -------------------------------------
    # (f) G or A vs time
    # -------------------------------------
    yy = [Ei/(R*T0) for Ei in AG]
    axs[1, 2].plot(times*factor,yy,color='k')
    axs[1, 2].axhline(y=dataeq[6]/(R*T0),ls=":",color="k",zorder=1)
    if "PT" in scenario: key = "G"
    if "VT" in scenario: key = "A"
    axs[1, 2].set_ylabel(rf'$\left({key:s}(\xi)-{key:s}(0)\right) / (RT)$ (mol)',fontsize=FONTSIZE[2])
    axs[1, 2].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[1, 2].set_title('(f)')

    # --- update global variable: last_fig ---
    fig.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    global last_fig
    last_fig = fig
    # --- Show and close figure ---
    # fig.set_size_inches(9.0,6.6)
    plt.show()
    plt.close(fig)
# ============================================= #

