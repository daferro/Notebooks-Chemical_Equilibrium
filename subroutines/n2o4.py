# ============================================== #
import numpy                as np                #
# ---------------------------------------------- #
import matplotlib.pyplot    as plt               #
import matplotlib.ticker    as mticker           #
import plotly.graph_objects as go                #
from   plotly.subplots      import make_subplots #
# ---------------------------------------------- #
from   constants            import P_o,c_o,R     #
from   constants            import NPOINTSXI     #
from   constants            import REL_XI_EQ     #
from   constants            import ZERO4         #
from   constants            import FONTSIZE      #
# ---------------------------------------------- #
import general
# ---------------------------------------------- #
# ============================================== #

NPOINTSYB   = 51   # number of yB points (thermo & kinetics)
last_info   = None
last_fig    = None
ARRHENIUS_A = None
ARRHENIUS_B = None
TFACTORS    = [("years"  ,1/3600/24/365)]
TFACTORS   += [("days"   ,1/3600/24    )]
TFACTORS   += [("hours"  ,1/3600       )]
TFACTORS   += [("min"    ,1/60        )]
TFACTORS   += [("s"      ,1           )]
TFACTORS   += [("ms"     ,1E3         )]
TFACTORS   += [("$\\mu$s",1E6         )]
TFACTORS   += [("ns"     ,1E9         )]

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
    xx2 = [0.0,1.0]
    yy2 = [m*xx_i + b for xx_i in xx2]
    # y-limits: fixed for a given (T,P) and wide enough to show both intercepts
    # of the tangent at equilibrium, i.e. the points where it crosses the
    # lines y(NO2) = 0 and y(NO2) = 1
    m_eq,b_eq = intercept_getline_N2O4(T,P,yB_eq)[0:2]
    ylim1 = min(min(lGm),b_eq,m_eq+b_eq)
    ylim2 = max(max(lGm),b_eq,m_eq+b_eq)
    delta = ylim2-ylim1
    ylim1 = ylim1 - delta*0.15
    ylim2 = ylim2 + delta*0.15
    ax2.plot(list_yB,lGm,'k-')
    ax2.plot(xx2,yy2,'r--',label=rf"${m:+.5f} \cdot y_{{\rm NO_2}} {b:+.5f}$")
    ax2.plot(xx1,yy1,'ro')
    # intercepts of the tangent: chemical potentials (relative to the reference)
    for xx_i,yy_i in zip(xx2,yy2):
        if ylim1 <= yy_i <= ylim2:
           ax2.plot(xx_i,yy_i,ls='none',marker='o',mfc='w',mec='r',mew=1.5,ms=8,clip_on=False,zorder=5)
    for ii in ["x","y"]: ax2.tick_params(axis=ii,labelsize=FONTSIZE[1])
    ax2.set_xlim( 0 , 1 )
    ax2.set_ylim(ylim1,ylim2)
    ax2.set_xlabel(r"$y_{\rm NO_2}$",fontsize=FONTSIZE[2])
    ax2.set_ylabel(r"$[G(y_{\rm NO_2})- G(0)]/n \cdot (RT)^{-1}$",fontsize=FONTSIZE[2])
    ax2.legend(loc="upper center",fontsize=FONTSIZE[0])
    # --- update global variable: last_fig ---
    global last_fig
    last_fig = plt.gcf()
    # --- Show and close figure ---
    plt.show()
    plt.close()
    # --- Information on the tangent (intercept method) ---
    print(rf"Tangent to Gm at y(NO2) = {yB:.6f}   [Gm is referred to G(0); values in units of RT]")
    print(rf"   * slope                               = {m:+.5f}")
    print(rf"   * intercept at y(NO2) = 0 [ ~ mu(N2O4)] = {b:+.5f}")
    print(rf"   * intercept at y(NO2) = 1 [ ~ mu(NO2) ] = {m+b:+.5f}")
    print(rf"   * 2*mu(NO2) - mu(N2O4)  (= -affinity)  = {2*(m+b)-b:+.5f}")
    print("")
    print(rf"Equilibrium requires mu(N2O4) = 2*mu(NO2): the intercept at y(NO2) = 1 must be")
    print(rf"one-half of the intercept at y(NO2) = 0 (zero affinity).")
    print(rf"This happens at y(NO2) = {yB_eq:.7f}")
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
# ---- Self-checks of the work done by hand ---- #
# ============================================= #
def selfcheck_thermo298():
    """Exercise 1: reaction magnitudes at 298 K obtained by hand from tabulated data."""
    refdata = load_n2o4_2no2()[0]
    DHo,DSo,DCPo,TREF = refdata
    DGo = general.get_DGo(TREF,refdata)
    Kp  = np.exp(-DGo/R/TREF)
    print("Enter the values you obtained BY HAND from the tabulated data (298 K).")
    print("The notebook only tells you whether they agree with its own values.")
    print("")
    res  = []
    res += [general.check_value("Delta_r H^o"    ,DHo/1000,"kJ/mol"   ,atol=0.06)]
    res += [general.check_value("Delta_r S^o"    ,DSo     ,"J/(mol K)",atol=0.15)]
    # 4.80 kJ/mol from Delta_r H^o - T*Delta_r S^o; 4.73 kJ/mol from tabulated Delta_f G^o
    res += [general.check_value("Delta_r G^o"    ,DGo/1000,"kJ/mol"   ,atol=0.09,
            hint="Delta_r G^o = Delta_r H^o - T*Delta_r S^o; mind kJ and J.")]
    res += [general.check_value("K_p^o"          ,Kp      ,""         ,rtol=0.04,
            hint="K_p^o = exp(-Delta_r G^o/RT), with Delta_r G^o in J/mol.")]
    res += [general.check_value("Delta_r C_p^o/R",DCPo/R  ,""         ,atol=0.01,
            hint="sum of nu_i*C_p,m,i with the stoichiometric coefficients, divided by R.")]
    general.summary_checks(res)
# --------------------------------------------- #
def selfcheck_xieq():
    """Exercise 3: extent of reaction at equilibrium (1 mol N2O4, 298 K, 1 bar)."""
    xieq = get_xieq_PT_N2O4(298.,1E5,1.0,0.0)
    print("Enter the equilibrium extent of reaction you obtained BY HAND")
    print("(1.00 mol of N2O4 initially, 298 K, constant pressure of 1 bar).")
    print("")
    res = [general.check_value("xi_eq",xieq,"mol",atol=0.003,
           hint="write K_p^o in terms of xi_eq (mole fractions times p/p^o) and solve for xi_eq.")]
    general.summary_checks(res)
# --------------------------------------------- #
def selfcheck_cp_limits():
    """Exercise 8: Delta_r C_p^o from equipartition in the two limiting cases."""
    print("Enter the values of Delta_r C_p^o/R you obtained BY HAND with the equipartition expression.")
    print("")
    res  = []
    res += [general.check_value("all rotations and vibrations fully active: Delta_r C_p^o/R",-2.0,"",atol=0.01,
            hint="a nonlinear molecule with N atoms has 3 rotations and 3N-6 vibrations.")]
    res += [general.check_value("vibrations neglected                     : Delta_r C_p^o/R",+4.0,"",atol=0.01,
            hint="without vibrations every nonlinear molecule has C_p,m = 4R.")]
    general.summary_checks(res)
# --------------------------------------------- #
def selfcheck_rate_constants(arrhenius,T=298.):
    """Exercise 9: forward and backward rate coefficients (with units)."""
    global ARRHENIUS_A, ARRHENIUS_B
    ARRHENIUS_A,ARRHENIUS_B = arrhenius
    DGo,Kp_o,Kc_o,kfw,kbw,string = get_constants_N2O4(T)
    print(rf"Enter the rate coefficients you obtained BY HAND at {T:.0f} K, in the units indicated.")
    print("")
    res  = []
    res += [general.check_value("k_fw",kfw     ,"1/s"      ,rtol=0.03,
            hint="k_fw = A*exp(-B/T).")]
    res += [general.check_value("k_bw",kbw*1000,"L/(mol s)",rtol=0.05,
            hint="k_bw = k_fw/K_c, with K_c = K_c^o * c^o and K_c^o = K_p^o * p^o/(c^o*R*T).")]
    general.summary_checks(res)
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
# ---- Reference molecular data (PART 2)   ---- #
# ============================================= #
# Molecular data obtained beforehand with the routines of this notebook
# (general.optimize_and_freqs; PySCF 2.11.0; 6-31G* basis set; DFT grid level 5).
# They allow the statistical-mechanical part to be carried out without running
# the electronic-structure calculations. Units: mass (amu), rotational constants (Hz),
# wavenumbers (1/m), E0 and ZPE (hartree). A fresh calculation may differ slightly
# (about 0.1 kJ/mol in Delta_r G^o), depending on the grid level and software version.
REFERENCE_DFT_INFO = "PySCF 2.11.0, 6-31G* basis set, DFT grid level 5"
REFERENCE_DFT = {}
REFERENCE_DFT[("b3lyp-d4","6-31g*")] = {
    "N2O4": {
        "natoms"  : 6,
        "mass"    : 92.010000,
        "rotcons" : [6.477259E+09, 3.645212E+09, 2.332532E+09],
        "islinear": False,
        "freqs"   : [9728.5, 22803.4, 30055.3, 43705.8, 50100.0, 68944.8, 75724.1, 84040.7, 133438.6, 146399.7, 183708.6, 186302.1],
        "ZPE"     : 0.02357793,
        "unpaired": 0,
        "charge"  : 0,
        "E0"      : -410.16703541,
        "xyz"     : [("O",-1.349000,1.104172,0.000000), ("O",1.349000,1.104172,0.000000), ("O",-1.349000,-1.104172,0.000000), ("O",1.349000,-1.104172,0.000000), ("N",-0.889838,0.000000,0.000000), ("N",0.889838,-0.000000,0.000000)],
    },
    "NO2": {
        "natoms"  : 3,
        "mass"    : 46.005000,
        "rotcons" : [2.345876E+11, 1.289036E+10, 1.221894E+10],
        "islinear": False,
        "freqs"   : [74959.5, 140638.1, 172660.0],
        "ZPE"     : 0.00884516,
        "unpaired": 1,
        "charge"  : 0,
        "E0"      : -205.06973401,
        "xyz"     : [("N",-0.000000,-0.000957,-0.000000), ("O",1.106916,0.469289,0.000000), ("O",-1.106916,0.469289,-0.000000)],
    },
}
REFERENCE_DFT[("o3lyp-d4","6-31g*")] = {
    "N2O4": {
        "natoms"  : 6,
        "mass"    : 92.010000,
        "rotcons" : [6.504970E+09, 3.566799E+09, 2.303659E+09],
        "islinear": False,
        "freqs"   : [9322.2, 21657.8, 28129.5, 43104.5, 49001.9, 68325.5, 75718.3, 84106.9, 134546.3, 148195.6, 186841.6, 189102.9],
        "ZPE"     : 0.02364858,
        "unpaired": 0,
        "charge"  : 0,
        "E0"      : -410.06535386,
        "xyz"     : [("O",-1.362638,1.101817,0.000000), ("O",1.362638,1.101817,0.000000), ("O",-1.362638,-1.101817,0.000000), ("O",1.362638,-1.101817,0.000000), ("N",-0.903399,0.000000,0.000000), ("N",0.903399,-0.000000,0.000000)],
    },
    "NO2": {
        "natoms"  : 3,
        "mass"    : 46.005000,
        "rotcons" : [2.357613E+11, 1.292175E+10, 1.225032E+10],
        "islinear": False,
        "freqs"   : [75362.0, 141528.5, 175828.6],
        "ZPE"     : 0.00894680,
        "unpaired": 1,
        "charge"  : 0,
        "E0"      : -205.01942708,
        "xyz"     : [("N",-0.000000,-0.000093,-0.000000), ("O",1.105571,0.468980,0.000000), ("O",-1.105571,0.468980,-0.000000)],
    },
}
REFERENCE_DFT[("pbe0-d4","6-31g*")] = {
    "N2O4": {
        "natoms"  : 6,
        "mass"    : 92.010000,
        "rotcons" : [6.563188E+09, 3.809745E+09, 2.410511E+09],
        "islinear": False,
        "freqs"   : [9880.8, 25275.1, 32708.3, 46475.4, 53223.9, 73274.6, 77995.2, 86001.8, 138629.8, 150660.3, 191477.3, 194573.6],
        "ZPE"     : 0.02460822,
        "unpaired": 0,
        "charge"  : 0,
        "E0"      : -409.75002583,
        "xyz"     : [("O",-1.320985,1.096920,0.000000), ("O",1.320985,1.096920,0.000000), ("O",-1.320985,-1.096920,0.000000), ("O",1.320985,-1.096920,0.000000), ("N",-0.865419,0.000000,0.000000), ("N",0.865419,-0.000000,0.000000)],
    },
    "NO2": {
        "natoms"  : 3,
        "mass"    : 46.005000,
        "rotcons" : [2.423270E+11, 1.305672E+10, 1.238919E+10],
        "islinear": False,
        "freqs"   : [76657.3, 145949.1, 181593.5],
        "ZPE"     : 0.00920835,
        "unpaired": 1,
        "charge"  : 0,
        "E0"      : -204.86015716,
        "xyz"     : [("N",0.000000,0.004115,0.000000), ("O",1.099841,0.466790,0.000000), ("O",-1.099841,0.466790,0.000000)],
    },
}
# --------------------------------------------- #
def load_reference_dft(molecule,UNPAIREDS,CHARGES,key,DFTGRID,GEOMINFO):
    """Alternative to optimize_and_freqs_n2o4: returns the molecular data of a
    previous calculation (REFERENCE_DFT) instead of running PySCF.

    The charge and the number of unpaired electrons entered by the student must
    be the right ones, and the rotational symmetry number is left unassigned
    (sigma = 1): it has to be set in the following section of the notebook."""
    print(rf" * Molecule: {molecule:s}")
    if key not in REFERENCE_DFT or molecule not in REFERENCE_DFT[key]:
       print(rf"     - no reference data for {molecule:s} at this level; run the calculation instead")
       return None
    ref = REFERENCE_DFT[key][molecule]
    # the student still has to know the electronic state of each species
    if int(CHARGES[molecule]) != ref["charge"] or int(UNPAIREDS[molecule]) != ref["unpaired"]:
       print(rf"     - the charge and/or the number of unpaired electrons you entered for {molecule:s}")
       print(rf"       do not correspond to its electronic ground state. Think about its Lewis")
       print(rf"       structure, run again the cell where they are defined and come back.")
       print("")
       return None
    print(rf"     - reference data loaded ({REFERENCE_DFT_INFO:s}); no calculation was run")
    # optimized geometry: write it, show it and print the geometric parameters
    xyz_opt = general.files_of_interest(molecule,key[0],key[1],DFTGRID)[1]
    symbols = [atom[0] for atom in ref["xyz"]]
    coords  = np.array([atom[1:] for atom in ref["xyz"]])
    try:
       general.data_2_xyz(symbols,coords,xyz_opt)
       try:
          view = general.create_visualization_xyz(xyz_opt)
          general.show_indented(view, indent_px=50)
       except Exception: pass
       print(general.geometric_info_xyz(xyz_opt,GEOMINFO[molecule]))
    except Exception: pass
    # data for the partition functions
    dftdata = {k:ref[k] for k in ("natoms","mass","islinear","ZPE","unpaired","E0")}
    dftdata["rotcons" ] = list(ref["rotcons"])
    dftdata["freqs"   ] = list(ref["freqs"])
    dftdata["rotsigma"] = 1   # not assigned: to be set by the student
    general.pyscf_printdata(dftdata)
    print("")
    return dftdata
# --------------------------------------------- #
def export_reference_dft(DFTDATA,LEVELS):
    """Prints the molecular data stored in DFTDATA in the format of REFERENCE_DFT,
    so that instructors can replace the reference data by those of their own run."""
    for key in LEVELS:
        print(rf'REFERENCE_DFT[("{key[0]:s}","{key[1]:s}")] = {{')
        for molecule in DFTDATA.keys():
            if key not in DFTDATA[molecule]: continue
            d = DFTDATA[molecule][key]
            print(rf'    "{molecule:s}": {{')
            print(rf'        "natoms"  : {d["natoms"]:d},')
            print(rf'        "mass"    : {d["mass"]:.6f},')
            print(rf'        "rotcons" : [' + ", ".join(["None" if x is None else "%.6E"%x for x in d["rotcons"]]) + "],")
            print(rf'        "islinear": {str(d["islinear"]):s},')
            print(rf'        "freqs"   : [' + ", ".join(["%.1f"%x for x in d["freqs"]]) + "],")
            print(rf'        "ZPE"     : {d["ZPE"]:.8f},')
            print(rf'        "unpaired": {d["unpaired"]:d},')
            print(rf'        "E0"      : {d["E0"]:.8f},')
            print("    },")
        print("}")
# ============================================= #


# ============================================= #
# ---- Specific for N2O4 <-> 2NO2 (PART 3) ---- #
# ============================================= #
def factor_for_time(t_i):
    for unitst,factor in TFACTORS:
        last_t = t_i*factor
        if last_t > 0.5: break
    return unitst,factor
# --------------------------------------------- #
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
    string += rf"     - equilibrium constant   Kp^o = {Kp_o:.3E}" + "\n"
    string += rf"     - equilibrium constant   Kc^o = {Kc_o:.3E}" + "\n"
    string += rf"     - forward  rate constant kfw  = {kfw:.3E} 1/s" + "\n"
    string += rf"     - backward rate constant kbw  = {kbw:.3E} m^3 / mol / s" + "\n"
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
    if pA == 0: Qp = np.inf
    else      : Qp = (pB/P_o)**2 / (pA/P_o)
    #Qp = np.where(pA == 0,np.inf,(pB/P_o)**2 / (pA/P_o))
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
    string  = rf"       (P,V,T)     = ({P*1E-5:.2f} bar, {V*1E3:.2f} L, {T0:.2f} K)"+"\n"
    string += rf"       num. moles  = {n:6.3f} mol"+"\n"
    string += rf"       extent (xi) = {xi:6.3f} mol"+"\n"
    if "VT" in scenario: string += rf"       A(xi)-A(0)  = {E/(R*T0):10.3E}*(RT) mol"+"\n"
    if "PT" in scenario: string += rf"       G(xi)-G(0)  = {E/(R*T0):10.3E}*(RT) mol"+"\n"
    if pA == 0.0       : string += rf"       Qp^o        =  infinity"+"\n"
    else               : string += rf"       Qp^o        = {(pB*pB)/(pA*P_o):10.3E}"+"\n"
    string += "\n"

    string += rf"       data for N2O4"+"\n"
   #string += rf"         - number of moles  = {nA:6.3f} mol"+"\n"
    string += rf"         - mole fraction    = {yA:6.3f} mol"+"\n"
    string += rf"         - partial pressure = {pA*1E-5:6.3f} bar"+"\n"
    string += rf"         - concentration    = {cA/1000:9.2E} M"+"\n"
    string += "\n"

    string += rf"       data for NO2"+"\n"
   #string += rf"         - number of moles  = {nB:6.3f} mol"+"\n"
    string += rf"         - mole fraction    = {yB:6.3f} mol"+"\n"
    string += rf"         - partial pressure = {pB*1E-5:6.3f} bar"+"\n"
    string += rf"         - concentration    = {cB/1000:9.2E} M"+"\n"
    string += "\n"
    return string
# --------------------------------------------- #
def plot_kinetics_N2O4(times,xis,xieq,T0,P0,V0,yA0,scenario):

    # select good units for time (among secs, milisecs, microsecs and nanosecs)
    unitst,factor = factor_for_time(times[-1])
   #for unitst,factor in [("s",1E0) , ("ms",1E3) , ("$\\mu$s",1E6) , ("ns",1E9)]:
   #    last_t = times[-1]*factor
   #    if last_t > 0.5: break

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
   #fig, axs = plt.subplots(2, 3, figsize=(12,6))
    fig, axs = plt.subplots(3, 2, figsize=(8,9))
    fig.suptitle(rf'$(P,V,T)_0 = ({P0*1E-5:.2f} \; {{\rm bar}},{V0*1E3:.2f} \; {{\rm L}},{T0:.2f} \; {{\rm K}})$; $y_{{\rm N_2O_4}}(0)={yA0:.2f}$', fontsize=FONTSIZE[2])

    # -------------------------------------
    # (a) ξ vs time [0,0]
    # -------------------------------------
    nA0,nB0 = xi_to_data_N2O4(0,T0,P0,V0,yA0,scenario)[0]
    xlim1   = -nB0/2
    xlim2   = +nA0
    axs[0,0].plot(times*factor,xis,ls='-',color='k')
    axs[0,0].axhline(y=xieq,ls=":",color="k",zorder=1)

    axs[0,0].set_ylabel('$\\xi$ (mol)',fontsize=FONTSIZE[2])
    # axs[0,0].set_ylim(xlim1,xlim2)
    axs[0,0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    axs[0,0].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[0,0].set_title('(a)')

    # -------------------------------------
    # (b) dξ/dt vs time
    # -------------------------------------
    Kpo,Kco,kfw = get_constants_N2O4(T0)[1:4]
    dxidt_ana   = kfw*np.array(nA)*(1-Qp/Kpo)
    axs[0,1].plot(times*factor,dxidt_ana/factor,ls='-',color='k')
    # dxidt_num   = np.gradient(xis,times)
    # axs[1, 1].plot(times*factor,dxidt_num/factor,'rx')
    axs[0,1].set_ylabel(rf'd$\xi$/d$t$ (mol/{unitst:s})',fontsize=FONTSIZE[2])
    axs[0,1].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[0,1].set_title('(b)')

    # -------------------------------------
    # (c) molar fraction
    # -------------------------------------
    axs[1,0].plot(times*factor,yA,color='k',label=r'i=N$_2$O$_4$')
    axs[1,0].axhline(y=dataeq[2][0],color="k",ls=":",zorder=1)

    axs[1,0].plot(times*factor,yB,color='r',label=r'i=NO$_2$')
    axs[1,0].axhline(y=dataeq[2][1],color="r",ls=":",zorder=1)

    axs[1,0].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    axs[1,0].set_ylabel(r'$y_i$',fontsize=FONTSIZE[2])
    axs[1,0].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[1,0].set_title('(c)')
    axs[1,0].legend(frameon=False)

    # -------------------------------------
    # (d) P(or V) vs time
    # -------------------------------------
    if "PT" in scenario:
        axs[1,1].plot(times*factor,np.array(V)*1E3 ,ls='-',color='k')
        axs[1,1].set_ylabel('$V$ (L)',fontsize=14)
    if "VT" in scenario:
        axs[1,1].plot(times*factor,np.array(P)*1E-5,ls='-',color='k')
        axs[1,1].set_ylabel('$P$ (bar)',fontsize=FONTSIZE[2])
    axs[1,1].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[1,1].set_title('(d)')

    # -------------------------------------
    # (e) G or A vs time
    # -------------------------------------
    yy = [Ei/(R*T0) for Ei in AG]
    axs[2,0].plot(times*factor,yy,color='k')
    axs[2,0].axhline(y=dataeq[6]/(R*T0),ls=":",color="k",zorder=1)
    if "PT" in scenario: key = "G"
    if "VT" in scenario: key = "A"
    axs[2,0].set_ylabel(rf'$\left({key:s}(\xi)-{key:s}(0)\right) / (RT)$ (mol)',fontsize=FONTSIZE[2])
    axs[2,0].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[2,0].set_title('(e)')

    # -------------------------------------
    # (f) Q vs time
    # -------------------------------------
    xx,yy=factor*times[1:], Qp[1:]
    axs[2,1].plot(xx,yy,ls='-',color='k',zorder=2)
    axs[2,1].axhline(y=dataeq[5],ls=":" ,color="k",zorder=1)
    axs[2,1].set_ylabel('$Q_p^\\circ$',fontsize=FONTSIZE[2])
    axs[2,1].set_xlabel(rf'Time ({unitst:s})',fontsize=FONTSIZE[2])
    axs[2,1].set_title('(f)')

    # --- update global variable: last_fig ---
    fig.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    global last_fig
    last_fig = fig
    # --- Show and close figure ---
    # fig.set_size_inches(9.0,6.6)
    plt.show()
    plt.close(fig)
# --------------------------------------------- #
def kinetics_N2O4(T0,P0,V0,yA0,scenario,arrhenius,output_widget):
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
    # String with information
    STRING += rf"   * Initial conditions:"+"\n\n"
    STRING += datatoinfo_N2O4(T0,P0,V0,yA0,0   ,scenario)
    STRING += rf"   * At equilibrium:"+"\n\n"
    STRING += datatoinfo_N2O4(T0,P0,V0,yA0,xieq,scenario)
    # calculate time for 99.9%
    xi_given = xieq*0.999
    if scenario == "VT": time_given = xi2time_VT_N2O4(xi_given,xi1,xi2,kbw,V0   )
    if scenario == "PT": time_given = xi2time_PT_N2O4(xi_given,xi1,xi2,kfw,alpha)
    unitst,factor = factor_for_time(time_given)
    STRING += rf"   * It took {time_given*factor:.2f} {unitst} for xi to reach 0.999*xi_eq"+"\n\n"
    # update global variable for string 
    last_info = STRING
    # plot data
    plot_kinetics_N2O4(np.array(times),xis,xieq,T0,P0,V0,yA0,scenario)
    # update string
    with output_widget:
         output_widget.clear_output()
         print(last_info)
# ============================================= #

