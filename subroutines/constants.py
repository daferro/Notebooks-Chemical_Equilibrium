# =====    Import libraries of interest    ===== #
# ---------------------------------------------- #
import numpy                as np                #
# ============================================== #

# ---- Define some constants of interest (SI) ----
m_u    = 1.66053886e-27      # atomic mass constant in kg
m_e    = 9.1093837E-31       # mass of electron
q_e    = 1.60217663E-19      # charge of electron
h      = 6.6260693E-34       # Planck's constant in J*s
k_B    = 1.3806505E-23       # Boltzmann's constant in J/K
c_0    = 2.99792558E8        # Speed of light in m/s
eps0   = 8.85418782E-12      # Vacuum permittivity
NA     = 6.02214076E23       # Avogadro's number
# ------------------------------------------------
P_o    = 1E5                 # standard pressure (P^o = 1E5 Pa = 1 bar)
c_o    = 1E3                 # 1 mol/L = 1E3 mol/m^3
R      = NA * k_B            # gas constant J/(K mol)
hbar   = h/(2*np.pi)
# ----- Universal constants to atomic units  -----
a_0    = (4*np.pi*eps0*hbar**2) / (m_e * q_e**2)
Eh     = q_e**2 / (4*np.pi * eps0 * a_0)
Hz_au  = Eh/hbar
# ------------------------------------------------
ZERO1  = 1e-300  # to avoid problems when n=0 in n*log(n)
ZERO2  = 1E-014  # if ni < ZERO2 --> ni = 0 (for chemical kinetics)
ZERO3  = 1E-010  # comparison of rotational constants (if equal --> linear)
ZERO4  = 1E-007  # for intercept method; if y=0 --> y=ZERO4 (so we can obtain the derivative)
# ------------------------------------------------
last_fig  = None
# ------------------------------------------------
FONTSIZE  = [11,12,14,15,16,20]
# ------------------------------------------------
NPOINTST  = 41    # number of Temperature  points
NPOINTSXI = 251   # number of xi points (thermo & kinetics)
REL_XI_EQ = 0.999 # plot until REL_XI_EQ * xieq
# ============================================== #
