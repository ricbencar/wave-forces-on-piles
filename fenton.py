# ==============================================================================
#  ENGINEERING TECHNICAL REFERENCE & THEORETICAL FORMULATION
# ==============================================================================
#  PROGRAM:      Nonlinear Wave Hydrodynamics Solver (Fenton's Stream Function)
#  METHOD:       Fourier Approximation Method for Steady Water Waves (N=50)
#  REFERENCE:    Fenton, J.D. (1988). "The numerical solution of steady water 
#                wave problems." Computers & Geosciences, 14(3), 357-368.
# ==============================================================================
#
#  1. INTRODUCTION & SCOPE
#  -----------------------------------------------------------------------------
#  This software calculates the hydrodynamics of steady, periodic surface gravity 
#  waves using high-order Stream Function theory. Unlike Linear (Airy) Theory, 
#  which assumes infinitesimal amplitudes, this method retains full nonlinearity 
#  in the boundary conditions.
#
#  Implementation Specifics:
#  - Solver: Scipy Levenberg-Marquardt (Damped Least Squares).
#  - Stability: Uses a Homotopy (continuation) method, stepping wave height 
#    incrementally from linear to target height to guarantee convergence.
#  - Regime: Applicable to stable waves in shallow, intermediate, and deep 
#    water regimes up to the Miche breaking limit.
#
#  2. GOVERNING FIELD EQUATIONS
#  -----------------------------------------------------------------------------
#  The fluid is modeled as inviscid, incompressible, and irrotational.
#  The flow is solved in a frame of reference moving with the wave celerity (c),
#  rendering the flow steady.
#
#  A. Field Equation (Laplace):
#     $\nabla^2 \psi = \frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial z^2} = 0$
#     Where $\psi(x,z)$ is the stream function. Velocities are defined as:
#     $u = \partial\psi/\partial z$ (Horizontal)
#     $w = -\partial\psi/\partial x$ (Vertical)
#
#  B. Bottom Boundary Condition (BBC) at z=0:
#     The seabed is impermeable (a streamline).
#     $\psi(x, 0) = -Q$
#     Where $Q$ is the volume flux per unit width in the moving frame.
#
#  3. FREE SURFACE BOUNDARY CONDITIONS
#  -----------------------------------------------------------------------------
#  The solution is constrained by two nonlinear conditions at the unknown 
#  free surface elevation $z = \eta(x)$:
#
#  A. Kinematic Boundary Condition (KBC):
#     The free surface is a streamline (constant $\psi$).
#     $\psi(x, \eta) = 0$
#
#  B. Dynamic Boundary Condition (DBC - Bernoulli):
#     Pressure is constant (atmospheric) along the surface.
#     $\frac{1}{2} \left[ \left(\frac{\partial \psi}{\partial x}\right)^2 + 
#     \left(\frac{\partial \psi}{\partial z}\right)^2 \right] + g\eta = R$
#     Where $R$ is the Bernoulli constant (Total Energy Head).
#
#  4. NUMERICAL SOLUTION (FOURIER ANSATZ)
#  -----------------------------------------------------------------------------
#  The stream function is approximated by a truncated Fourier series of order N 
#  (N=50) that analytically satisfies the Field Equation and Bottom BC:
#
#  $\psi(x,z) = -(\bar{u} + c) z + \sum_{j=1}^{N} B_j \frac{\sinh(jkz)}{\cosh(jkd)} \cos(jkx)$
#
#  Deep Water Numerical Stability:
#  To prevent floating-point overflow when $kd \gg 1$, the code replaces the 
#  hyperbolic ratio with asymptotic exponentials when arguments > 25.0:
#  $\frac{\sinh(jkz)}{\cosh(jkd)} \approx \exp(jk(z-d))$
#
#  Optimization Vector (State Space):
#  The solver minimizes residuals for the vector $X = [k, \eta_0...\eta_N, B_1...B_N, Q, R]$.
#
#  5. DERIVED PHYSICAL PARAMETERS & OUTPUT DEFINITIONS
#  -----------------------------------------------------------------------------
#  Upon convergence, the software calculates the following engineering parameters
#  derived from the solved Fourier coefficients (B_j).
#
#  A. FUNDAMENTAL WAVE GEOMETRY & PHASE
#  ------------------------------------
#  1. Wavelength (L):
#     Horizontal distance between crests. Solved via dispersion relation.
#     $L = c \cdot T = 2\pi / k$
#
#  2. Celerity (c):
#     Phase velocity. $c = L / T$.
#
#  B. KINEMATICS (VELOCITIES & ACCELERATIONS)
#  ------------------------------------------
#  1. Horizontal Velocity ($u$):
#     $u(x,z) = c - \overline{u} + \sum_{j=1}^N jkB_j \frac{\cosh(jkz)}{\cosh(jkd)} \cos(jkx)$
#
#  2. Vertical Velocity ($w$):
#     $w(x,z) = \sum_{j=1}^N jkB_j \frac{\sinh(jkz)}{\cosh(jkd)} \sin(jkx)$
#
#  3. Max Acceleration ($a_x$):
#     Total derivative (Convective acceleration).
#     $a_x = \frac{Du}{Dt} = u \frac{\partial u}{\partial x} + w \frac{\partial u}{\partial z}$
#
#  4. Velocity Asymmetry:
#     $Asymmetry = |u_{crest}| / |u_{trough}|$
#
#  C. DYNAMICS (INTEGRAL PROPERTIES)
#  ---------------------------------
#  Computed using exact integral invariants (Fenton Eqs 14-16).
#
#  1. Impulse (I):
#     Total wave momentum ($kg \cdot m/s$).
#     $I = \rho(c d - Q)$
#
#  2. Energy Density (E):
#     Mean Energy ($J/m^2$).
#     $PE = \frac{1}{2}\rho g \overline{\eta^2}$
#     $KE = \frac{1}{2}(cI - Q\rho U_c)$
#     $E = PE + KE$
#
#  3. Power / Energy Flux (P):
#     Rate of energy transfer ($W/m$).
#     $P = c(3KE - 2PE) + \frac{1}{2} \overline{u_b^2}(I + \rho c d) + \frac{1}{2}\rho Q U_c^2$
#
#     *Note on $\overline{u_b^2}$ (Mean Square Bed Velocity):*
#     To avoid deep-water integration errors, this is computed algebraically:
#     $\overline{u_b^2} = 2(R - gd) - c^2$
#
#  4. Radiation Stress (Sxx):
#     Excess momentum flux ($N/m$).
#     $S_{xx} = 4(KE) - 3(PE) + \rho \overline{u_b^2} d + 2\rho I U_c$
#
#  5. Mean Stokes Drift ($U_{drift}$):
#     $U_{drift} = I / (\rho d)$
#
#  D. STABILITY & REGIME CLASSIFICATION
#  ------------------------------------
#  1. Ursell Number ($U_r$):
#     $U_r = H L^2 / d^3$ (Values > 26 indicate significant nonlinearity).
#
#  2. Miche Limit ($H_{max}$):
#     Theoretical max height before breaking.
#     $H_{max} = 0.142 L \tanh(kd)$
#
#  3. Saturation (Breaking Index):
#     $Saturation = H / H_{max}$
#     - If $> 1.0$: Wave is BREAKING.
#     - If $< 1.0$: Wave is STABLE.
#
#  4. Regime:
#     - Shallow: $d/L < 0.05$
#     - Intermediate: $0.05 < d/L < 0.5$
#     - Deep: $d/L > 0.5$
#
# ==============================================================================
#  BIBLIOGRAPHY
# ==============================================================================
#
#  1.  Fenton, J.D. (1999). "Numerical methods for nonlinear waves." 
#      In P.L.-F. Liu (Ed.), Advances in Coastal and Ocean Engineering (Vol. 5, 
#      pp. 241–324). World Scientific: Singapore.
#      [Primary Source: Comprehensive review of fully-nonlinear methods including 
#      Fourier approximation, Boundary Integral Equation (BIE) methods, and 
#      Local Polynomial Approximation].
#      URL: https://johndfenton.com/Papers/Fenton99Liu-Numerical-methods-for-nonlinear-waves.pdf
#
#  2.  Fenton, J.D. (1988). "The numerical solution of steady water wave problems."
#      Computers & Geosciences, 14(3), 357–368.
#      [The core algorithm for high-accuracy Stream Function Theory].
#      URL: https://doi.org/10.1016/0098-3004(88)90066-0
#
#  3.  Fenton, J.D. (1985). "A fifth-order Stokes theory for steady waves."
#      Journal of Waterway, Port, Coastal, and Ocean Engineering, 111(2), 216–234.
#      [Standard analytical theory for deep/intermediate water pile design].
#      URL: https://doi.org/10.1061/(ASCE)0733-950X(1985)111:2(216)
#
#  4.  Fenton, J.D. (1978). "Wave forces on vertical bodies of revolution."
#      Journal of Fluid Mechanics, 85(2), 241–255.
#      [Foundational diffraction theory for large diameter piles].
#      URL: https://johndfenton.com/Papers/Fenton78-Waves-on-bodies-of-revolution.pdf
#
#  5.  Fenton, J.D. (1990). "Nonlinear wave theories." In B. Le Méhauté & 
#      D.M. Hanes (Eds.), The Sea: Ocean Engineering Science (Vol. 9, Part A).
#      John Wiley & Sons.
#      [Comprehensive guide for selecting wave theories: Stokes vs Cnoidal vs Stream].
#      URL: https://www.johndfenton.com/Papers/Fenton90b-Nonlinear-wave-theories.pdf
# ==============================================================================

import numpy as np
from scipy.optimize import least_squares
import warnings
import sys

# ==============================================================================
#  GLOBAL CONSTANTS & CONFIGURATION
# ==============================================================================

# Physical Constants (Matched exactly to C++ Phys namespace)
G_STD = 9.8066          # Standard Gravity [m/s^2]
RHO   = 1025.0          # Density of Seawater [kg/m^3]

# Numerical Configuration
DTYPE = np.float64      # Precision for floating point arithmetic
N_FOURIER = 50          # Order of Fourier Series (N). 50 ensures convergence 
                        # even for highly nonlinear near-breaking waves.
N_NUMBERS = 8           # formatting precision for output text

# Suppress optimization warnings (e.g., initial Jacobian singular matrix)
# that occur normally during the first iterations of the solver.
warnings.filterwarnings('ignore')


# ==============================================================================
#  CORE SOLVER CLASS
# ==============================================================================

class FentonStreamFunction:
    """
    Implements Fenton's (1988) Fourier Approximation method for steady water waves.
    Calculates kinematics, dynamics, and integral properties (Energy, Power, Flux).
    """

    def __init__(self, H, T, d, Uc=0.0):
        """
        Initialize the Wave Problem.

        Parameters:
        -----------
        H  : float : Target Wave Height [m]
        T  : float : Wave Period [s]
        d  : float : Water Depth [m]
        Uc : float : Ambient Current Velocity [m/s] (Eulerian / Lab frame)
        """
        # --- Input Parameters ---
        self.H_target = float(H)    # The wave height we want to solve for
        self.T_target = float(T)    # The wave period
        self.d        = float(d)    # The mean water depth
        self.Uc       = float(Uc)   # Current velocity (Doppler shift)
        self.g        = G_STD       # Gravity
        self.N        = N_FOURIER   # Fourier Order
        
        # Calculation mode: Strictly Eulerian (Mean Current in Fixed Frame)
        self.current_type = 'Eulerian' 
        
        # --- Solution State Variables ---
        self.k = 0.0                # Wavenumber (2*pi/L)
        self.L = 0.0                # Wavelength
        self.c = 0.0                # Celerity (Phase Speed)
        self.converged = False      # Solver status flag
        
        # Fourier Coefficients (B_j) for j=1 to N
        self.Bj = np.zeros(self.N, dtype=DTYPE)
        
        # Free surface elevation at discrete nodes
        self.eta_nodes = np.zeros(self.N+1, dtype=DTYPE)
        
        # --- Derived Hydrodynamic Properties ---
        self.eta_crest = 0.0        # Max elevation above Still Water Level (SWL)
        self.eta_trough = 0.0       # Min elevation below SWL
        self.steepness = 0.0        # H/L
        self.rel_depth = 0.0        # d/L
        self.ursell = 0.0           # Ursell Number (Nonlinearity parameter)
        self.regime = ""            # Shallow / Intermediate / Deep
        
        # --- Breaking & Stability ---
        self.breaking_index = 0.0   # Ratio of H to H_max
        self.is_breaking = False    # True if Miche criterion is exceeded
        self.breaking_limit_miche = 0.0
        
        # --- Kinematics ---
        self.u_bed = 0.0            # Max horizontal velocity at seabed
        self.tau_bed = 0.0          # Bed shear stress estimate
        self.acc_max = 0.0          # Maximum fluid acceleration
        self.u_surf = 0.0           # Maximum horizontal velocity at crest
        self.w_max = 0.0            # Maximum vertical velocity
        self.asymmetry = 0.0        # Velocity asymmetry (Crest/Trough ratio)
        self.ExcursionBed = 0.0     # Horizontal particle excursion at bed
        
        # --- Integral Properties ---
        self.Cg = 0.0               # Group Velocity
        self.Power = 0.0            # Energy Flux [W/m]
        self.EnergyDensity = 0.0    # Mean Energy [J/m^2]
        self.Sxx = 0.0              # Radiation Stress [N/m]
        self.Impulse = 0.0          # Wave Momentum [kg*m/s]
        self.MassTransport = 0.0    # Stokes Drift [m/s]
        self.BernoulliR = 0.0       # The Bernoulli Constant (Energy Head)

    # --------------------------------------------------------------------------
    #  STATE MANAGEMENT HELPER FUNCTIONS
    # --------------------------------------------------------------------------

    def _pack_state(self, k, eta, B, Q, R):
        """
        Flattens physical variables into a 1D array for the Least-Squares solver.
        Vector X = [k, eta_0...eta_N, B_1...B_N, Q, R]
        """
        return np.concatenate(([k], eta, B, [Q, R])).astype(DTYPE)

    def _unpack_state(self, x):
        """
        Unpacks the 1D optimization vector back into physical variables.
        """
        M = self.N
        k = x[0]                        # Wavenumber
        etas = x[1 : M+2]               # Surface elevations at nodes
        Bs = x[M+2 : 2*M+2]             # Fourier Coefficients
        Q = x[-2]                       # Mean volume flux (Moving frame)
        R = x[-1]                       # Bernoulli constant
        return k, etas, Bs, Q, R

    # --------------------------------------------------------------------------
    #  MATHEMATICAL BASIS FUNCTIONS
    # --------------------------------------------------------------------------

    def _basis_functions(self, k, z_vals_from_bed):
        """
        Calculates the hyperbolic terms for the Stream Function expansion.
        
        psi_j ~ B_j * sinh(jkz) / cosh(jkd)
        
        Numerical Stability Note:
        For deep water or high frequencies, j*k*d becomes large, causing
        sinh/cosh to overflow. We replace the ratio with exp(-jk(d-z)) 
        when the argument exceeds 25.0.
        """
        kd = k * self.d
        z_vals = np.atleast_1d(z_vals_from_bed).astype(DTYPE)
        
        # Pre-allocate matrices for Sine (S) and Cosine (C) hyperbolic terms
        S = np.zeros((self.N, len(z_vals)), dtype=DTYPE)
        C = np.zeros((self.N, len(z_vals)), dtype=DTYPE)
        
        for j in range(1, self.N + 1):
            idx = j - 1
            arg_check = j * kd
            
            # -- Numerical Overflow Guard --
            if arg_check > 25.0:
                # Use asymptotic approximation: sinh(x)/cosh(Y) -> exp(x-Y)
                # This handles the deep water limit gracefully.
                exp_term = np.exp(j * k * (z_vals - self.d))
                S[idx, :] = exp_term
                C[idx, :] = exp_term
            else:
                # Standard definition
                arg = j * k * z_vals
                denom = np.cosh(j * kd)
                S[idx, :] = np.sinh(arg) / denom 
                C[idx, :] = np.cosh(arg) / denom
        
        return S, C

    # --------------------------------------------------------------------------
    #  OPTIMIZATION OBJECTIVE FUNCTION (RESIDUALS)
    # --------------------------------------------------------------------------

    def _residuals(self, x, H_curr):
        """
        Calculates the error in the Boundary Conditions for the current guess 'x'.
        The solver tries to drive these residuals to zero.
        
        Equations Solved:
        1. Current Def:  c - Q/d = Uc (Mean Eulerian Velocity)
        2. Height Def:   eta[0] - eta[N] = H (Wave height constraint)
        3. Mean Level:   Mean(eta) = d (Conservation of mass/depth)
        4. Kinematic BC: Surface is a streamline (psi = constant)
        5. Dynamic BC:   Pressure is constant (Bernoulli)
        """
        # Unpack current optimization guess
        k, etas, Bs, Q, R = self._unpack_state(x)
        
        # Prevent non-physical wavenumber (singularity guard)
        if k <= 1e-8: k = 1e-8 
        
        # Calculate Celerity from Dispersion relation implied by Period T
        c = (2 * np.pi) / (k * self.T_target)
        
        # Determine Frame Velocity (U_frame)
        # For Eulerian: Frame velocity = Wave Speed - Current
        U_frame = c - self.Uc
            
        # Get Hyperbolic Basis terms at the surface nodes
        S_mat, C_mat = self._basis_functions(k, etas) 
        
        # Setup Grid for Fourier Summation
        x_nds = np.linspace(0, np.pi/k, self.N+1) # Horizontal nodes (half wave)
        phases = k * x_nds
        js = np.arange(1, self.N+1)
        
        # Tensor broadcasting for summation
        cos_t = np.cos(np.outer(js, phases))
        
        # Scaling factor for non-dimensionalization stability
        sc = np.sqrt(self.g / k**3)
        
        # -- Sum Fourier Series Terms --
        # Stream function perturbation
        psi_pert = np.sum(Bs[:,None] * S_mat * cos_t, axis=0) * sc
        
        # Horizontal velocity perturbation (u = d_psi/dz)
        u_pert = np.sum(Bs[:,None] * (js[:,None]*k) * C_mat * cos_t, axis=0) * sc
        
        # Vertical velocity perturbation (v = -d_psi/dx)
        v_pert = np.sum(Bs[:,None] * (js[:,None]*k) * S_mat * (-np.sin(np.outer(js, phases))), axis=0) * sc
        
        # -- Calculate Residuals (Errors) --
        
        # 1. Kinematic BC: psi(x, eta) - Q = 0? (Normalized by sqrt(gd)*d)
        res_kin = (-U_frame * etas + psi_pert + Q) / (np.sqrt(self.g * self.d) * self.d)
        
        # 2. Dynamic BC (Bernoulli): 0.5(u^2+w^2) + g*eta = R?
        u_tot = U_frame - u_pert  # Total horizontal velocity in moving frame
        bern = 0.5 * (u_tot**2 + v_pert**2) + self.g * etas
        res_dyn = (bern - R) / (self.g * self.d)
        
        # 3. Wave Height Constraint: Crest - Trough = H_curr?
        res_h = (etas[0] - etas[-1] - H_curr) / self.d
        
        # 4. Mean Water Level Constraint: Average eta = d?
        mean_eta = (np.sum(etas) - 0.5*etas[0] - 0.5*etas[-1]) / self.N
        res_lvl = (mean_eta - self.d) / self.d
        
        # 5. Current Definition Constraint
        # For Eulerian, c - Q/d is effectively handled by setting U_frame, 
        # so this residual is effectively 0 by definition.
        res_cur = 0.0
             
        # Concatenate all errors into a single vector
        return np.concatenate(([res_cur, res_h, res_lvl], res_kin, res_dyn))

    # --------------------------------------------------------------------------
    #  KINEMATICS CALCULATOR
    # --------------------------------------------------------------------------

    def get_kinematics(self, z_bed, phase=0.0):
        """
        Calculates fluid velocities and acceleration at a specific (z, phase).
        
        Returns:
        u_abs : Horizontal velocity in fixed frame
        w_abs : Vertical velocity
        ax    : Horizontal acceleration
        """
        # Get basis functions at height z
        S, C = self._basis_functions(self.k, [z_bed])
        S = S.flatten(); C = C.flatten()
        js = np.arange(1, self.N+1)
        sc = np.sqrt(self.g / self.k**3)
        
        # Calculate Perturbation Velocities
        term_u = self.Bj * (js * self.k) * C * np.cos(js * phase)
        u_pert = np.sum(term_u) * sc
        
        term_w = self.Bj * (js * self.k) * S * np.sin(js * phase)
        w_pert = np.sum(term_w) * sc
        
        # Determine Frame Velocity (Eulerian)
        U_frame = self.c - self.Uc
            
        # Transform to Stationary Frame
        # u_fix = (c - U_frame) + u_pert = (c - (c-Uc)) + u_pert = Uc + u_pert
        u_abs = (self.c - U_frame) + u_pert 
        w_abs = w_pert
        
        # Calculate Acceleration (Convective terms)
        # ax = u * du/dx + w * du/dz (steady flow approx)
        du_dx = np.sum(self.Bj * (js*self.k)**2 * C * (-np.sin(js*phase))) * sc
        du_dz = np.sum(self.Bj * (js*self.k)**2 * S * np.cos(js*phase)) * sc
        u_moving = u_abs - self.c  # Velocity relative to wave form
        
        ax = u_moving * du_dx + w_pert * du_dz
        
        return u_abs, w_abs, ax

    # --------------------------------------------------------------------------
    #  INTEGRAL PROPERTIES CALCULATOR
    # --------------------------------------------------------------------------

    def _calc_integral_props(self):
        """
        Computes derived engineering properties using exact integral invariants
        (Fenton Eqs 14-16) rather than linear approximations.
        Includes Energy, Power, Radiation Stress, and Mass Transport.
        """
        # Calculate mean potential energy term via numerical integration (DCT)
        Ej = np.zeros(self.N, dtype=DTYPE)
        for j in range(1, self.N+1):
            sum_cos = 0.0
            # Trapezoidal integration over the wave profile
            for m in range(self.N+1):
                val = (self.eta_nodes[m] - self.d) * np.cos(j * m * np.pi / self.N)
                weight = 0.5 if (m==0 or m==self.N) else 1.0
                sum_cos += val * weight
            Ej[j-1] = (2.0/self.N) * sum_cos

        # Mean current velocity (u1) - Eulerian definition
        u1 = self.Uc

        # Impulse (I) - Momentum per unit surface area
        I = RHO * (self.c * self.d - self.Q) 
        
        # Potential Energy (PE)
        PE = 0.25 * RHO * self.g * np.sum(Ej**2)
        
        # Kinetic Energy (KE)
        KE = 0.5 * (self.c * I - u1 * self.Q * RHO)
        
        # Total Energy
        E_total = PE + KE
        
        self.Impulse = I
        self.EnergyDensity = E_total 
        self.MassTransport = I / (RHO * self.d) # Stokes Drift

        # Algebraic Mean Square Bed Velocity (ub2_alg)
        # This exact identity (Eq 14 in Fenton) must be used for Sxx and Power 
        # to match high-precision C++ output, regardless of depth.
        ub2_alg = 2.0 * (self.R - self.g * self.d) - self.c**2
            
        # Energy Flux (Power) P [Watts/m]
        # Uses ub2_alg for consistency with Fenton's derivation
        term1 = self.c * (3.0 * KE - 2.0 * PE)
        term2 = 0.5 * ub2_alg * (I + RHO * self.c * self.d)
        term3 = self.c * u1 * RHO * self.Q
        self.Power = term1 + term2 + term3
        
        # Radiation Stress Sxx [Newtons/m]
        # Uses ub2_alg for consistency
        self.Sxx = 4.0*KE - 3.0*PE + ub2_alg*(RHO*self.d) + 2.0*RHO*u1*self.Q
        
        # Group Velocity (Cg)
        if E_total > 1e-6: 
            self.Cg = self.Power / E_total
        else: 
            self.Cg = 0.0

    # --------------------------------------------------------------------------
    #  MAIN SOLVER ROUTINE
    # --------------------------------------------------------------------------

    def solve(self):
        """
        Executes the solution strategy:
        1. Estimates initial guess using Linear (Airy) Theory.
        2. Steps up the Wave Height gradually to ensure solver stability.
        3. Performs final optimization at target Height.
        4. Post-processes results to get engineering parameters.
        """
        # --- 1. Initial Guess (Linear Theory) ---
        L0 = (self.g * self.T_target**2) / (2 * np.pi)
        
        # Explicit approximation for wavenumber k
        if (self.d / L0) < 0.05: 
            k0 = 2*np.pi / (self.T_target * np.sqrt(self.g * self.d))
        else: 
            k0 = 2*np.pi / L0
            
        # Iterative refinement for k0 considering current (Doppler)
        u_doppler = self.Uc 
        for _ in range(20):
            sig = 2*np.pi/self.T_target - k0*u_doppler
            if sig <= 0: sig = 1e-5 # Safety against strong counter-currents
            k0 = 0.5*k0 + 0.5*(sig**2 / (self.g * np.tanh(k0 * self.d)))
            
        # Generate initial surface profile (small cosine wave)
        x_nds = np.linspace(0, np.pi/k0, self.N+1)
        eta_i = self.d + (0.01/2)*np.cos(k0*x_nds)
        
        # Initial Flux Q and Bernoulli R
        B_i = np.zeros(self.N)
        Q_i = (2*np.pi/k0/self.T_target - self.Uc)*self.d
        R_i = 0.5*(Q_i/self.d)**2 + self.g*self.d
        
        # Pack into vector
        x_curr = self._pack_state(k0, eta_i, B_i, Q_i, R_i)
        
        # --- 2. Height Stepping (Homotopy Method) ---
        # We solve for small waves first, using the result as the seed for larger waves.
        # This prevents the solver from diverging on highly nonlinear waves.
        steps = np.linspace(0.01, self.H_target, 4)
        
        for i, h in enumerate(steps):
            # Use 'trf' (Trust Region Reflective) for robustness early on,
            # 'lm' (Levenberg-Marquardt) for speed later.
            method = 'lm' if i > 2 else 'trf'
            try:
                res = least_squares(self._residuals, x_curr, args=(h,), 
                                    method=method, max_nfev=2000)
                x_curr = res.x
            except: 
                pass # Continue to next step even if strict convergence failed

        # --- 3. Final High-Precision Solution ---
        try:
            res_final = least_squares(self._residuals, x_curr, args=(self.H_target,), 
                                      method='lm', ftol=1e-14)
            x_curr = res_final.x
        except: 
            pass

        # Unpack Final State
        self.k, self.eta_nodes, self.Bj, self.Q, self.R = self._unpack_state(x_curr)
        self.L = 2 * np.pi / self.k
        self.c = self.L / self.T_target
        self.converged = True
        self.BernoulliR = self.R
        
        # --- 4. Post-Processing & Derived Physics ---
        self.eta_crest = self.eta_nodes[0] - self.d
        self.eta_trough = self.eta_nodes[-1] - self.d
        
        # Dimensionless Parameters
        self.ursell = (self.H_target * self.L**2) / (self.d**3)
        self.steepness = self.H_target / self.L
        self.rel_depth = self.d / self.L
        
        # Breaking Criteria (Miche Limit)
        self.breaking_limit_miche = 0.142 * self.L * np.tanh(self.k * self.d)
        self.breaking_index = self.H_target / self.breaking_limit_miche
        self.is_breaking = self.H_target > self.breaking_limit_miche
        
        # Regime Classification
        if self.rel_depth < 0.05: self.regime = "Shallow"
        elif self.rel_depth < 0.5: self.regime = "Intermediate"
        else: self.regime = "Deep"
        
        # Compute Integrals (Energy, Power, etc.)
        self._calc_integral_props()
        
        # Calculate Peak Velocities
        self.u_bed, _, _ = self.get_kinematics(0.0, 0.0)
        
        # Bed Shear Stress (Quadratic drag approximation)
        cf_est = 0.005 
        self.tau_bed = 0.5 * RHO * cf_est * (self.u_bed**2)
        
        # Particle Excursion at Bed
        self.ExcursionBed = abs(self.u_bed) * self.T_target / (2 * np.pi)
        
        # Surface Velocities
        self.u_surf, _, _ = self.get_kinematics(self.d + self.eta_crest, 0.0)
        u_trough, _, _ = self.get_kinematics(self.d + self.eta_trough, np.pi)
        
        # Nonlinear Asymmetry Check (should be > 1.0 for nonlinear waves)
        self.asymmetry = abs(self.u_surf / u_trough)
        
        # Scan phase 0 to pi to find max acceleration and vertical velocity
        scan_phases = np.linspace(0, np.pi, 20)
        max_ax = 0.0; max_w = 0.0
        for ph in scan_phases:
            # Approximate z-level at this phase
            idx = int(ph / np.pi * self.N)
            if idx >= self.N: idx = self.N
            z_exact = self.eta_nodes[idx]
            
            _, w, ax = self.get_kinematics(z_exact, ph)
            if abs(ax) > max_ax: max_ax = abs(ax)
            if abs(w) > max_w: max_w = abs(w)
            
        self.acc_max = max_ax
        self.w_max = max_w


# ==============================================================================
#  I/O & UTILITY CLASSES
# ==============================================================================

def get_input(prompt_text):
    """Safely retrieves positive floating point input from CLI."""
    while True:
        try:
            val_str = input(prompt_text)
            if not val_str: return None
            val = float(val_str)
            if val < 0:
                print("  ! Value must be positive.")
                continue
            return val
        except ValueError:
            print("  ! Invalid input. Please enter a number.")

class Tee:
    """
    Simulates the Unix 'tee' command.
    Writes output to both the console (stdout) and a log file simultaneously.
    """
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()


# ==============================================================================
#  MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    # 1. User Interface: Capture Wave Parameters
    print("\n--- FENTON WAVE SOLVER INPUT ---")
    H_in = get_input("Wave Height (H) [m]: ")
    T_in = get_input("Wave Period (T) [s]: ")
    d_in = get_input("Water Depth (d) [m]: ")
    Uc_in = get_input("Current Vel (Uc)[m/s]: ")
    
    # Graceful exit on empty input
    if None in [H_in, T_in, d_in, Uc_in]: return
    
    # 2. Run Simulations
    #    Case A: No Current (Pure wave)
    solver0 = FentonStreamFunction(H_in, T_in, d_in, Uc=0.0)
    solver0.solve()
    
    #    Case B: With Ambient Current
    solverC = FentonStreamFunction(H_in, T_in, d_in, Uc=Uc_in)
    solverC.solve()

    # 3. Initialize Output Logger (File: output.txt)
    logger = Tee("output.txt", "w")
    
    # --- Formatting Helpers ---
    def log(s):
        logger.write(s)

    def log_line(width=78):
        log("+" + "-" * (width - 2) + "+\n")

    # --- Print Header ---
    log_line()
    log(f"|{'NONLINEAR WAVE HYDRODYNAMICS SOLVER (FENTON)':^76}|\n")
    log_line()
    log("| Please enter the wave parameters:\n")
    log(f"|   > Wave Height (H) [m]: {H_in}\n")
    log(f"|   > Wave Period (T) [s]: {T_in}\n")
    log(f"|   > Water Depth (d) [m]: {d_in}\n")
    log(f"|   > Current Vel (Uc)[m/s]: {Uc_in}\n")
    log(f"|{' ':^76}|\n")
    log(f"|{' Solving full nonlinear system...':<76}|\n")
    log_line()

    # --- Print Results Table ---
    log(f"|{'CALCULATED HYDRODYNAMIC PARAMETERS':^76}|\n")
    log_line()
    
    # Column Widths Setup: | 24 | 17 | 17 | 15 | => Total 78 chars
    log(f"| {'PARAMETER':<22} | {'NO CURRENT':^15} | {'WITH CURRENT':^15} | {'UNIT':<13} |\n")
    
    # Separator Line
    sep = f"|{'-'*24}+{'-'*17}+{'-'*17}+{'-'*15}|\n"
    log(sep)

    has_current = (Uc_in != 0.0)
    
    # Helper to mask the 'Current' column if Uc=0 was entered
    def val_c(v): return v if has_current else "-"

    # Helper to print a single data row
    def row(label, val1, val2, unit):
        def fmt(v):
            if isinstance(v, str): return v
            if v is None: return "-"
            try:
                sign_offset = 1 if v < 0 else 0
                int_part = int(abs(v))
                len_int = len(str(int_part))
                decimals = N_NUMBERS - (sign_offset + len_int) - 1
                if decimals < 0: decimals = 0
                return f"{v:.{decimals}f}"
            except: return str(v)

        s1 = fmt(val1)
        s2 = fmt(val2)
        log(f"| {label:<22} | {s1:^15} | {s2:^15} | {unit:<13} |\n")

    # -- Fill Data Rows --
    row("Wavelength (L)", solver0.L, val_c(solverC.L), "m")
    row("Wave Number (k)", solver0.k, val_c(solverC.k), "rad/m")
    row("Ang. Frequency (w)", 2*np.pi/solver0.T_target, val_c(2*np.pi/solverC.T_target), "rad/s")
    row("Celerity (c)", solver0.c, val_c(solverC.c), "m/s")
    row("Crest Elev (eta_c)", solver0.eta_crest, val_c(solverC.eta_crest), "m")
    row("Trough Elev (eta_t)", solver0.eta_trough, val_c(solverC.eta_trough), "m")
    log(sep)

    row("Energy Density (E)", solver0.EnergyDensity/1000, val_c(solverC.EnergyDensity/1000), "kJ/m2")
    row("Power (Flux)", solver0.Power/1000, val_c(solverC.Power/1000), "kW/m")
    row("Group Vel (Cg)", solver0.Cg, val_c(solverC.Cg), "m/s")
    row("Rad Stress (Sxx)", solver0.Sxx/1000, val_c(solverC.Sxx/1000), "kN/m")
    row("Impulse (I)", solver0.Impulse, val_c(solverC.Impulse), "kg m/s")
    row("Mean Stokes Drift", solver0.MassTransport, val_c(solverC.MassTransport), "m/s")
    log(sep)
    
    row("Max Surf Vel (usurf)", solver0.u_surf, val_c(solverC.u_surf), "m/s")
    row("Max Bed Vel (ubed)", solver0.u_bed, val_c(solverC.u_bed), "m/s")
    row("Max Accel (ax)", solver0.acc_max, val_c(solverC.acc_max), "m/s2")
    row("Vel Asymmetry", solver0.asymmetry, val_c(solverC.asymmetry), "-")
    log(sep)

    # Breaking Status Check
    warn0 = "BREAKING!" if solver0.is_breaking else "STABLE"
    warnC = "BREAKING!" if solverC.is_breaking else "STABLE" if has_current else "-"
    
    row("Miche Limit (Hmax)", solver0.breaking_limit_miche, val_c(solverC.breaking_limit_miche), "m")
    row("Saturation (H/Hmax)", solver0.breaking_index, val_c(solverC.breaking_index), "-")
    row("Breaking Status", warn0, warnC, "-")
    row("Ursell Number", solver0.ursell, val_c(solverC.ursell), "-")
    row("Regime", solver0.regime, val_c(solverC.regime), "-")
    log_line()
    log("\n")

    # --- Print Glossary ---
    log_line()
    log(f"|{'PARAMETER DEFINITIONS & GLOSSARY':^76}|\n")
    log_line()
    log(f"| {'VARIABLE':<22} | {'DESCRIPTION':<49} |\n")
    log(f"|{'-'*24}+{'-'*51}|\n")
    
    # Dictionary of terms for the glossary
    terms = [
        ("Wavelength (L)", "Horizontal distance between two wave crests."),
        ("Wave Number (k)", "Spatial frequency of the wave (2*pi / L)."),
        ("Ang. Frequency (w)", "Temporal frequency of the wave (2*pi / T)."),
        ("Celerity (c)", "Phase velocity (Speed of the wave form)."),
        ("Crest Elev (eta_c)", "Crest Level relative to MWL."),
        ("Trough Elev (eta_t)", "Trough Level relative to MWL."),
        ("Energy Density (E)", "Total mean energy (PE+KE) per unit surface area."),
        ("Power (Flux)", "Rate of energy transfer across vertical section."),
        ("Group Vel (Cg)", "Velocity of energy transport (Power/Energy)."),
        ("Rad Stress (Sxx)", "Excess momentum flux in propagation direction."),
        ("Impulse (I)", "Total wave momentum per meter width."),
        ("Mean Stokes Drift", "Net mass transport velocity averaged over depth."),
        ("Max Surf Vel", "Max horizontal fluid velocity at the crest."),
        ("Max Bed Vel", "Max horizontal fluid velocity at the seabed."),
        ("Max Accel (ax)", "Maximum horizontal acceleration (inertial load)."),
        ("Vel Asymmetry", "Ratio of Crest Vel to Trough Vel (>1 nonlinear)."),
        ("Miche Limit (Hmax)", "Maximum wave height before breaking."),
        ("Saturation", "Ratio of current Height to max stable Height."),
        ("Breaking Status", "Stability flag based on Miche Criterion."),
        ("Ursell Number", "Nonlinearity parameter (H*L^2 / d^3)."),
        ("Regime", "Classification by depth ratio Shallow/Inter/Deep")
    ]
    
    # Iterate and print glossary rows
    for name, desc in terms:
        log(f"| {name:<22} | {desc:<49} |\n")
    
    log_line()
    
    # Close resources
    logger.close()

if __name__ == "__main__":
    main()