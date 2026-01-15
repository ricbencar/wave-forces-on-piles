#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ==============================================================================
#  HIGH-PRECISION WAVE HYDRODYNAMICS & STRUCTURAL IMPACT SOLVER
# ==============================================================================
#  MODULE:   script.py
#  TYPE:     Nonlinear BVP Solver & Transient Load Calculator
#  METHOD:   Fenton's Fourier Approximation
#  ENGINE:   Python 3 + Numba JIT (High Performance)
#  LICENSE:  MIT / Academic Open Source
# ==============================================================================
#
#  PROGRAM DESCRIPTION:
#  This software calculates the hydrodynamics (kinematics and dynamics) and 
#  structural loading of steady, finite-amplitude water waves acting on a 
#  vertical cylindrical pile.
#
#  It implements the "Fourier Approximation Method" for the Nonlinear Stream 
#  Function as developed by J.D. Fenton (1988). Unlike Linear (Airy) theory 
#  or Stokes 5th Order approximations, this numerical method satisfies the 
#  full nonlinear boundary conditions to machine precision (limited only by the 
#  truncation order N).
#
#  OPTIMIZATION NOTE:
#  This version uses Numba JIT (Just-In-Time) compilation to accelerate the 
#  core mathematical kernels (Basis functions, Kinematics, and Force Integration).
#  The "Hot Loops" that run thousands of times are now compiled to machine code,
#  offering performance comparable to C/Fortran while retaining Python flexibility.
#
#  LIMITATIONS:
#  - Restricted to H/d <= 0.6.
#  - Impulsive slamming loads are not calculated (requires breaking waves).
#
# ==============================================================================
#  THEORETICAL MANUAL & NUMERICAL DOCUMENTATION
# ==============================================================================
#
#  1. INTRODUCTION & PHYSICAL SCOPE
#  -----------------------------------------------------------------------------
#  This solver addresses the Boundary Value Problem (BVP) of nonlinear water 
#  waves propagating over a horizontal bed and their interaction with vertical 
#  cylindrical structures. Unlike Linear Wave Theory (Airy), which assumes 
#  infinitesimal wave amplitude (H << d), or Stokes Expansion methods which 
#  diverge in shallow water, the Fourier Approximation Method (Stream Function 
#  Theory) is a numerical method accurate to machine precision for:
#    a. Waves within the H/d <= 0.6 limit.
#    b. Any water depth (Shallow, Intermediate, Deep).
#    c. Nonlinear wave-current interaction (Doppler shifting).
#
#  The kinematic field calculated here drives the Quasi-Static Loading model:
#    - Morison Equation (Drag + Inertia).
#
#  2. MATHEMATICAL FORMULATION: FENTON'S STREAM FUNCTION THEORY
#  -----------------------------------------------------------------------------
#  The method solves for the Stream Function Psi(x,z) in a 2D domain.
#
#  2.1 GOVERNING FIELD EQUATIONS
#      Assumption: Fluid is inviscid, incompressible, and irrotational.
#      - Incompressibility: div(u) = 0
#      - Irrotationality:   curl(u) = 0
#      Consequently, a Scalar Potential (Phi) and Stream Function (Psi) exist.
#      The governing equation is the Laplace Equation:
#      
#      ∇²Psi = (d²Psi/dx²) + (d²Psi/dz²) = 0
#
#      Velocity relationships: u = dPsi/dz, w = -dPsi/dx.
#
#  2.2 COORDINATE SYSTEM & REFERENCE FRAME
#      We utilize a coordinate system moving with the steady wave celerity (C).
#      - (x, z): Stationary frame coordinates (Seabed at z=0).
#      - (X, z): Moving frame coordinates, where X = x - C*t.
#      
#      In this moving frame, the wave profile is stationary in time, reducing 
#      the problem from unsteady (t-dependent) to steady state.
#
#  2.3 THE FOURIER ANSATZ (Analytical Solution Structure)
#      Fenton (1988) defines a truncated Fourier series of order N that 
#      automatically satisfies the Laplace equation and the Bottom Boundary 
#      Condition (w=0 at z=0).
#
#      Psi(X, z) = -U_bar * z + SUM_{j=1}^{N} [ B_j * sinh(j*k*z)/cosh(j*k*d) * cos(j*k*X) ]
#
#      Where:
#      - U_bar: Mean fluid speed in the moving frame (related to mass flux).
#      - k:     Wavenumber (2*pi/L).
#      - B_j:   Dimensionless Fourier coefficients (The unknowns).
#      - d:     Water depth.
#
#  2.4 BOUNDARY CONDITIONS (The Constraint System)
#      The coefficients B_j must be solved such that they satisfy conditions 
#      at the free surface z = eta(X).
#
#      A. Kinematic Boundary Condition (KBC):
#         The free surface is a streamline. No flow crosses the interface.
#         Psi(X, eta(X)) = -Q
#         Where Q is the constant volume flux per unit width in the moving frame.
#
#      B. Dynamic Boundary Condition (DBC - Bernoulli):
#         Pressure is constant (atmospheric) along the free surface.
#         0.5 * [ (dPsi/dX)^2 + (dPsi/dz)^2 ] + g * eta(X) = R
#         Where R is the Bernoulli constant (Total Energy Head).
#
#  3. NUMERICAL SOLVER ALGORITHM
#  -----------------------------------------------------------------------------
#  The problem is recast as a nonlinear optimization problem.
#
#  3.1 SYSTEM OF EQUATIONS
#      We discretize the wave phase (0 to pi, due to symmetry) into M nodes.
#      The Unknowns Vector x (Size N + 3) contains:
#      - [B_1 ... B_N]: The stream function coefficients.
#      - k:             The Wavenumber.
#      - Q:             The Mass Flux.
#      - R:             The Bernoulli Constant.
#
#  3.2 OPTIMIZATION OBJECTIVE
#      We define a residual vector 'r' combining errors from KBC, DBC, and 
#      geometric definitions (Wave Height H and Mean Water Level d).
#      Minimization target: sum(r^2) -> 0.
#
#      Solver: Scipy 'least_squares' using Trust Region Reflective (TRF) method.
#      Jacobian: Calculated numerically via 2-point finite difference.
#
#  3.3 ADAPTIVE HOMOTOPY (continuation Method)
#      Nonlinear solvers fail if the initial guess is too far from the solution.
#      This script implements a homotopy wrapper:
#      1. Initialize with Linear Wave Theory (Airy) for H ~ 0.
#      2. Step H from 0 -> Target_H in 'n' increments (Linear steps).
#      3. Use the solution of step i as the initial guess for step i+1.
#
#  4. FORCE CALCULATION: THE MORISON EQUATION
#  -----------------------------------------------------------------------------
#  Applied when the structure is hydrodynamically transparent (D/L < 0.2).
#  The total force is the superposition of Drag (viscous) and Inertia (mass).
#
#  dF(z,t) = dF_Drag + dF_Inertia
#
#  4.1 DRAG COMPONENT (Velocity Dependent)
#      dF_D = 0.5 * rho * Cd * D * (u + Uc) * |u + Uc| * dz
#      - Nonlinear dependence on u|u|.
#      - Cd is a function of Reynolds number (Re) and roughness (k/D).
#      - Includes steady current Uc in the velocity vector.
#
#  4.2 INERTIA COMPONENT (Acceleration Dependent)
#      dF_I = rho * Cm * (pi * D^2 / 4) * du/dt * dz
#      - Dominated by fluid acceleration field (du/dt).
#      - Cm = 1 + Ca (Added mass coefficient).
#
#  4.3 PHASE LAG
#      Drag forces peak at the wave crest (max velocity). Inertia forces peak 
#      at the zero-crossing (max acceleration). The total force peak occurs 
#      at a phase angle between 0 and 90 degrees, found via golden-section search.
#
# ==============================================================================
#  DEPENDENCIES & INSTALLATION
# ==============================================================================
#  PREREQUISITES:
#  - Python 3.8+ (Required for f-strings and type hinting support).
#  - A standard Python environment (CPython).
#  - Numba: Required for JIT compilation.
#
#  INSTALLATION COMMAND:
#  $ pip install numpy scipy matplotlib numba
#
#  LIBRARY BREAKDOWN & UTILIZATION:
#  -----------------------------------------------------------------------------
#  1. NumPy (Numerical Python)
#     - Role: Core Linear Algebra & Tensor Engine.
#     - Critical Usage in Solver:
#       * Precision Control: Enforces IEEE 754 Double Precision (float64).
#       * Vectorization: Acceleration of trigonometric basis functions.
#
#  2. SciPy (Scientific Python)
#     - Role: Nonlinear Optimization & Numerical Methods.
#     - Critical Usage: `scipy.optimize.least_squares` (TRF Algorithm).
#
#  3. Matplotlib (Plotting Library)
#     - Role: Technical Visualization & Reporting (PDF/PNG).
#
#  4. Numba (JIT Compiler)
#     - Role: Performance Acceleration.
#     - Usage: Decorates critical math functions (@jit) to compile them 
#       into optimized machine code, significantly speeding up the solver 
#       loops and force integration.
#
# ==============================================================================
#  BIBLIOGRAPHY & REFERENCES
# ==============================================================================
#
#  *** THEORETICAL BASIS (FENTON: STREAM FUNCTION, KINEMATICS & NUMERICAL METHODS) ***
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
#
#  *** HOCINE OUMERACI (BREAKING WAVE IMPACT, SLAMMING & RINGING) ***
#  6.  Wienke, J., & Oumeraci, H. (2005). "Breaking wave impact force on a vertical 
#      and inclined slender pile—theoretical and large-scale model investigations."
#      Coastal Engineering, 52(5), 435–462.
#      [CRITICAL: Separates quasi-static (Morison) from dynamic (slamming) forces].
#      URL: https://doi.org/10.1016/j.coastaleng.2004.12.008
#
#  7.  Irschik, K., Sparboom, U., & Oumeraci, H. (2004). "Breaking wave loads on a 
#      slender pile in shallow water." Proceedings of the 29th ICCE, 4, 3968–3980.
#      [Focuses on shallow water impacts where Stream Function may reach limits].
#      URL: https://www.worldscientific.com/doi/abs/10.1142/9789812701916_0045
#
#  8.  Kortenhaus, A., & Oumeraci, H. (1998). "Classification of wave loading on 
#      monolithic coastal structures." Proceedings of the 26th ICCE, 1, 867–879.
#      [Defines transition zones between pulsating and impulsive load regimes].
#      URL: https://icce-ojs-tamu.tdl.org/icce/article/download/5654/5324/0
#
#  9.  Muttray, M., & Oumeraci, H. (2005). "Theoretical and experimental study on 
#      wave damping inside a perforated caisson." Ocean Engineering, 32(14), 1803–1818.
#      [Relevant for piles with scour protection or permeable outer layers].
#      URL: https://www.sciencedirect.com/science/article/abs/pii/S0378383905000591
#
#  *** ENGINEERING MANUALS & STANDARDS ***
#  10. U.S. Army Corps of Engineers (USACE). (2002). "Coastal Engineering Manual 
#      (CEM)." Engineer Manual 1110-2-1100. Washington, D.C.
#      [The modern successor to the SPM; standard for wave mechanics].
#      URL: https://www.publications.usace.army.mil/USACE-Publications/Engineer-Manuals/u43544q/636F617374616C20656E67696E656572696E67206D616E75616C/
#
#  11. U.S. Army Corps of Engineers (USACE). (1984). "Shore Protection Manual 
#      (SPM)." Vol. I & II. 4th Edition. CERC, Vicksburg, MS.
#      [Classic reference; still widely used for historical comparison and empirical data].
#      URL: https://usace.contentdm.oclc.org/digital/collection/p16021coll11/id/1934/
#
#  12. CIRIA, CUR, CETMEF. (2007). "The Rock Manual. The Use of Rock in 
#      Hydraulic Engineering." (2nd Edition). C683, CIRIA, London.
#      [Standard for pile scour protection design and rock interaction].
#      URL: https://www.ciria.org/ItemDetail?iProductCode=C683
#
#  13. DNV (Det Norske Veritas). (2014). "Environmental Conditions and Environmental 
#      Loads." Recommended Practice DNV-RP-C205.
#      [Industry standard for offshore pile design and Morison coefficients].
#      URL: https://www.dnv.com/energy/standards-guidelines/dnv-rp-c205-environmental-conditions-and-environmental-loads/
#
#  *** TEXTBOOKS (WAVE MECHANICS & FORCES) ***
#  14. Sumer, B. M., & Fredsøe, J. (2006). "Hydrodynamics Around Cylindrical 
#      Structures." (Revised Edition). World Scientific.
#      [The 'Bible' for flow around piles, vortex shedding, and scour].
#      URL: https://doi.org/10.1142/6248
#
#  15. Sarpkaya, T., & Isaacson, M. (1981). "Mechanics of Wave Forces on 
#      Offshore Structures." Van Nostrand Reinhold.
#      [Classic text on diffraction and inertia/drag regimes].
#      URL: https://www.amazon.com/-/pt/dp/0521896258/
#
#  16. Goda, Y. (2010). "Random Seas and Design of Maritime Structures." 
#      (3rd Edition). World Scientific.
#      [Essential for spectral analysis and statistical design of piles].
#      URL: https://doi.org/10.1142/7425
#
#  17. Dean, R. G., & Dalrymple, R. A. (1991). "Water Wave Mechanics for 
#      Engineers and Scientists." World Scientific.
#      [Foundational pedagogy for linear and nonlinear wave theory].
#      URL: https://doi.org/10.1142/1232
#
# ==============================================================================
# ==============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares, minimize_scalar
import sys
import os

# -- Numba Imports for JIT Acceleration --
# We import 'jit' for compilation, 'float64' for type strictness, and 'prange'
# if parallel loops were needed (kept simple here for stability).
from numba import jit, float64, int64

# ==============================================================================
#  SECTION 1: PHYSICAL CONSTANTS & CONFIGURATION
# ==============================================================================

# -- Default Simulation Parameters (User Editable via Console) --
DEF_WAVE_HEIGHT    = 3.0000      # H (m): Vertical distance from trough to crest.
DEF_WAVE_PERIOD    = 9.0000      # T (s): Time for one full wave cycle.
DEF_DEPTH          = 5.0000      # d (m): Water depth (SWL to seabed).
DEF_CURRENT        = 1.0000      # Uc (m/s): Ambient current (+ in wave direction).
DEF_CURRENT_TYPE   = "Eulerian"  # 'Eulerian' (fixed point) or 'Stokes' (transport).
DEF_PILE_DIAMETER  = 1.5000      # D (m): Structure diameter.
DEF_MARINE_GROWTH  = 0.0500      # t_mg (m): Thickness of bio-fouling.
DEF_SOLVER_ORDER   = 50          # N: Fourier truncation order (Precision control).
DEF_HOMOTOPY_STEPS = 5           # Number of steps to ramp up wave height.

# -- Physical Constants (IEEE 754 Double Precision) --
RHO          = np.float64(1025.0)      # Density of Seawater (kg/m^3).
G_STD        = np.float64(9.8066)      # Standard Gravity (m/s^2).
NU_SEAWATER  = np.float64(1.05e-6)     # Kinematic Viscosity (m^2/s) at 20C.
DTYPE        = np.float64              # Enforce 64-bit precision for matrix algebra.

# -- Plot individual image files --
DEF_SAVE_PNGS = True
DEF_PLOT_CYCLES = 2 # Single parameter to control x-axis length (e.g., 1.0, 2.0, 3.5)

# ==============================================================================
#  SECTION 1.5: NUMBA ACCELERATED KERNELS
#  These functions perform the heavy lifting and are compiled to machine code.
# ==============================================================================

@jit(nopython=True, cache=True)
def _fast_basis(k, d, N, z_vals):
    """
    JIT-compiled calculation of Sinh/Cosh basis matrices.
    Replaces the loop in the original _basis_functions method.
    """
    n_z = len(z_vals)
    S = np.zeros((N, n_z), dtype=np.float64)
    C = np.zeros((N, n_z), dtype=np.float64)
    kd = k * d
    
    # Loop unrolling for basis functions
    for j in range(1, N + 1):
        idx = j - 1
        arg_check = j * kd
        
        # Stability check for large arguments (avoid overflow)
        if arg_check > 20.0:
            # Asymptotic approximation
            for i in range(n_z):
                val = np.exp(j * k * (z_vals[i] - d))
                S[idx, i] = val
                C[idx, i] = val
        else:
            denom = np.cosh(j * kd)
            for i in range(n_z):
                arg = j * k * z_vals[i]
                S[idx, i] = np.sinh(arg) / denom 
                C[idx, i] = np.cosh(arg) / denom
    return S, C

@jit(nopython=True, cache=True)
def _fast_residuals(k, etas, Bs, Q, R, d, g, T_target, Uc, current_type_is_eulerian, N, H_curr):
    """
    JIT-compiled residual calculation for the optimization solver.
    This replaces the original _residuals method, avoiding Python overhead 
    during the 100s of iterations required by least_squares.
    """
    if k <= 1e-8: k = 1e-8
    c = (2 * np.pi) / (k * T_target)

    # Determine frame velocity
    if current_type_is_eulerian:
        U_frame = c - Uc
    else:
        U_frame = Q / d

    # Calculate Basis matrices (calls the fast kernel above)
    S_mat, C_mat = _fast_basis(k, d, N, etas)
    
    # Grid setup
    x_nds = np.linspace(0, np.pi / k, N + 1)
    
    # Pre-allocate residuals array
    # Size: (1 Current) + (1 Wave Height) + (1 Level) + (N+1 Kinematic) + (N+1 Dynamic)
    res_len = 3 + (N+1) + (N+1)
    residuals = np.zeros(res_len, dtype=np.float64)
    
    sc = np.sqrt(g / k**3)
    
    # Main Loop over surface nodes
    for i in range(N + 1):
        phase = k * x_nds[i]
        
        # Summation for stream function (psi), u, and v
        psi_pert = 0.0
        u_pert = 0.0
        v_pert = 0.0
        
        for j in range(1, N + 1):
            idx = j - 1
            cos_t = np.cos(j * phase)
            sin_t = np.sin(j * phase)
            
            term_common = Bs[idx]
            psi_pert += term_common * S_mat[idx, i] * cos_t
            u_pert   += term_common * (j * k) * C_mat[idx, i] * cos_t
            v_pert   += term_common * (j * k) * S_mat[idx, i] * (-sin_t)
            
        psi_pert *= sc
        u_pert   *= sc
        v_pert   *= sc
        
        # A. Kinematic BC Residual: Psi(eta) = -Q
        residuals[3 + i] = (-U_frame * etas[i] + psi_pert + Q) / (np.sqrt(g * d) * d)
        
        # B. Dynamic BC Residual: Bernoulli Constant
        u_tot = U_frame - u_pert
        bern = 0.5 * (u_tot**2 + v_pert**2) + g * etas[i]
        residuals[3 + (N+1) + i] = (bern - R) / (g * d)

    # Geometric Residuals
    residuals[1] = (etas[0] - etas[-1] - H_curr) / d
    
    # Mean Water Level Residual (Trapezoidal Integration)
    sum_eta = 0.0
    for i in range(N + 1):
        w = 0.5 if (i == 0 or i == N) else 1.0
        sum_eta += etas[i] * w
    mean_eta = sum_eta / N
    residuals[2] = (mean_eta - d) / d
    
    # Current Definition Residual
    residuals[0] = 0.0
    if not current_type_is_eulerian:
        residuals[0] = ((c - Q/d) - Uc) / np.sqrt(g * d)
        
    return residuals

@jit(nopython=True, cache=True)
def _fast_kinematics(y, x, k, d, N, Bj, g, c, U_frame, R, rho):
    """
    JIT-compiled point kinematics calculator.
    Returns tuple: (u_fix, v_fix, ax, az, p)
    """
    z_arr = np.array([y], dtype=np.float64)
    # Reuse the fast basis calculation for a single point
    S_mat, C_mat = _fast_basis(k, d, N, z_arr)
    # S_mat is (N, 1) here
    
    sc = np.sqrt(g / k**3)
    
    u_p = 0.0
    v_p = 0.0
    dup_dx = 0.0
    dup_dz = 0.0
    dwp_dx = 0.0
    dwp_dz = 0.0
    
    # Summation Loop for Field Variables and Gradients
    for j in range(1, N + 1):
        idx = j - 1
        jk = j * k
        arg_x = jk * x
        
        cos_kx = np.cos(arg_x)
        sin_kx = np.sin(arg_x)
        
        B_val = Bj[idx]
        
        # Velocities
        u_p += B_val * jk * C_mat[idx, 0] * cos_kx
        v_p += B_val * jk * S_mat[idx, 0] * sin_kx 
        
        # Gradients (Convective Acceleration Terms)
        jk2 = jk * jk
        dup_dx += B_val * jk2 * C_mat[idx, 0] * (-sin_kx)
        dup_dz += B_val * jk2 * S_mat[idx, 0] * cos_kx
        dwp_dx += B_val * jk2 * S_mat[idx, 0] * cos_kx
        dwp_dz += B_val * jk2 * C_mat[idx, 0] * sin_kx

    u_p *= sc
    v_p *= sc
    dup_dx *= sc
    dup_dz *= sc
    dwp_dx *= sc
    dwp_dz *= sc
    
    # Transform to Fixed Frame
    u_fix = (c - U_frame) + u_p
    v_fix = v_p
    
    # Calculate Total Acceleration (Convective)
    # Since flow is steady in moving frame, da/dt = (u-c) * du/dx + w * du/dz
    ax = (u_fix - c) * dup_dx + v_fix * dup_dz
    az = (u_fix - c) * dwp_dx + v_fix * dwp_dz
    
    # Calculate Pressure (Bernoulli)
    u_w = U_frame - u_p
    p = rho * (R - g * y - 0.5 * (u_w**2 + v_p**2))
    
    return u_fix, v_fix, ax, az, p

@jit(nopython=True, cache=True)
def _fast_force_integral(zs, d, k, ph, N, Bj, g, c, U_frame, R, rho, cd, cm, d_eff):
    """
    JIT-compiled integration of Morison forces over the depth vector zs.
    Replaces the loop in scan_force.Significantly speeds up the force scanning phase.
    """
    f_tot_sum = 0.0
    m_tot_sum = 0.0
    fd_tot = 0.0
    fi_tot = 0.0
    
    n_pts = len(zs)
    area = np.pi * d_eff**2 / 4.0
    
    for i in range(n_pts - 1):
        z_curr = zs[i]
        z_next = zs[i+1]
        
        dz = z_next - z_curr
        if dz < 1e-9: continue
        
        z_mid = (z_curr + z_next) * 0.5
        y_bed = z_mid + d
        
        # Calculate kinematics inline (calling the fast kernel)
        u, _, ax, _, _ = _fast_kinematics(y_bed, -ph/k, k, d, N, Bj, g, c, U_frame, R, rho)
        
        # Morison Equation Terms
        fd_local = 0.5 * rho * cd * d_eff * u * np.abs(u)
        fi_local = rho * cm * area * ax
        
        ft_local = fd_local + fi_local
        
        # Trapezoidal/Rectangular Integration
        f_tot_sum += ft_local * dz
        m_tot_sum += ft_local * dz * y_bed
        fd_tot += fd_local * dz
        fi_tot += fi_local * dz
        
    return f_tot_sum, m_tot_sum, fd_tot, fi_tot

# ==============================================================================
#  SECTION 2: I/O UTILITIES & LOGGING
# ==============================================================================

class DualWriter:
    """
    I/O MANAGER CLASS:
    Intercepts the standard output stream to replicate console output to a file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout                 
        self.filename = filename
        self.log = None

    def __enter__(self):
        self.log = open(self.filename, "w", encoding='utf-8') 
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        if self.log:
            self.log.close()

    def write(self, message):
        self.terminal.write(message)               
        if self.log:
            self.log.write(message)                

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()

    def isatty(self):
            return getattr(self.terminal, 'isatty', lambda: False)()
		
def get_input(prompt, default_val):
    try:
        input_prompt = f"{prompt:<45} [{default_val}]: "
        user_input = input(input_prompt).strip()
        if not user_input: return default_val      
        if isinstance(default_val, str): 
            return user_input                      
        return float(user_input)                   
    except Exception: 
        return default_val                         

def print_row(description, value, unit=""):
    if isinstance(value, (float, int, np.floating, np.integer)):
        val_str = f"{value:<15.4f}" 
    else:
        val_str = f"{str(value):<15}" 
    print(f"{description:<41}| {val_str} | {unit}")

def add_param_box(ax, wave, extra_text=""):
    ursell = (wave.H_target * wave.L**2) / (wave.d**3)
    steepness = wave.H_target / wave.L
    
    text_str = (
        f"INPUTS:\n"
        f"H = {wave.H_target:.2f} m\n"
        f"T = {wave.T_target:.2f} s\n"
        f"d = {wave.d:.2f} m\n"
        f"Uc ({wave.current_type}) = {wave.Uc:.2f} m/s\n"
        f"Order = {wave.N}\n\n"
        f"PARAMS:\n"
        f"L = {wave.L:.2f} m\n"
        f"Ur = {ursell:.1f}\n"
        f"H/L = {steepness:.4f}"
    )
    if extra_text:
        text_str += f"\n\nOUTPUTS:\n{extra_text}"
        
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, zorder=10)

# ==============================================================================
#  SECTION 3: CORE SOLVER LOGIC (FENTON STREAM FUNCTION)
# ==============================================================================

class FentonWave:
    """
    PRIMARY HYDRODYNAMIC SOLVER:
    Implements the Fourier Approximation Method for the Nonlinear Stream Function.
    
    [OPTIMIZATION] This class has been refactored to call static Numba-compiled 
    kernels (_fast_residuals, _fast_kinematics) for heavy arithmetic.
    """
    def __init__(self, H, T, d, current, current_type='Eulerian', N_target=DEF_SOLVER_ORDER, n_steps=DEF_HOMOTOPY_STEPS):
        # -- 1. Store Inputs --
        self.H_target = float(H)    
        self.T_target = float(T)    
        self.d = float(d)           
        
        if self.d <= 0: raise ValueError("Water depth (d) must be positive.")
        if self.T_target <= 0: raise ValueError("Wave period (T) must be positive.")
        
        # -- CRITICAL LIMIT CHECK: H/d <= 0.6 --
        if (self.H_target / self.d) > 0.6:
             print(f"\n[!] LIMIT EXCEEDED: H/d = {self.H_target/self.d:.3f} > 0.6. Calculation aborted.")
             self.converged = False
             self.solver_history = [{
                "H": self.H_target, "Err": 0.0, 
                "Msg": "H/d > 0.6 Limit", "Status": "ABORT", "Type": "Limit"
             }]
             self.k = 0.0
             return

        if self.H_target >= self.d: print("WARNING: Wave height exceeds water depth.")

        self.Uc = float(current)    
        self.current_type = current_type 
        self.g = float(G_STD)
        self.N = int(N_target)      
        self.n_steps = int(n_steps) 
        
        # -- 2. Solver Diagnostics --
        self.solver_history = []    
        self.converged = False      
        
        # -- 3. State Vectors --
        self.Bj = np.zeros(self.N, dtype=DTYPE)
        self.Ej = np.zeros(self.N, dtype=DTYPE)
        self.eta_nodes = np.zeros(self.N+1, dtype=DTYPE) 
        
        # -- 4. Physical Properties Container --
        self.prop_KE = 0.0; self.prop_PE = 0.0   
        self.prop_I = 0.0; self.prop_Sxx = 0.0   
        self.prop_F = 0.0; self.prop_ub2 = 0.0   
        self.prop_q_vol = 0.0; self.prop_S = 0.0 
        self.prop_R = 0.0; self.prop_r = 0.0     
        self.prop_u1 = 0.0; self.prop_u2 = 0.0   
        self.prop_U_frame = 0.0                  
        
        # -- 5. Wave Parameters --
        self.k = 0.0 
        self.L = 0.0 
        self.c = 0.0 
        self.Q = 0.0 
        self.R = 0.0 
        
        # -- 6. Execution --
        try:
            self._solve_adaptive()                
            self._calculate_integral_properties() 
        except Exception as e:
            self.converged = False
            self.solver_history.append({
                "H": self.H_target, "Err": 999.9, 
                "Msg": f"Exception: {e}", "Status": "ERR", "Type": "Fail"
            })

    def _pack_state(self, k, eta, B, Q, R):
        return np.concatenate(([k], eta, B, [Q, R])).astype(DTYPE)

    def _unpack_state(self, x):
        M = self.N
        k = x[0]                    
        etas = x[1 : M+2]           
        Bs = x[M+2 : 2*M+2]         
        Q = x[-2]                   
        R = x[-1]                   
        return k, etas, Bs, Q, R

    def _residuals(self, x, H_curr):
        """
        Wrapper to call the JIT-compiled residual function.
        Handles the interface between Scipy (Python) and Numba (C-like speed).
        """
        k, etas, Bs, Q, R = self._unpack_state(x)
        is_eulerian = (self.current_type == 'Eulerian')
        
        return _fast_residuals(
            k, etas, Bs, Q, R, 
            self.d, self.g, self.T_target, self.Uc, 
            is_eulerian, self.N, H_curr
        )

    def _solve_adaptive(self):
        """
        Homotopy Solver Strategy.
        Uses standard linear stepping since H/d is restricted to <= 0.6.
        """
        # A. Initialization: Linear Wave Theory (Airy)
        L0 = (self.g * self.T_target**2) / (2 * np.pi)
        
        if (self.d / L0) < 0.05:
            k0 = 2*np.pi / (self.T_target * np.sqrt(self.g * self.d))
        else:
            k0 = 2*np.pi / L0

        u_doppler = self.Uc if self.current_type == 'Eulerian' else 0.0
        
        for _ in range(20):
            sig = 2*np.pi/self.T_target - k0*u_doppler
            k0 = 0.5*k0 + 0.5*(sig**2/(self.g*np.tanh(k0*self.d)))
            
        x_nds = np.linspace(0, np.pi/k0, self.N+1)
        eta_i = self.d + (0.01/2)*np.cos(k0*x_nds)
        B_i = np.zeros(self.N); Q_i = (2*np.pi/k0/self.T_target - self.Uc)*self.d
        R_i = 0.5*(Q_i/self.d)**2 + self.g*self.d
        
        x_curr = self._pack_state(k0, eta_i, B_i, Q_i, R_i)
        
        # B. Stepping Configuration - Simple Linear Steps
        default_steps = globals().get('DEF_HOMOTOPY_STEPS', 5)
        user_steps = getattr(self, 'n_steps', default_steps)
        n_steps = max(3, user_steps)
        
        h_start_log = 0.01
        
        # Simple linear distribution for safe regime
        steps = np.linspace(h_start_log, self.H_target, n_steps)
        
        self.solver_history.append({"H":h_start_log, "Err":0, "Status":"Init", "Type":"Start"})
        
        print("-" * 65)
        print(f"   {'Type':<12} | {'Height (H)':<12} | {'Error':<12} | {'Status'}")
        print("-" * 65)
        
        method = 'trf'
        
        for i, h in enumerate(steps):
            self.H = h
            is_last = (i == len(steps) - 1)
            
            # Hybrid Solver Switching
            if i > 0.85 * n_steps: method = 'lm'
            
            try:
                tol = 2.3e-16
                res = least_squares(
                    self._residuals, x_curr, args=(h,), 
                    method=method, tr_solver='exact', 
                    ftol=tol, xtol=tol, gtol=tol, 
                    max_nfev=8000
                )
                x_curr = res.x 
                err = np.mean(np.abs(res.fun))
                success = res.success
            except Exception:
                success = False
                err = 999.0

            status_code = "OK"
            if err > 1e-5: status_code = "FAIL"
            if err > 1.0: status_code = "FAIL"

            if is_last:
                if err < 1e-10: status_code = "CONVERGED"
                elif err < 2e-3: status_code = "ACCEPTED"
                else: status_code = "DRIFT"
            
            step_label = "Final" if is_last else f"Step {i+1}"
            
            if i == 0 or i % 5 == 0 or i >= len(steps) - 5:
                print(f"   {step_label:<12} | {self.H:.3f}        | {err:.1e}      | {status_code}")
                
            self.solver_history.append({"H":h, "Err":err, "Status":status_code, "Type":step_label})

        # D. Final Polish 
        if status_code != "CONVERGED":
             try:
                print("       -> Attempting Final Polish...", end="", flush=True)
                
                res_final = least_squares(
                    self._residuals, x_curr, args=(self.H,), 
                    method='lm', 
                    ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
                    max_nfev=1000 
                )
                print(" Done.")
                
                err_lm = np.mean(np.abs(res_final.fun))
                
                if err_lm < err:
                    x_curr = res_final.x
                    new_status = "CONVERGED" if err_lm < 1e-10 else "ACCEPTED"
                    self.solver_history[-1]['Err'] = err_lm
                    self.solver_history[-1]['Status'] = new_status
                    print(f"   {'Final Polish':<12} | {self.H:.3f}        | {err_lm:.1e}      | RECOVERED")
             except:
                print("") 
                pass

        # E. Unpack Final Solution
        self.k, self.eta_nodes, self.Bj, self.Q, self.R = self._unpack_state(x_curr)
        self.L = 2*np.pi/self.k; self.c = self.L/self.T_target
        self.converged = True

    def _compute_elevation_coeffs(self):
        N = self.N
        self.Ej = np.zeros(N, dtype=DTYPE)
        for j in range(1, N+1):
            sum_cos = 0.0
            for m in range(N+1):
                val = (self.eta_nodes[m] - self.d) * np.cos(j * m * np.pi / N)
                weight = 0.5 if (m==0 or m==N) else 1.0
                sum_cos += val * weight
            self.Ej[j-1] = (2.0/N) * sum_cos

    def _calculate_integral_properties(self):
        k = self.k; d = self.d; g = self.g; c = self.c; rho = RHO
        Q_frame = self.Q
        R_bern = self.R
        
        if self.current_type == 'Eulerian':
            self.prop_u1 = self.Uc
            self.prop_U_frame = c - self.Uc
            self.prop_u2 = c - Q_frame/d
        else:
            self.prop_u2 = self.Uc
            self.prop_u1 = c - Q_frame/d
            self.prop_U_frame = Q_frame/d
            
        self.prop_q_vol = self.prop_U_frame * d - Q_frame 
        self.prop_I = rho * (c * d - Q_frame) 
        
        self._compute_elevation_coeffs()
        self.prop_PE = 0.25 * rho * g * np.sum(self.Ej**2) 
        self.prop_KE = 0.5 * (c * self.prop_I - self.prop_u1 * Q_frame * rho)
        
        KE = self.prop_KE; PE = self.prop_PE; I = self.prop_I
        r_excess = R_bern - g*d
        
        # 1. Keep ub2_calculation as it was (Integral Method)
        u_bed_sq_sum = 0.0
        x_nds = np.linspace(0, self.L, self.N * 4) 
        
        for x_loc in x_nds:
            u, _, _, _, _ = self.get_kinematics_at_y(0.0, x_loc)
            u_bed_sq_sum += u**2
            
        self.prop_ub2 = u_bed_sq_sum / len(x_nds)
        
        # 2. Sxx and F (using Algebraic ub2 for consistency with Fenton)
        # Fenton uses 2(R-gd) - c^2 for these specific flux/power calculations
        ub2_alg = 2.0 * (self.R - g * d) - c**2
        
        # Use ub2_alg ONLY for Sxx and F
        self.prop_Sxx = 4.0 * KE - 3.0 * PE + ub2_alg * (rho * d) + 2.0 * self.prop_u1 * rho * Q_frame
        
        term_energy_transport = c * (3.0 * KE - 2.0 * PE)
        term_bed_work = 0.5 * ub2_alg * (I + rho * c * d)
        term_current_interact = c * self.prop_u1 * rho * Q_frame
        self.prop_F = term_energy_transport + term_bed_work + term_current_interact
        
        self.prop_S = (self.prop_Sxx - 2.0 * c * self.prop_I + rho * (c**2 + 0.5 * g * d) * d)
        self.prop_R = self.R; self.prop_r = r_excess

    def get_eta_at_x(self, x):
        def func(y):
            # Calls the optimized basis function to speed up eta finding
            S, _ = _fast_basis(self.k, self.d, self.N, y)
            S = S.flatten()
            psi_p = np.sum(self.Bj * S * np.cos(np.arange(1,self.N+1)*self.k*x)) * np.sqrt(self.g/self.k**3)
            return -self.prop_U_frame*y + psi_p + self.Q
        return least_squares(func, self.d, ftol=1e-14, xtol=1e-14).x[0]

    def get_kinematics_at_y(self, y, x):
        """
        Public interface for kinematics.
        Now redirects to the static JIT-compiled _fast_kinematics kernel.
        """
        return _fast_kinematics(
            y, x, self.k, self.d, self.N, self.Bj, 
            self.g, self.c, self.prop_U_frame, self.R, RHO
        )

# ==============================================================================
#  SECTION 4: HYDRODYNAMIC FORCE MODULE
# ==============================================================================

def get_morison_coefficients(mg_thickness):
    print("\n" + "="*80 + "\n MORISON COEFFICIENTS SELECTION\n" + "="*80)
    
    presets = {
        "1": ("BS 6349-1",                     (0.70, 2.00), (1.30, 2.00)),
        "2": ("USACE (CEM)",                   (0.70, 1.50), (1.20, 1.50)),
        "3": ("DNV-RP-C205 (North Sea)",       (0.65, 1.60), (1.15, 1.30)),
        "4": ("API RP 2A-WSD",                 (0.65, 1.60), (1.05, 1.20)),
        "5": ("ISO 19902",                     (0.65, 1.60), (1.05, 1.20)),
        "6": ("User Defined (Manual Input)",   None,         None)
    }

    for k, v in presets.items(): 
        print(f"{k}. {v[0]}")
    
    choice = str(get_input("Select Method [1-6]", "1"))
    
    is_smooth = (mg_thickness <= 0.001)
    print(f"   -> Detected Surface State: {'SMOOTH' if is_smooth else 'ROUGH'} (mg = {mg_thickness:.3f}m)")

    if choice == "6":
        c_d_in = get_input("Enter Drag Coeff (Cd)", 1.30)
        c_m_in = get_input("Enter Inertia Coeff (Cm)", 2.00)
        return float(c_m_in), float(c_d_in), "User Defined"

    src, smooth_vals, rough_vals = presets.get(choice, presets["1"])
    cd, cm = smooth_vals if is_smooth else rough_vals
    
    return cm, cd, src

def scan_force(wave, dia, mg, cm, cd):
    """
    Force Integration Scanner.
    [OPTIMIZATION] Now utilizes the _fast_force_integral JIT kernel.
    """
    if not wave.converged or wave.k <= 1e-9:
        return {
            'F': 0.0, 'M': 0.0, 'Ph': 0.0, 
            'Fd': 0.0, 'Fi': 0.0, 'FSlam': 0.0,
            'Breaking': False, 'Pr': [], 'MF': 0.0, 'MZ': 0.0,
            'Max_M_Abs': 0.0
        }

    d_eff = dia + 2*mg 
    
    # --- Helper: Total Force at Phase 'ph' ---
    def get_force_vector(ph, num_depth_steps=200):
        # 1. Kinematic Setup (z=0 is MWL)
        eta = wave.get_eta_at_x(-ph/wave.k) - wave.d 
        
        # Linear grid sufficient for non-slamming
        z_nodes = np.linspace(-wave.d, eta, num_depth_steps)
        zs = np.sort(z_nodes)
        
        # [OPTIMIZATION] Replaced explicit Python loop with call to JIT kernel.
        # This speeds up the integration significantly.
        f_tot_sum, m_tot_sum, fd_tot, fi_tot = _fast_force_integral(
            zs, wave.d, wave.k, ph, wave.N, wave.Bj, 
            wave.g, wave.c, wave.prop_U_frame, wave.R, 
            RHO, cd, cm, d_eff
        )
            
        return f_tot_sum, m_tot_sum, fd_tot, fi_tot

    # --- STEP 1: Coarse Scan ---
    phases = np.linspace(0, 2*np.pi, 180)   
    
    best_ph_guess = 0.0
    max_f_guess = 0.0
    max_m_abs = 0.0  
    
    for ph in phases:
        f_val, m_val, _, _ = get_force_vector(ph, num_depth_steps=100) 
        
        if abs(f_val) > max_f_guess:
            max_f_guess = abs(f_val)
            best_ph_guess = ph
            
        if abs(m_val) > max_m_abs:
            max_m_abs = abs(m_val)
            
    # --- STEP 2: Fine Optimization ---
    def optim_target(x):
        f, _, _, _ = get_force_vector(x, num_depth_steps=200)
        return -abs(f)

    bnds = (best_ph_guess - 0.2, best_ph_guess + 0.2)
    if bnds[0] < 0: bnds = (0, 0.4) 
    
    res_opt = minimize_scalar(optim_target, bounds=bnds, method='bounded')
    best_ph = res_opt.x

    # --- STEP 3: Final Calculation at Peak ---
    f_final, m_final, fd_final, fi_final = get_force_vector(best_ph, num_depth_steps=500)
    
    # --- STEP 4: Generate Force Density Profile for Plotting ---
    # (This part is only run once at the end, so no need for JIT optimization here)
    eta = wave.get_eta_at_x(-best_ph/wave.k) - wave.d
    
    disp_z = np.linspace(eta, -wave.d, 51)
    prof = []
    mx_lf = 0; mx_lz = 0
    
    for z_mwl in disp_z:
        if z_mwl > eta: continue 

        y_bed = z_mwl + wave.d
        u, _, ax, _, p = wave.get_kinematics_at_y(y_bed, -best_ph/wave.k)
        
        fd = 0.5*RHO*cd*d_eff*u*abs(u)
        fi = RHO*cm*(np.pi*d_eff**2/4)*ax
        
        ft = fd + fi 
        p_dyn = p + RHO*wave.g*z_mwl
        
        prof.append({'z':z_mwl, 'u':u, 'ax':ax, 'p':p_dyn, 'fd':fd, 'fi':fi, 'fs':0.0, 'ftot':ft})
        
        if abs(ft) > mx_lf: 
            mx_lf = abs(ft)
            mx_lz = z_mwl
        
    return {
        'F': f_final, 
        'M': m_final,       
        'Ph': best_ph, 
        'Fd': fd_final, 'Fi': fi_final, 'FSlam': 0.0,
        'Breaking': False, 'Pr': prof, 
        'MF': mx_lf, 'MZ': mx_lz,
        'Max_M_Abs': max_m_abs 
    }
	
# ==============================================================================
#  SECTION 5: REPORTING & VISUALIZATION
# ==============================================================================

def generate_report(wave, res, h, d, t, uc, ct, dia, mg, deff, kc, src, cd, cm, re_num):
    print("================================================================================")
    print(" WAVE FORCE CALCULATOR - EXECUTIVE SUMMARY")
    print("================================================================================")

    if not wave.converged or wave.k <= 1e-9:
        print("\n [!] SOLVER FAILED: SKIPPING QUANTITATIVE REPORT TO PREVENT ERRORS.")
        return

    hb_limit = 0.142 * wave.L * np.tanh(wave.k * d)
    deep_ratio = d / wave.L
    
    regime = "SHALLOW"
    if deep_ratio > 0.05 and deep_ratio < 0.5: regime = "INTERMEDIATE"
    if deep_ratio >= 0.5: regime = "DEEP WATER"
    
    steepness = h / wave.L
    ursell = (h * wave.L**2) / (d**3)
    lever_arm = res['M'] / res['F'] if abs(res['F']) > 1e-4 else 0.0

    conv_status = "CONVERGED" if wave.converged else "FAILED"
    err_val = wave.solver_history[-1]['Err'] if wave.solver_history else 0.0
    
    print(f" SOLVER STATUS:        {conv_status} (Final Residual: {err_val:.1e})")
    print(f" ALGORITHM:            Fenton Fourier Stream Function (Order {wave.N})")
    print(f" HYDRODYNAMICS:        {regime} WATER")
    print(f"                       d/L = {deep_ratio:.4f}  |  H/L = {steepness:.4f}  |  Ur = {ursell:.1f}")
    
    brk_msg = "STABLE (No Breaking)"
    if h > hb_limit: brk_msg = "CAUTION: WAVE NEAR BREAKING LIMIT"
    
    print(f" STABILITY CHECK:      {brk_msg} (H/d = {h/d:.3f})")
    print(f"                       (Limit H ~ {hb_limit:.2f} m based on Miche Criterion)")
    
    print("-" * 80)
    
    def print_force_row(label, value, unit=""):
        if isinstance(value, (float, np.floating)):
            val_str = f"{value:.4f}"
        else:
            val_str = str(value)
        print(f" {label:<39} | {val_str:<15} | {unit}")

    max_otm = res.get('Max_M_Abs', res['M'])

    print_force_row("MAX. BASE SHEAR:", res['F']/1000, "kN")
    print_force_row("  |-> Drag Comp.:", res['Fd']/1000, "kN")
    print_force_row("  |-> Inertia Comp.:", res['Fi']/1000, "kN")
        
    print_force_row("MAX. OTM (MUDLINE):", max_otm/1000, "kNm")
    print_force_row("EFFECTIVE LEVER ARM:", lever_arm, "m (Height from Seabed)")

    print("")
    
    print("================================================================================")
    print(" 1. ENVIRONMENTAL & STRUCTURE DATA")
    print("================================================================================")
    print_row("Wave Height (H)", h, "m")
    print_row("Wave Period (T)", t, "s")
    print_row("Water Depth (d)", d, "m")
    print_row("Current Velocity (Uc)", uc, "m/s")
    print_row("Current Definition", ct, "-")
    print_row("Local Gravity (g)", G_STD, "m/s2")
    print_row("Kinematic Viscosity (nu)", NU_SEAWATER*1e6, "10^-6 m2/s")
    print("---------------------------------------------------------------------------")
    print_row("Pile Diameter", dia, "m")
    print_row("Marine Growth", mg, "m")
    print_row("Effective Diameter (D)", deff, "m")
    print_row("Roughness Ratio (2*mg/D)", 2*mg/dia, "-")
    print("---------------------------------------------------------------------------")
    print_row("Calculated KC Number", kc, "-")
    print_row("Reynolds Number (Re)", re_num/1e6, "10^6 -")
    print_row("Surface State", "ROUGH" if mg>0 else "SMOOTH", "-")
    print_row("Coefficient Source", src, "-")
    print_row("Drag Coefficient (Cd)", cd, "-")
    print_row("Inertia Coefficient (Cm)", cm, "-")
    
    print("\n" + "="*80)
    print(f" 2. FENTON STREAM FUNCTION SOLUTION (ORDER {wave.N})")
    print("================================================================================")
    print("\n   --- SOLVER CONVERGENCE HISTORY ---")
    print(f"   {'STEP TYPE':<12} | {'TARGET H (m)':<12} | {'MEAN ERROR':<12} | {'STATUS'}")
    print("   " + "-"*55)
    for l in wave.solver_history:
        print(f"   {l['Type']:<12} | {l['H']:<12.3f} | {l['Err']:<12.1e} | {l['Status']}")
    print("   " + "-"*55)
    
    k=wave.k; g=wave.g; rho=RHO
    print("\n" + "-"*100)
    print("   # INTEGRAL QUANTITIES - FENTON (1988) DEFINITIONS")
    print("   # (1) Quantity  (2) Symbol  (3) Dimensionless/(g,k)  (4) Dimensionless/(g,d)")
    print("-" * 100)
    
    def pr_fenton(name, sym, phys, scale_gk, scale_gd):
        print(f"{name:<35} | {sym:<10} | {phys*scale_gk:<15.8f} | {phys*scale_gd:<15.8f}")
        
    s_len_k = k; s_len_d = 1.0/d
    s_tim_k = np.sqrt(g*k); s_tim_d = np.sqrt(g/d)
    s_vel_k = 1.0/np.sqrt(g/k); s_vel_d = 1.0/np.sqrt(g*d)
    s_flx_k = k**1.5/np.sqrt(g); s_flx_d = 1.0/np.sqrt(g * d**3)
    s_enr_k = k/g; s_enr_d = 1.0/(g*d) 
    s_mom_k = k**1.5/(rho*np.sqrt(g)); s_mom_d = 1.0/(rho*d*np.sqrt(g*d))
    s_prs_k = k**2/(rho*g); s_prs_d = 1.0/(rho*g*d**2) 
    s_pwr_k = k**2.5/(rho*g**1.5); s_pwr_d = 1.0/(rho * g**1.5 * d**2.5)
    s_ub2_k = 1.0/(g/k); s_ub2_d = 1.0/(g*d)
    
    pr_fenton("Water depth", "(d)", d, s_len_k, s_len_d)
    pr_fenton("Wave length", "(lambda)", wave.L, s_len_k, s_len_d)
    pr_fenton("Wave height", "(H)", h, s_len_k, s_len_d)
    pr_fenton("Wave period", "(tau)", t, s_tim_k, s_tim_d)
    pr_fenton("Wave speed", "(c)", wave.c, s_vel_k, s_vel_d)
    pr_fenton("Eulerian current", "(u1_)", wave.prop_u1, s_vel_k, s_vel_d)
    pr_fenton("Stokes current", "(u2_)", wave.prop_u2, s_vel_k, s_vel_d)
    pr_fenton("Mean fluid speed in frame", "(U_)", wave.prop_U_frame, s_vel_k, s_vel_d)
    pr_fenton("Volume flux due to waves", "(q)", wave.prop_q_vol, s_flx_k, s_flx_d)
    pr_fenton("Bernoulli constant (Excess)", "(r)", wave.prop_r, s_enr_k, s_enr_d)
    pr_fenton("Volume flux (Total)", "(Q)", wave.Q, s_flx_k, s_flx_d)
    pr_fenton("Bernoulli constant (Total)", "(R)", wave.prop_R, s_enr_k, s_enr_d)
    pr_fenton("Momentum flux (Total)", "(S)", wave.prop_S, s_prs_k, s_prs_d)
    pr_fenton("Impulse", "(I)", wave.prop_I, s_mom_k, s_mom_d)
    pr_fenton("Kinetic energy", "(T)", wave.prop_KE, s_prs_k, s_prs_d)
    pr_fenton("Potential energy", "(V)", wave.prop_PE, s_prs_k, s_prs_d)
    pr_fenton("Mean sq bed velocity", "(ub2_)", wave.prop_ub2, s_ub2_k, s_ub2_d)
    pr_fenton("Radiation stress", "(Sxx)", wave.prop_Sxx, s_prs_k, s_prs_d)
    pr_fenton("Wave power", "(F)", wave.prop_F, s_pwr_k, s_pwr_d)

    print("\n" + "="*80)
    print(" 3. FOURIER COEFFICIENTS (Bj & Ej)")
    print("================================================================================")
    print(f"   Wavenumber k = {k:.6f} rad/m")
    print(f"   Wave Length L= {wave.L:.4f} m")
    print("   " + "-"*65)
    print(f"   {'j':<5} | {'B[j] (Stream)':<18} | {'E[j] (Elevation)':<18}")
    print("   " + "-"*65)
    for i, (bj, ej) in enumerate(zip(wave.Bj, wave.Ej)):
        print(f"   {i+1:<5} | {bj:<18.8e} | {ej * k:<18.8e}")
    print("   " + "-"*65)
    
    print("\n" + "="*80)

    print(" 4. FORCE & MOMENT CALCULATION RESULTS")
    print("================================================================================")
    print_row("Maximum Total Force (Base Shear)", res['F']/1000, "kN")
    print_row("Phase of Max Force", np.degrees(res['Ph']), "deg")
    print_row("Time of Max Force", res['Ph']/wave.k/wave.c, "s")
    print("------------------------------------------------------------")
    print_row("Max Overturning Moment (Sync)", res['M']/1000, "kNm")
    print_row("Max Overturning Moment (True)", res.get('Max_M_Abs', res['M'])/1000, "kNm")
    
    if abs(res['F']) > 1e-4:
        elev_bed = res['M']/res['F']
        elev_msl = elev_bed - d
        print_row("Center of Effort (from Bed)", elev_bed, "m")
        print_row("Center of Effort (from MSL)", elev_msl, "m")
    else:
        print_row("Center of Effort", "N/A", "(Force ~ 0)")
    
    print("------------------------------------------------------------")
    print_row("Drag Component @ Max Load", res['Fd']/1000, "kN")
    print_row("Inertia Component @ Max Load", res['Fi']/1000, "kN")
    print("------------------------------------------------------------")
    print_row("Max Local Force Density", res['MF']/1000, "kN/m")
    print_row("Elevation of Max Local Load", res['MZ'], "m")
    
    print("\n   --- FORCE DISTRIBUTION PROFILE AT MAX LOAD PHASE ---")
    print("   " + "-"*100)
    print(f"   {'Elev Z(m)':<12} | {'Vel(m/s)':<12} | {'Acc(m/s2)':<12} | {'P_dyn(kPa)':<12} | {'Fd (kN/m)':<12} | {'Fi (kN/m)':<12} | {'Ftot (kN/m)':<12}")
    print("   " + "-"*100)
    for p in res['Pr']:
        print(f"   {p['z']:<12.3f} | {p['u']:<12.3f} | {abs(p['ax']):<12.3f} | {p['p']/1000:<12.3f} | {p['fd']/1000:<12.3f} | {abs(p['fi'])/1000:<12.3f} | {p['ftot']/1000:<12.3f}")
    print("   " + "-"*100)

def generate_plots(wave, res, h, d, t, dia, mg, cm, cd):
    """
    Generates PDF Report + Full High-Res PNGs for every Figure.
    
    Production Routine:
    - Figure 1: Wave Kinematics (Profile & Velocities).
    - Figure 2: Wave Dynamics (Accelerations & Pressures).
    - Figure 3: Solver Validation (Linear vs Nonlinear).
    - Figure 4: 2D Wave Field Visualization.
    - Figure 5: Force Spectrum & Harmonic Reconstruction.
    - Figure 6: Vertical Pressure Profiles (Fluid & Structural).
    - Figure 7: Hydrodynamic Forces & Moments Time Series.
    
    Format: A3 Landscape (16.5 x 11.7 inches).
    """
    # Use Agg backend to prevent GUI windows from opening during batch processing
    plt.switch_backend('Agg')
    
    print("Generating High-Resolution Plots (PDF + Full PNGs)...")
    
    # Ensure output directory exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Structural parameters
    d_eff = dia + 2*mg
    A3_SIZE = (16.5, 11.7) 
    
    # --- Visualization Helper Functions ---
    def add_msl_line(ax):
        """Adds Mean Sea Level dashed line."""
        ax.axhline(0, color='darkblue', linestyle='--', linewidth=3, alpha=0.8, label="Mean Sea Level (MSL)")

    def add_water_fill(ax):
        """Adds light blue background for water column."""
        ax.axhspan(-d, 0, color='deepskyblue', alpha=0.1, zorder=0)

    def save_figure(fig, filename):
        """Saves figure to PNG if enabled globally."""
        if globals().get('DEF_SAVE_PNGS', True):
            fig.savefig(f"{output_dir}/{filename}.png", dpi=300)

    # Begin PDF Context
    with PdfPages('plots.pdf') as pdf:
        
        # ======================================================================
        # FIG 1: WAVE KINEMATICS
        # ======================================================================
        fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=A3_SIZE)
        fig1.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.92, wspace=0.20)
        fig1.suptitle("1. Wave Kinematics (Profile & Velocities)", fontsize=18, fontweight='bold')
        
        # --- Plot 1a: Surface Elevation (Multi-Cycle) ---
        # Generate time series for N cycles
        plot_phases = np.linspace(0, 2 * np.pi * DEF_PLOT_CYCLES, int(200 * DEF_PLOT_CYCLES))
        times_p = plot_phases / (2 * np.pi) * t
        etas = [wave.get_eta_at_x(p/wave.k)-d for p in plot_phases]
        
        add_water_fill(ax1a)
        add_msl_line(ax1a)
        ax1a.axhline(-d, color='brown', linestyle='-', linewidth=4, label="Seabed")
        
        ax1a.plot(times_p, etas, 'b-', lw=4, label="Fenton (Nonlinear)")
        ax1a.fill_between(times_p, etas, -d, color='skyblue', alpha=0.4)
        
        # Annotate Min/Max Elevation
        eta_max = max(etas); eta_min = min(etas)
        ax1a.text(0, eta_max, f" Crest: +{eta_max:.2f}m", color='b', ha='left', va='bottom', fontsize=12, fontweight='bold')
        ax1a.text(t/2, eta_min, f" Trough: {eta_min:.2f}m", color='b', ha='center', va='top', fontsize=12)

        ax1a.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
        ax1a.set_ylabel("Elevation (m)", fontsize=14, fontweight='bold')
        ax1a.set_title("Free Surface Profile", fontsize=16)
        ax1a.grid(True, alpha=0.5)
        ax1a.set_xlim(0, t * DEF_PLOT_CYCLES)
        ax1a.set_ylim(bottom=-d) 
        
        # --- Dynamic Phase Axis (Top X-Axis for Fig 1a) ---
        max_deg = 360 * DEF_PLOT_CYCLES
        ax1a_ph = ax1a.twiny()
        ax1a_ph.set_xlim(0, max_deg)
        ax1a_ph.set_xlabel("Phase (degrees)", color='darkred', fontsize=12)
        
        # Smart tick spacing to avoid clutter on multi-cycle plots
        if DEF_PLOT_CYCLES <= 2: tick_step = 90
        elif DEF_PLOT_CYCLES <= 4: tick_step = 180
        else: tick_step = 360
            
        ax1a_ph.set_xticks(np.arange(0, max_deg + 0.1, tick_step))
        ax1a_ph.tick_params(axis='x', colors='darkred', labelsize=10)
        
        add_param_box(ax1a, wave)

        # --- Plot 1b: Velocity Profiles ---
        eta_crest = wave.get_eta_at_x(0) - d
        eta_trough = wave.get_eta_at_x(np.pi/wave.k) - d
        z_crest = np.linspace(-d, eta_crest, 100)
        z_trough = np.linspace(-d, eta_trough, 100)
        z_mid = np.linspace(-d, wave.get_eta_at_x(np.pi/2/wave.k)-d, 100)

        u_crest = [wave.get_kinematics_at_y(z+d, 0)[0] for z in z_crest]
        u_trough = [wave.get_kinematics_at_y(z+d, np.pi/wave.k)[0] for z in z_trough]
        w_mid = [wave.get_kinematics_at_y(z+d, (np.pi/2)/wave.k)[1] for z in z_mid]

        add_water_fill(ax1b)
        add_msl_line(ax1b)
        ax1b.axhline(-d, color='brown', linestyle='-', linewidth=4)

        ax1b.plot(u_crest, z_crest, 'r-', lw=4, label="Horiz U (Crest)")
        ax1b.plot(u_trough, z_trough, 'b--', lw=4, label="Horiz U (Trough)")
        ax1b.plot(w_mid, z_mid, 'g-.', lw=4, label="Vert W (ZeroCross)")
        
        u_surf = u_crest[-1]
        ax1b.plot(u_surf, eta_crest, 'ro', markersize=10)
        ax1b.text(u_surf, eta_crest, f" MAX $u_{{surf}}$\n{u_surf:.2f}", color='r', fontsize=12, fontweight='bold', ha='right', va='bottom')

        ax1b.set_xlabel("Velocity (m/s)", fontsize=14, fontweight='bold')
        ax1b.set_ylabel("Depth z (m)", fontsize=14, fontweight='bold')
        ax1b.set_title("Velocity Profiles", fontsize=16)
        ax1b.legend(fontsize=12, loc='lower right')
        ax1b.grid(True, alpha=0.5)
        ax1b.set_ylim(bottom=-d) 
        
        pdf.savefig(fig1)
        save_figure(fig1, "Figure_1")
        plt.close(fig1)

        # ======================================================================
        # FIG 2: WAVE DYNAMICS
        # ======================================================================
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=A3_SIZE)
        fig2.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.92, wspace=0.20)
        fig2.suptitle("2. Wave Dynamics (Accelerations & Pressures)", fontsize=18, fontweight='bold')
        
        # --- Plot 2a: Accelerations ---
        phase_max_load = res['Ph'] 
        eta_max_load = wave.get_eta_at_x(phase_max_load/wave.k) - d
        z_max_load = np.linspace(-d, eta_max_load, 100)
        acc_max_load = [abs(wave.get_kinematics_at_y(z+d, -phase_max_load/wave.k)[2]) for z in z_max_load]
        
        eta_inertia = wave.get_eta_at_x(np.pi/2/wave.k) - d
        z_inertia = np.linspace(-d, eta_inertia, 100)
        acc_inertia = [abs(wave.get_kinematics_at_y(z+d, -np.pi/2/wave.k)[2]) for z in z_inertia]
        
        add_water_fill(ax2a)
        add_msl_line(ax2a)
        ax2a.axhline(-d, color='brown', linestyle='-', linewidth=4, label="Seabed")
        
        ax2a.plot(acc_max_load, z_max_load, 'k-', lw=4, label="At Max Load Phase")
        ax2a.plot(acc_inertia, z_inertia, 'b--', lw=3, alpha=0.6, label="Max Inertia Envelope")
        
        acc_surf = acc_max_load[-1]
        ax2a.plot(acc_surf, eta_max_load, 'ko', markersize=10)
        ax2a.text(acc_surf, eta_max_load, f" MAX $a_{{x}}$\n {acc_surf:.2f} m/s^2", color='k', fontsize=12, fontweight='bold', ha='left', va='top')
        
        ax2a.set_xlabel("Acceleration |ax| (m/s^2)", fontsize=14, fontweight='bold')
        ax2a.set_ylabel("Depth z (m)", fontsize=14, fontweight='bold')
        ax2a.set_title("Acceleration Profiles", fontsize=16)
        ax2a.legend(fontsize=12); ax2a.grid(True, alpha=0.5)
        ax2a.set_ylim(bottom=-d) 

        # --- Plot 2b: Pressure ---
        def get_pdyn(z, ph):
            _, _, _, _, p_tot = wave.get_kinematics_at_y(z+d, -ph/wave.k)
            # This returns static + dynamic head, convert to kPa
            return (p_tot + RHO * G_STD * z) / 1000.0 

        pd_max = [get_pdyn(z, phase_max_load) for z in z_max_load]
        eta_tr = wave.get_eta_at_x(np.pi/wave.k) - d
        z_tr = np.linspace(-d, eta_tr, 100)
        pd_tr = [get_pdyn(z, np.pi) for z in z_tr]
        
        add_water_fill(ax2b)
        add_msl_line(ax2b)
        ax2b.axhline(-d, color='brown', linestyle='-', linewidth=4)
        
        ax2b.plot(pd_max, z_max_load, 'r-', lw=4, label="At Max Load")
        ax2b.plot(pd_tr, z_tr, 'b--', alpha=0.6, label="At Trough")
        
        pd_surf = pd_max[-1]
        ax2b.plot(pd_surf, eta_max_load, 'ro', markersize=10)
        ax2b.text(pd_surf, eta_max_load, f" MAX $P_{{dyn}}$\n {pd_surf:.2f} kPa", color='r', fontsize=12, fontweight='bold', ha='right', va='bottom')
        
        ax2b.set_xlabel("Dyn. Pressure (kPa)", fontsize=14, fontweight='bold')
        ax2b.set_ylabel("Depth z (m)", fontsize=14, fontweight='bold')
        ax2b.set_title("Dynamic Pressure Profiles", fontsize=16)
        ax2b.legend(fontsize=12); ax2b.grid(True, alpha=0.5)
        ax2b.set_ylim(bottom=-d) 
        
        pdf.savefig(fig2)
        save_figure(fig2, "Figure_2")
        plt.close(fig2)

        # ======================================================================
        # FIG 3: WAVE PROFILE COMPARISON
        # ======================================================================
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=A3_SIZE, sharex=True)
        fig3.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.92, hspace=0.20)
        fig3.suptitle("3. Solver Validation: Linear vs. Nonlinear Physics", fontsize=18, fontweight='bold')
        
        eta_airy = (h / 2.0) * np.cos(2 * np.pi * times_p / t)
        
        # --- Plot 3a: Profiles ---
        add_water_fill(ax3a)
        add_msl_line(ax3a)
        
        ax3a.plot(times_p, etas, 'b-', lw=4, label=f"Fenton (Order {wave.N})")
        ax3a.plot(times_p, eta_airy, 'r--', lw=3, alpha=0.7, label="Airy (Linear)")
        ax3a.axhline(h/2, color='r', linestyle=':', alpha=0.5); ax3a.axhline(-h/2, color='r', linestyle=':', alpha=0.5)
        
        ax3a.fill_between(times_p, etas, eta_airy, where=(np.array(etas) > np.array(eta_airy)), color='orange', alpha=0.3)

        ax3a.set_ylabel("Elevation (m)", fontsize=14, fontweight='bold')
        ax3a.legend(loc='upper right', fontsize=12)
        ax3a.grid(True, linestyle='--', alpha=0.5)
        ax3a.text(0, max(etas)*1.05, "Crest Peaking\n(Sharper)", ha='center', color='blue', fontsize=12, fontweight='bold')
        ax3a.text(t/2, min(etas)*1.05, "Trough Flattening\n(Shallower)", ha='center', va='top', color='blue', fontsize=12, fontweight='bold')
        ax3a.set_title("Wave Profile Comparison", fontsize=16)
        ax3a.set_ylim(bottom=-d) 
        
        # --- Plot 3b: Residuals ---
        diff = np.array(etas) - np.array(eta_airy)
        ax3b.plot(times_p, diff, 'k-', lw=3, label="Nonlinear Residual")
        ax3b.fill_between(times_p, diff, 0, color='gray', alpha=0.1)
        ax3b.axhline(0, color='k', lw=1)
        
        idx_max_diff = np.argmax(np.abs(diff))
        ax3b.plot(times_p[idx_max_diff], diff[idx_max_diff], 'ro')
        ax3b.text(times_p[idx_max_diff], diff[idx_max_diff], f" Max Deviation: {diff[idx_max_diff]:.2f}m", 
                 ha='left', fontweight='bold', color='r', fontsize=12)

        ax3b.set_ylabel("Deviation (m)", fontsize=14, fontweight='bold')
        ax3b.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
        ax3b.legend(loc='upper right', fontsize=12); ax3b.grid(True, linestyle='--', alpha=0.5)
        ax3b.set_xlim(0, t * DEF_PLOT_CYCLES)

        pdf.savefig(fig3)
        save_figure(fig3, "Figure_3")
        plt.close(fig3)

        # ======================================================================
        # FIG 4: 2D FIELDS
        # ======================================================================
        fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=A3_SIZE, sharex=True)
        fig4.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.90, hspace=0.20)
        fig4.suptitle(f"4. 2D Wave Field Visualization (H={h}m, T={t}s)", fontsize=18, fontweight='bold')
        
        n_x, n_z = 60, 40
        phases_grid = np.linspace(0, 2*np.pi, n_x)
        X_mesh = np.zeros((n_z, n_x)); Z_mesh = np.zeros((n_z, n_x))
        Vel_Mag = np.zeros((n_z, n_x)); P_Dyn = np.zeros((n_z, n_x))
        
        for i, ph in enumerate(phases_grid):
            eta_loc = wave.get_eta_at_x(ph/wave.k) - d
            z_col = np.linspace(-d, eta_loc, n_z)
            X_mesh[:, i] = np.degrees(ph); Z_mesh[:, i] = z_col
            for j, z in enumerate(z_col):
                u, w, _, _, p_tot = wave.get_kinematics_at_y(z+d, ph/wave.k)
                Vel_Mag[j, i] = np.sqrt(u**2 + w**2)
                P_Dyn[j, i] = (p_tot + RHO * G_STD * z) / 1000.0

        cf1 = ax4a.contourf(X_mesh, Z_mesh, Vel_Mag, levels=25, cmap='viridis', extend='max')
        cbar1 = fig4.colorbar(cf1, ax=ax4a, label="Velocity Mag (m/s)")
        surf = [wave.get_eta_at_x(p/wave.k)-d for p in phases_grid]
        ax4a.plot(np.degrees(phases_grid), surf, 'k-', lw=3)
        ax4a.fill_between(np.degrees(phases_grid), surf, max(surf)*1.2, color='white')
        add_msl_line(ax4a) 
        ax4a.set_ylabel("Depth z (m)", fontsize=14, fontweight='bold')
        ax4a.set_title("Velocity Magnitude Field", fontsize=16)
        ax4a.set_ylim(bottom=-d) 
        
        cf2 = ax4b.contourf(X_mesh, Z_mesh, P_Dyn, levels=25, cmap='plasma', extend='both')
        cbar2 = fig4.colorbar(cf2, ax=ax4b, label="Dyn. Pressure (kPa)")
        ax4b.plot(np.degrees(phases_grid), surf, 'k-', lw=3)
        ax4b.fill_between(np.degrees(phases_grid), surf, max(surf)*1.2, color='white')
        add_msl_line(ax4b) 
        ax4b.set_ylabel("Depth z (m)", fontsize=14, fontweight='bold'); ax4b.set_xlabel("Phase (degrees)", fontsize=14, fontweight='bold')
        ax4b.set_title("Dynamic Pressure Field", fontsize=16)
        ax4b.set_xlim(0, 360); ax4b.set_xticks([0, 90, 180, 270, 360])
        ax4b.set_ylim(bottom=-d) 
        
        pdf.savefig(fig4)
        save_figure(fig4, "Figure_4")
        plt.close(fig4)

        # ======================================================================
        # FIG 5: SPECTRA
        # ======================================================================
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=A3_SIZE)
        fig5.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.92, wspace=0.20)
        fig5.suptitle("5. Force Spectrum & Harmonic Reconstruction", fontsize=18, fontweight='bold')

        n_fft = 4096 
        t_fft = np.linspace(0, t, n_fft, endpoint=False) 
        ph_fft = (t_fft / t) * 2 * np.pi
        f_series = []
        for ph in ph_fft:
            # Using JIT here indirectly via scan_force's internal helper would be ideal,
            # but for plotting purposes we stick to the class method to keep code simple.
            # The bottleneck is not here (4096 points is fast compared to optimization).
            eta = wave.get_eta_at_x(-ph/wave.k) - d
            zs = np.linspace(-d, eta, 40); dz = zs[1] - zs[0]
            ft_sum, _, _, _ = _fast_force_integral(zs, wave.d, wave.k, ph, wave.N, wave.Bj, 
                                                   wave.g, wave.c, wave.prop_U_frame, wave.R, 
                                                   RHO, cd, cm, d_eff)
            f_series.append(ft_sum / 1000.0) 

        F_hat = np.fft.fft(f_series)
        freqs = np.fft.fftfreq(n_fft, d=(t_fft[1]-t_fft[0]))
        mag_half = np.abs(F_hat)[:n_fft//2] * (2.0 / n_fft)
        
        # --- Plot 5a: Spectrum ---
        ax5a.bar(freqs[:n_fft//2], mag_half, width=0.05, color='purple', alpha=0.7, label="Energy")
        f_fund = 1.0/t
        for h_mult in [1, 2, 3, 4, 5]:
            fh = h_mult * f_fund; idx = np.argmin(np.abs(freqs[:n_fft//2] - fh))
            val = mag_half[idx]
            ax5a.plot(fh, val, 'ro')
            ax5a.text(fh, val, f" {h_mult}f", ha='center', va='bottom', fontsize=12, color='darkred')
        
        ax5a.set_xlabel("Frequency (Hz)", fontsize=14, fontweight='bold'); ax5a.set_ylabel("Force Amp (kN)", fontsize=14, fontweight='bold')
        ax5a.set_xlim(0, 6.0/t); ax5a.grid(True, alpha=0.5)
        ax5a.set_title("Frequency Domain (Amplitude)", fontsize=16)
        
        add_param_box(ax5a, wave, f"Fund. Freq: {f_fund:.3f} Hz")

        # --- Plot 5b: Reconstruction ---
        coeffs = F_hat[:6]
        t_recon = np.linspace(0, t, 100)
        def reconstruct(n_harm, label, color, style):
            rec = np.zeros_like(t_recon, dtype=complex)
            for k in range(n_harm + 1):
                 mag = np.abs(F_hat[k]) * (1.0/n_fft if k==0 else 2.0/n_fft)
                 phase = np.angle(F_hat[k])
                 rec += mag * np.cos(2*np.pi*k*t_recon/t + phase)
            ax5b.plot(t_recon, np.real(rec), color=color, linestyle=style, lw=3, label=label)

        ax5b.plot(t_fft, f_series, 'k-', lw=5, alpha=0.2, label="Full Signal")
        reconstruct(1, "1f", "blue", "--"); reconstruct(3, "Sum 1-3f", "green", "-."); reconstruct(5, "Sum 1-5f", "red", "-")
        
        ax5b.set_xlabel("Time (s)", fontsize=14, fontweight='bold'); ax5b.set_ylabel("Recon. Force (kN)", fontsize=14, fontweight='bold')
        ax5b.legend(fontsize=12, loc='upper right'); ax5b.grid(True, alpha=0.5)
        ax5b.set_title("Harmonic Reconstruction", fontsize=16); ax5b.set_xlim(0, t)

        pdf.savefig(fig5)
        save_figure(fig5, "Figure_5")
        plt.close(fig5)

        # ======================================================================
        # FIG 6: FORCE DISTRIBUTIONS (PRESSURE kPa vs LINE LOAD kN/m)
        # ======================================================================
        fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=A3_SIZE)
        fig6.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.92, wspace=0.20)
        fig6.suptitle(f"6. Vertical Force Profiles at Max Load Phase ({np.degrees(res['Ph']):.1f}°)", fontsize=18, fontweight='bold')

        # --- 1. PREPARE DATA ---
        zs_prof = np.array([p['z'] for p in res['Pr']])
        
        # A. EQUIVALENT STRUCTURAL PRESSURES (kPa)
        #    Pressure = Force_Line / D_eff
        f_drag_press = np.array([p['fd'] for p in res['Pr']]) / 1000.0 / d_eff
        f_inert_press = np.array([p['fi'] for p in res['Pr']]) / 1000.0 / d_eff
        f_tot_press = np.array([p['ftot'] for p in res['Pr']]) / 1000.0 / d_eff

        # B. LINE LOAD DENSITIES (kN/m)
        #    Raw Morison output
        f_drag_line = np.array([p['fd'] for p in res['Pr']]) / 1000.0
        f_inert_line = np.array([p['fi'] for p in res['Pr']]) / 1000.0
        f_tot_line = np .array([p['ftot'] for p in res['Pr']]) / 1000.0

        # --- CHART 6a: EQUIVALENT FORCE PRESSURE (Left) ---
        add_water_fill(ax6a)
        add_msl_line(ax6a)
        ax6a.axhline(-d, color='brown', linestyle='-', linewidth=4, label="Seabed")
        
        ax6a.plot(f_drag_press, zs_prof, 'r--', linewidth=2, label="Drag Pressure")
        ax6a.plot(f_inert_press, zs_prof, 'g:', linewidth=2, label="Inertia Pressure")
        ax6a.plot(f_tot_press, zs_prof, 'k-', linewidth=4, label="Total Force Pressure")
        ax6a.fill_betweenx(zs_prof, 0, f_tot_press, color='gray', alpha=0.1)

        # Mark Max Pressure
        idx_max_p = np.argmax(np.abs(f_tot_press))
        ax6a.plot(f_tot_press[idx_max_p], zs_prof[idx_max_p], 'ro', markersize=10)
        ax6a.text(f_tot_press[idx_max_p], zs_prof[idx_max_p], 
                 f" Peak: {f_tot_press[idx_max_p]:.1f} kPa", 
                 color='darkred', fontweight='bold', ha='left', va='center')

        ax6a.set_xlabel("Equivalent Pressure (kPa) [F/D]", fontsize=14, fontweight='bold')
        ax6a.set_ylabel("Elevation z (m)", fontsize=14, fontweight='bold')
        ax6a.set_title("Force as Pressure (kPa)", fontsize=16)
        ax6a.legend(fontsize=12, loc='lower right')
        ax6a.grid(True, alpha=0.5)
        ax6a.set_ylim(bottom=-d)
		
        add_param_box(ax6a, wave)
        
        # --- CHART 6b: LINE LOAD DENSITY (Right) ---
        add_water_fill(ax6b)
        add_msl_line(ax6b)
        ax6b.axhline(-d, color='brown', linestyle='-', linewidth=4, label="Seabed")
        
        ax6b.plot(f_drag_line, zs_prof, 'r--', linewidth=2, label="Drag Load")
        ax6b.plot(f_inert_line, zs_prof, 'g:', linewidth=2, label="Inertia Load")
        ax6b.plot(f_tot_line, zs_prof, 'k-', linewidth=4, label="Total Line Load")
        ax6b.fill_betweenx(zs_prof, 0, f_tot_line, color='red', alpha=0.1)
        
        # Mark Max Line Load
        idx_max_l = np.argmax(np.abs(f_tot_line))
        ax6b.plot(f_tot_line[idx_max_l], zs_prof[idx_max_l], 'ro', markersize=10)
        ax6b.text(f_tot_line[idx_max_l], zs_prof[idx_max_l], 
                 f" Peak: {f_tot_line[idx_max_l]:.1f} kN/m\n @ z={zs_prof[idx_max_l]:.1f}m", 
                 color='darkred', fontweight='bold', fontsize=12, ha='left', va='center')

        ax6b.set_xlabel("Line Force Density (kN/m)", fontsize=14, fontweight='bold')
        ax6b.set_ylabel("Elevation z (m)", fontsize=14, fontweight='bold')
        ax6b.set_title("Force Distribution (kN/m)", fontsize=16)
        ax6b.legend(fontsize=12, loc='lower right')
        ax6b.grid(True, alpha=0.5)
        ax6b.set_ylim(bottom=-d)

        add_param_box(ax6b, wave)
        
        pdf.savefig(fig6)
        save_figure(fig6, "Figure_6")
        plt.close(fig6)

        # ======================================================================
        # FIG 7: FORCES & MOMENTS TIME SERIES
        # ======================================================================
        fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=A3_SIZE)
        fig7.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.92, wspace=0.20)
        fig7.suptitle("7. Hydrodynamic Forces & Moments (Time Series)", fontsize=18, fontweight='bold')
        
        base_phases = np.linspace(0, 2 * np.pi * DEF_PLOT_CYCLES, int(200 * DEF_PLOT_CYCLES)) 
        plot_phases = np.sort(np.append(base_phases, res['Ph']))
        times = plot_phases / (2 * np.pi) * t
        
        forces_total, forces_drag, forces_inertia, moments_total = [], [], [], []
        
        for ph in plot_phases:
            eta = wave.get_eta_at_x(-ph/wave.k) - d
            zs = np.linspace(-d, eta, 200); dz = zs[1] - zs[0]
            
            # Use JIT kernel for speed, then manually separate components for plotting
            f_tot_sum, m_tot_sum, fd_tot, fi_tot = _fast_force_integral(
                zs, wave.d, wave.k, ph, wave.N, wave.Bj, 
                wave.g, wave.c, wave.prop_U_frame, wave.R, 
                RHO, cd, cm, d_eff
            )

            forces_total.append(f_tot_sum/1000.0)
            forces_drag.append(fd_tot/1000.0)
            forces_inertia.append(fi_tot/1000.0)
            moments_total.append(m_tot_sum/1000.0)

        forces_total = np.array(forces_total)
        moments_total = np.array(moments_total)

        # --- Plot 7a: Forces ---
        ax7a.plot(times, forces_total, 'k-', lw=4, label="Total Force")
        ax7a.plot(times, forces_drag, 'r--', alpha=0.5, label="Drag")
        ax7a.plot(times, forces_inertia, 'g--', alpha=0.5, label="Inertia")

        max_f = res['F'] / 1000.0; idx_max_f = np.argmin(np.abs(plot_phases - res['Ph']))
        ax7a.plot(times[idx_max_f], max_f, 'ko')
        ax7a.text(times[idx_max_f], max_f, f" MAX: {max_f:.1f}kN", color='k', fontweight='bold', ha='left', va='top', fontsize=12)
        ax7a.set_xlabel("Time (s)", fontsize=14, fontweight='bold'); ax7a.set_ylabel("Force (kN)", fontsize=14, fontweight='bold')
        ax7a.set_title("Base Shear Force", fontsize=16); ax7a.legend(loc='upper right', fontsize=12); ax7a.grid(True, alpha=0.5)
        ax7a.set_xlim(0, t * DEF_PLOT_CYCLES)
        
        max_deg = 360 * DEF_PLOT_CYCLES
        ax7a_ph = ax7a.twiny(); ax7a_ph.set_xlim(0, max_deg)
        ax7a_ph.set_xlabel("Phase (degrees)", color='darkred', fontsize=12)
        tick_step = 90 if DEF_PLOT_CYCLES <= 2 else 180
        ax7a_ph.set_xticks(np.arange(0, max_deg + 0.1, tick_step))
        ax7a_ph.tick_params(axis='x', colors='darkred', labelsize=10)
        
        add_param_box(ax7a, wave, f"Max Force: {max_f:.2f} kN")

        # --- Plot 7b: Moments ---
        ax7b.plot(times, moments_total, color='saddlebrown', lw=4, label="Overt. Moment")
        ax7b.fill_between(times, moments_total, 0, color='brown', alpha=0.2)
        
        max_m_true = res.get('Max_M_Abs', res['M']) / 1000.0
        idx_max_m = np.argmax(np.abs(moments_total))
        
        ax7b.plot(times[idx_max_m], moments_total[idx_max_m], 'o', color='saddlebrown', markersize=10)
        ax7b.text(times[idx_max_m], moments_total[idx_max_m], f" MAX: {max_m_true:.1f}kNm", 
                 color='saddlebrown', fontweight='bold', ha='left', va='top', fontsize=12)

        ax7b.set_xlabel("Time (s)", fontsize=14, fontweight='bold')
        ax7b.set_ylabel("Moment (kNm)", fontsize=14, fontweight='bold')
        ax7b.set_title("Overturning Moment (Mudline)", fontsize=16)
        ax7b.grid(True, alpha=0.5)
        ax7b.set_xlim(0, t * DEF_PLOT_CYCLES)

        ax7b_ph = ax7b.twiny(); ax7b_ph.set_xlim(0, max_deg)
        ax7b_ph.set_xlabel("Phase (degrees)", color='darkred', fontsize=12)
        ax7b_ph.set_xticks(np.arange(0, max_deg + 0.1, tick_step))
        ax7b_ph.tick_params(axis='x', colors='darkred', labelsize=10)
        
        add_param_box(ax7b, wave, f"Max OTM: {max_m_true:.2f} kNm")

        pdf.savefig(fig7)
        save_figure(fig7, "Figure_7")
        plt.close(fig7)
		
# ==============================================================================
#  SECTION 6: MAIN ENTRY POINT
# ==============================================================================

def main():
    # 1. Console Session (Not Logged to File)
    print("================================================================================")
    print(" WAVE FORCES ON CILYNDRICAL PILES CALCULATOR ")
    print("================================================================================")
    
    # Inputs
    h = get_input("Wave Height (H)", DEF_WAVE_HEIGHT)
    t = get_input("Wave Period (T)", DEF_WAVE_PERIOD)
    d = get_input("Water Depth (d)", DEF_DEPTH)
    uc = get_input("Current Velocity (Uc)", DEF_CURRENT)
    ct = "Eulerian"
    dia = get_input("Pile Diameter", DEF_PILE_DIAMETER)
    mg = get_input("Marine Growth", DEF_MARINE_GROWTH)
    
    # Interactive Menu
    cm, cd, src = get_morison_coefficients(mg)
    
    # Solver
    print("\nRunning Solver...", end="\n")
    wave = FentonWave(h, t, d, uc, ct)
    print(" Done.")
    
    # 2. Start Logging (Only the Executive Summary goes to output.txt)
    with DualWriter("output.txt"): 
        
        # Safety Checks
        if not wave.converged or wave.k <= 1e-9:
            print("\nCRITICAL ERROR: Solver failed to converge.")
            print("Skipping kinematics calculation to prevent crash.")
            
            kc = 0.0; re_num = 0.0; deff = dia + 2*mg
            res = {
                'F': 0.0, 'M': 0.0, 'Ph': 0.0, 
                'Fd': 0.0, 'Fi': 0.0, 'FSlam': 0.0,
                'Breaking': False, 'Pr': [], 'MF': 0.0, 'MZ': 0.0,
                'Max_M_Abs': 0.0
            }
        else:
            # Post-Processing
            deff = dia + 2*mg
            eta_crest_abs = wave.get_eta_at_x(0) 
            u_c, _, _, _, _ = wave.get_kinematics_at_y(eta_crest_abs, 0) 
            u_orbital = u_c - wave.prop_u1
            kc = u_orbital * t / deff
            re_num = u_c * deff / NU_SEAWATER
            
            res = scan_force(wave, dia, mg, cm, cd)
        
        # Report Generation (Logged)
        generate_report(wave, res, h, d, t, uc, ct, dia, mg, deff, kc, src, cd, cm, re_num)
        
    # 3. Plotting (Console Only - Not Logged to File)
    if wave.converged:
        generate_plots(wave, res, h, d, t, dia, mg, cm, cd)
        print("Done. Results saved to 'output.txt' and 'plots.pdf'. Report generated.")
    else:
        print("Plots skipped due to solver failure.")

if __name__ == "__main__":
    main()