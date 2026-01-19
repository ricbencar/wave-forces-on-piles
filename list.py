"""
==============================================================================
FENTON STREAM FUNCTION SOLVER
==============================================================================

DESCRIPTION:
  Calculates nonlinear water wavelength (L) using Fenton's Stream Function.
  This script performs heavy numerical solving for a range of wave parameters
  and outputs the results to a tab-separated text file.

  FEATURES:
  - Parallel Processing: Maximizes usage of all available CPU cores.
  - Stability: Handles numerical singularities and non-physical input automatically.
  - Filters: Strictly adheres to Miche breaking criterion and H/d limits.

CONFIGURATION:
  - Current (Uc): -2.5 to 2.5 m/s (step 0.5)
  - Height (H):   1.0 to 15.0 m   (step 2.0)
  - Period (T):   1.0 to 21.0 s   (step 1.0)
  - Depth (d):    5.0 to 50.0 m   (step 5.0)

OUTPUT:
  - list.txt: Tab-separated columns (H, T, d, Uc, L)

USAGE:
  python list.py
==============================================================================
"""

import os
import multiprocessing
import warnings
import time
import shutil
import sys

# --- ENVIRONMENT CONFIGURATION ---
# Optimize for high-throughput parallel processing by disabling 
# internal linear algebra threading to avoid core oversubscription.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import least_squares
from numba import njit

# Suppress runtime warnings for expected numerical bounds checks
warnings.filterwarnings("ignore")

# ==============================================================================
#  GLOBAL CONFIGURATION
# ==============================================================================

# 1. Current Velocity [m/s]
UC_VALUES = np.arange(-2.5, 3.0, 0.5).tolist()

# 2. Wave Heights [m]
H_RANGE = np.arange(1, 15, 2).tolist()

# 3. Wave Periods [s]
T_RANGE = np.arange(1.0, 21.0, 1.0).tolist()

# 4. Water Depths [m]
D_RANGE = np.arange(5, 55, 5).tolist()

# System Constants
OUTPUT_FILE = "list.txt"
G_STD = 9.8066      # Gravity (m/s^2)
N_FOURIER = 50      # Stream function order

# ==============================================================================
#  NUMERICAL CORE (JIT COMPILED)
# ==============================================================================

@njit(cache=True, fastmath=True)
def _fast_basis(k, d, N, z_vals):
    """
    JIT-compiled calculation of Sinh/Cosh basis matrices.
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

@njit(cache=True, fastmath=True)
def _fast_residuals(x, H_curr, T_target, d, Uc, g, N):
    """
    JIT-compiled residual calculation.
    """
    # Unpack Vector
    k = x[0]
    etas = x[1 : N+2]
    Bs = x[N+2 : 2*N+2]
    Q = x[-2]
    R = x[-1]

    if k <= 1e-8: k = 1e-8
    c = (2 * np.pi) / (k * T_target)

    # Determine frame velocity (Eulerian assumed)
    U_frame = c - Uc

    # Calculate Basis matrices
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
        
    return residuals

# ==============================================================================
#  SOLVER LOGIC (FENTONWAVE CLASS)
# ==============================================================================

def solve_case(H, T, d, Uc, guess_vector=None):
    """
    Solves a single wave case (H, T, d, Uc) using Homotopy strategy.
    """
    g = G_STD
    N = N_FOURIER

    # --- 0. Validity Checks ---
    # Enforce strict H/d limit
    if (H / d) > 0.6:
        return 0.0, None, "LIMIT_HD"
    
    # Breaking Limit Check (Miche Criterion approx) - Pre-check to save time
    try:
        L_linear_approx = (g * T**2) / (2 * np.pi) * np.tanh(2 * np.pi * d / ((g * T**2) / (2 * np.pi)))
        if H > 0.142 * L_linear_approx * np.tanh(2 * np.pi * d / L_linear_approx) * 1.5: 
            return 0.0, None, "BREAK_PRE"
    except:
        return 0.0, None, "BREAK_PRE"

    # --- 1. Initialization: Linear Wave Theory (Airy) ---
    def get_linear_guess():
        # Using errstate to silently handle overflows for impossible wave cases
        with np.errstate(all='ignore'):
            L0 = (g * T**2) / (2 * np.pi)
            
            if (d / L0) < 0.05:
                k0 = 2*np.pi / (T * np.sqrt(g * d))
            else:
                k0 = 2*np.pi / L0

            # Doppler shift iteration
            u_doppler = Uc 
            for _ in range(20):
                sig = 2*np.pi/T - k0*u_doppler
                val_tanh = np.tanh(k0 * d)
                if val_tanh < 1e-5: val_tanh = 1e-5
                
                k0_new = 0.5*k0 + 0.5*(sig**2/(g*val_tanh))
                
                if not np.isfinite(k0_new) or k0_new > 1000.0:
                    break
                k0 = k0_new
            
            if not np.isfinite(k0) or k0 <= 1e-8: k0 = 1e-8

            x_nds = np.linspace(0, np.pi/k0, N+1)
            eta_i = d + (0.01/2)*np.cos(k0*x_nds)
            B_i = np.zeros(N)
            Q_i = (2*np.pi/k0/T - Uc)*d
            R_i = 0.5*(Q_i/d)**2 + g*d
            
            return np.concatenate(([k0], eta_i, B_i, [Q_i, R_i]))

    if guess_vector is not None:
        x_curr = guess_vector
    else:
        x_curr = get_linear_guess()

    # --- 2. Adaptive Homotopy (Stepping) ---
    n_steps = 5
    h_start_log = 0.01
    steps = np.linspace(h_start_log, H, n_steps)
    
    method = 'trf'
    success = False
    
    try:
        for i, h_step in enumerate(steps):
            
            # Hybrid Solver Switching
            if i > 0.85 * n_steps: 
                method = 'lm'
            
            tol = 2.3e-16
            
            res = least_squares(_fast_residuals, x_curr, 
                                args=(h_step, T, d, Uc, g, N), 
                                method=method, 
                                tr_solver='exact',
                                ftol=tol, xtol=tol, gtol=tol, 
                                max_nfev=8000)
            
            x_curr = res.x
            if not res.success and res.cost > 1e-5:
                return 0.0, None, "FAIL_STEP"

        # --- 3. Final Polish ---
        res_final = least_squares(_fast_residuals, x_curr, 
                                  args=(H, T, d, Uc, g, N), 
                                  method='lm', 
                                  ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
                                  max_nfev=1000)
        x_curr = res_final.x
        err = np.mean(np.abs(res_final.fun))
        
        if err < 1e-5: 
            success = True
        else:
            success = False

    except Exception:
        success = False

    if not success:
        return 0.0, None, "FAIL_CONV"

    # --- 4. Validation & Post-Calc ---
    k_final = x_curr[0]
    
    if np.isnan(k_final) or k_final <= 1e-8:
         return 0.0, None, "FAIL_NAN"

    L = 2 * np.pi / k_final
    
    # Filter: Physical Realism Cap
    if L > 10000.0 or L < 1.0:
        return 0.0, None, "FAIL_PHYS"
    
    # Filter: Wave Breaking (Miche Criterion)
    breaking_limit = 0.142 * L * np.tanh(k_final * d)
    if H > breaking_limit:
        return L, x_curr, "BREAK"
    
    return L, x_curr, "OK"


def process_block(args):
    """Worker function for multiprocessing."""
    uc, d, h_range, t_range, block_id, total_blocks = args
    valid_results = []
    
    # "Hot start" cache for optimization speed
    prev_period_solutions = [None] * len(h_range)

    for t in t_range:
        last_h_solution = None 
        current_period_solutions = []

        for i_h, h in enumerate(h_range):
            # Determine best starting guess
            guess = None
            if last_h_solution is not None:
                guess = last_h_solution
            elif prev_period_solutions[i_h] is not None:
                guess = prev_period_solutions[i_h]

            L, final_vec, status = solve_case(h, t, d, uc, guess_vector=guess)
            
            if status == "OK" and final_vec is not None:
                valid_results.append((h, t, d, uc, L))
                last_h_solution = final_vec
                current_period_solutions.append(final_vec)
            else:
                last_h_solution = None
                current_period_solutions.append(None)
        
        prev_period_solutions = current_period_solutions

    return (uc, d, valid_results)

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def main():
    print(f"--- FENTON STREAM FUNCTION SOLVER ---")
    
    if os.path.exists(OUTPUT_FILE):
        print(f"WARNING: Output file '{OUTPUT_FILE}' already exists.")
        print("Rename or delete it to run a new calculation.")
        return

    print(f"Target Output : {OUTPUT_FILE}")
    print(f"Configuration : Uc({len(UC_VALUES)}) x D({len(D_RANGE)}) x T({len(T_RANGE)}) x H({len(H_RANGE)})")
    
    # Generate Task Queue
    tasks = []
    idx = 0
    total_tasks = len(UC_VALUES) * len(D_RANGE)
    
    for uc in UC_VALUES:
        for d in D_RANGE:
            idx += 1
            tasks.append((uc, d, H_RANGE, T_RANGE, idx, total_tasks))

    # Explicitly grab CPU count
    cpu_count = multiprocessing.cpu_count()
    
    print(f"System Cores  : {cpu_count}")
    print(f"Task Blocks   : {total_tasks}")
    print(f"Strategy      : {cpu_count} Processes x 1 Thread/Process")
    print(f"Starting calculation pool...")
    print("-" * 60)
    
    t0 = time.time()
    all_data_points = []
    
    # Run Parallel Execution with explicit core count
    with multiprocessing.Pool(processes=cpu_count) as pool:
        for i, result in enumerate(pool.imap(process_block, tasks)):
            uc_res, d_res, points = result
            
            percent = ((i + 1) / total_tasks) * 100
            elapsed = time.time() - t0
            
            print(f"[{percent:5.1f}%] Block {i+1}/{total_tasks}: "
                  f"Uc={uc_res:<4.1f} d={d_res:<4.1f} "
                  f"-> {len(points)} valid waves "
                  f"(Time: {elapsed:.1f}s)")
            
            all_data_points.extend(points)

    t1 = time.time()
    print("-" * 60)
    print(f"Calculation finished in {t1-t0:.2f} seconds.")
    print(f"Total valid data points: {len(all_data_points)}")

    # Write Results
    print(f"Writing to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, "w") as f:
            f.write("H\tT\td\tUc\tL\n")
            for row in all_data_points:
                h, t, d, uc_val, l_val = row
                f.write(f"{h:.2f}\t{t:.2f}\t{d:.2f}\t{uc_val:.2f}\t{l_val:.4f}\n")
        print(f"Success.")
    except IOError as e:
        print(f"Error writing file: {e}")

    # Cleanup
    cache_dir = "__pycache__"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()