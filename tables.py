"""
==============================================================================
FENTON STREAM FUNCTION SOLVER (High-Performance/Optimized)
==============================================================================

DESCRIPTION:
  This script calculates the Wavelength (L) of nonlinear water waves using 
  Fenton's Stream Function theory. It generates lookup tables for various 
  combinations of:
    - Current (Uc)
    - Water Depth (d)
    - Wave Period (T)
    - Wave Height (H)

  It employs three major optimizations to achieve ~100x speedup over standard approaches:
  1. Multiprocessing: Distributes work across all available CPU cores.
  2. Numba (JIT): Compiles heavy matrix/trigonometric functions to machine code.
  3. 2D Hot-Starting: Uses the solution from the previous Period (T) or 
     Height (H) as a highly accurate initial guess for the next calculation, 
     drastically reducing the number of solver iterations required.

  ROBUSTNESS FEATURES:
  - Adaptive Homotopy: If the solver fails to converge immediately, it 
    automatically retries by ramping up the wave height in small increments 
    (linear -> nonlinear evolution) to guide the solution.

PREREQUISITES:
  pip install numpy scipy numba

USAGE:
  python tables.py
==============================================================================
"""

import numpy as np
from scipy.optimize import least_squares
import warnings
import time
import multiprocessing
import sys
from numba import njit
import shutil
import os

# ==============================================================================
#  USER CONFIGURATION
# ==============================================================================

# 1. Currents [m/s]
#    Positive values follow wave direction, negative oppose it.
UC_VALUES = [0.0, 0.5, 1.0, -0.5, -1.0]

# 2. Wave Heights H [m]
H_RANGE = [1.0, 2.5, 5.0, 7.5, 10.0, 12.5]

# 3. Wave Periods T [s]
T_RANGE = list(range(1, 22, 2))

# 4. Water Depths d [m]
D_RANGE = list(range(5, 51, 5))

# Output Filename
OUTPUT_FILE = "output.txt"

# Physical & Numerical Constants
G_STD = 9.8066
N_FOURIER = 20  # Number of Fourier components.

# ==============================================================================
#  JIT COMPILED MATH CORE (NUMBA)
# ==============================================================================

@njit(cache=True, fastmath=True)
def _calc_basis_numba(k, d, N, z_vals):
    """
    Computes the hyperbolic basis matrices (Sinh and Cosh terms) efficiently.
    
    Parameters:
      k      : Wavenumber (2*pi/L)
      d      : Water depth
      N      : Number of Fourier components
      z_vals : Array of surface elevations (eta)
      
    Optimization:
      - Uses fastmath for vectorization.
      - Includes a check for 'Deep Water' conditions (kd > 25) to prevent 
        floating-point overflow by using exponential approximations.
    """
    M = len(z_vals)
    S = np.zeros((N, M))
    C = np.zeros((N, M))
    kd = k * d
    
    for j in range(1, N + 1):
        idx = j - 1
        arg_check = j * kd
        
        # Deep water optimization: If kd is large, cosh/sinh explode.
        # We simplify the terms relative to the bottom boundary.
        if arg_check > 25.0:
            for m in range(M):
                val = np.exp(j * k * (z_vals[m] - d))
                S[idx, m] = val
                C[idx, m] = val
        else:
            denom = np.cosh(j * kd)
            inv_denom = 1.0 / denom # Pre-calculate division
            for m in range(M):
                arg = j * k * z_vals[m]
                S[idx, m] = np.sinh(arg) * inv_denom
                C[idx, m] = np.cosh(arg) * inv_denom
    return S, C

@njit(cache=True, fastmath=True)
def _residuals_numba(x, H_curr, T_target, d, Uc, g, N):
    """
    Computes the residual vector (error) for the nonlinear system.
    This function is called thousands of times, so it is compiled to machine code.
    
    The vector 'x' contains:
      - k (wavenumber)
      - etas (surface elevations at nodes)
      - Bs (Fourier coefficients)
      - Q (Bernoulli constant)
      - R (Bernoulli constant)
    """
    # --- 1. Unpack Optimization Vector ---
    k = x[0]
    if k <= 1e-8: k = 1e-8 # Prevent division by zero
    
    etas = x[1 : N+2]
    Bs   = x[N+2 : 2*N+2]
    Q    = x[-2]
    R    = x[-1]

    # --- 2. Wave Celerity Setup ---
    c = (2 * np.pi) / (k * T_target)
    U_frame = c - Uc  # Velocity in the moving frame of reference
    
    # --- 3. Compute Basis Functions ---
    S_mat, C_mat = _calc_basis_numba(k, d, N, etas)
    
    # --- 4. Fourier Summation (Perturbation Velocities) ---
    psi_pert = np.zeros_like(etas)
    u_pert   = np.zeros_like(etas)
    v_pert   = np.zeros_like(etas)
    
    # Equispaced nodes over half a wavelength (symmetry)
    x_nds = np.linspace(0, np.pi/k, N+1)
    
    # Scaling factor for non-dimensionalization
    sc = np.sqrt(g / k**3)
    
    for i in range(len(etas)):
        phase = k * x_nds[i]
        sum_psi = 0.0
        sum_u   = 0.0
        sum_v   = 0.0
        
        for j in range(1, N + 1):
            idx = j - 1
            cos_t = np.cos(j * phase)
            sin_t = np.sin(j * phase)
            B_val = Bs[idx]
            
            # Stream function and velocity components
            sum_psi += B_val * S_mat[idx, i] * cos_t
            sum_u   += B_val * (j * k) * C_mat[idx, i] * cos_t
            sum_v   += B_val * (j * k) * S_mat[idx, i] * (-sin_t)
            
        psi_pert[i] = sum_psi * sc
        u_pert[i]   = sum_u * sc
        v_pert[i]   = sum_v * sc

    # --- 5. Calculate Residuals (Errors) ---
    
    # Kinematic Boundary Condition (Surface is a streamline)
    res_kin = (-U_frame * etas + psi_pert + Q) / (np.sqrt(g * d) * d)
    
    # Dynamic Boundary Condition (Bernoulli equation constant on surface)
    u_tot = U_frame - u_pert
    bern = 0.5 * (u_tot**2 + v_pert**2) + g * etas
    res_dyn = (bern - R) / (g * d)
    
    # Wave Height Definition Error
    res_h = (etas[0] - etas[-1] - H_curr) / d
    
    # Mean Water Level Error (Must integrate to depth d)
    sum_eta = np.sum(etas) - 0.5*etas[0] - 0.5*etas[-1]
    mean_eta = sum_eta / N
    res_lvl = (mean_eta - d) / d
    
    # --- 6. Pack Output ---
    out = np.empty(3 + 2 * len(etas))
    out[0] = 0.0 # Dummy for alignment if needed
    out[1] = res_h
    out[2] = res_lvl
    out[3 : 3+len(etas)] = res_kin
    out[3+len(etas) :]   = res_dyn
    
    return out

# ==============================================================================
#  WORKER LOGIC
# ==============================================================================

def solve_case(H, T, d, Uc, guess_vector=None):
    """
    Solves for a specific wave case (H, T, d, Uc).
    
    Strategy:
    1. If a 'guess_vector' (Hot-Start) is provided, attempt to solve immediately.
    2. If no guess is provided, generate a Linear Theory (Airy Wave) guess.
    3. If convergence fails, switch to 'Homotopy' (ramping H from small to large).
    
    Returns:
        (Wavelength, SolutionVector, StatusString)
    """
    g = G_STD
    N = N_FOURIER

    # --- 1. Linear Theory Initialization (Fallback / First Run) ---
    def get_linear_guess():
        # Iterate to solve dispersion relation for k
        L0 = (g * T**2) / (2 * np.pi)
        if (d / L0) < 0.05: 
            k0 = 2*np.pi / (T * np.sqrt(g * d)) # Shallow approx
        else: 
            k0 = 2*np.pi / L0
            
        u_doppler = Uc 
        # Newton-Raphson for current interaction
        for _ in range(10): 
            sig = 2*np.pi/T - k0*u_doppler
            if sig <= 0: sig = 1e-5 
            k0 = 0.5*k0 + 0.5*(sig**2 / (g * np.tanh(k0 * d)))
            
        x_nds = np.linspace(0, np.pi/k0, N+1)
        eta_i = d + (0.01/2)*np.cos(k0*x_nds) # Tiny amplitude start
        B_i = np.zeros(N)
        Q_i = (2*np.pi/k0/T - Uc)*d
        R_i = 0.5*(Q_i/d)**2 + g*d
        return np.concatenate(([k0], eta_i, B_i, [Q_i, R_i]))

    if guess_vector is not None:
        x_start = guess_vector
    else:
        x_start = get_linear_guess()

    # --- 2. Solve Strategy ---
    
    # Strategy A: Fast Path
    # If hot-starting, we assume we are close to the solution.
    # We take 2 steps to verify convergence.
    steps_fast = [H] if guess_vector is not None else [H*0.5, H]
    
    x_curr = x_start.copy()
    success = False
    
    try:
        for h_step in steps_fast:
            res = least_squares(_residuals_numba, x_curr, 
                                args=(h_step, T, d, Uc, g, N), 
                                method='lm', max_nfev=600, ftol=1e-8)
            x_curr = res.x
            
        if res.success or res.cost < 1e-5:
            success = True
    except:
        success = False

    # Strategy B: Robust Path (Homotopy)
    # If Fast Path failed, we reset to Linear Theory and ramp up H slowly
    # (10%, 20% ... 100%) to stay within the basin of convergence.
    if not success:
        x_curr = get_linear_guess() 
        steps_robust = np.linspace(H/10, H, 10) 
        
        try:
            for h_step in steps_robust:
                res = least_squares(_residuals_numba, x_curr, 
                                    args=(h_step, T, d, Uc, g, N), 
                                    method='lm', max_nfev=800, ftol=1e-8)
                x_curr = res.x
            
            if res.success or res.cost < 1e-4:
                success = True
        except:
            success = False

    # --- 3. Final Validation ---
    if not success:
        return 0.0, None, "FAIL"

    k_final = x_curr[0]
    L = 2 * np.pi / k_final
    
    # Physics Check: Miche Breaking Limit
    # H_break approx 0.142 * L * tanh(kd)
    breaking_limit = 0.142 * L * np.tanh(k_final * d)
    
    if H > breaking_limit:
        return L, x_curr, "BREAK"
    
    return L, x_curr, f"{L:.3f}"


def process_depth_block(args):
    """
    Worker function executed by Multiprocessing pool.
    Calculates the grid of Wave Periods (rows) vs Wave Heights (cols)
    for a specific Current (Uc) and Depth (d).
    """
    uc, d, h_range, t_range, block_id, total_blocks = args
    
    output_lines = []
    output_lines.append(f"\n  [ DEPTH d = {d} m ]")
    
    # --- Dynamic Table Formatting ---
    # Calculate exact width needed based on number of H columns
    col_width_per_h = 15
    base_width = 12
    table_line_len = base_width + (len(h_range) * col_width_per_h)
    
    separator_line = "  " + "-" * (table_line_len - 2) # Adjust for margin

    output_lines.append(separator_line)
    
    row_label = "T \\ H"
    header = f"  {row_label:<8} |" 
    
    for h in h_range:
        header += f" {f'H={h}m':^12} |"
    output_lines.append(header)
    output_lines.append(separator_line)

    # --- 2D Hot-Start Storage ---
    # Stores the solution vector of the previous row (Period)
    # to act as a guess for the current row.
    prev_period_solutions = [None] * len(h_range)

    for t in t_range:
        row_str = f"  {f'T={t}s':<8} |"
        last_h_solution = None 
        current_period_solutions = []

        for i_h, h in enumerate(h_range):
            guess = None
            
            # PRIORITY 1: Vertical Guess (Previous Period, same Height)
            if prev_period_solutions[i_h] is not None:
                guess = prev_period_solutions[i_h]
            # PRIORITY 2: Horizontal Guess (Same Period, previous Height)
            elif last_h_solution is not None:
                guess = last_h_solution
            # FALLBACK: Guess remains None -> solve_case generates Linear guess

            L, final_vec, status = solve_case(h, t, d, uc, guess_vector=guess)
            
            row_str += f" {status:^12} |"
            
            # Only propagate VALID solutions for hot-starting.
            # "BREAK" or "FAIL" states are bad guesses.
            is_valid = (status != "BREAK" and status != "FAIL")
            
            if is_valid and final_vec is not None:
                last_h_solution = final_vec
                current_period_solutions.append(final_vec)
            else:
                # Store None to force a Linear reset for the next T
                current_period_solutions.append(None)
                # Do NOT update last_h_solution, so the next H tries to find
                # a valid guess from further back if possible.
        
        prev_period_solutions = current_period_solutions
        output_lines.append(row_str)

    output_lines.append(separator_line)
    return (uc, d, "\n".join(output_lines))
    
# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def main():
    print(f"--- FENTON SOLVER: OPTIMIZED + ROBUST MODE ---")
    print(f"Target Output : {OUTPUT_FILE}")
    
    # Prepare Task List
    tasks = []
    idx = 0
    total_tasks = len(UC_VALUES) * len(D_RANGE)
    
    for uc in UC_VALUES:
        for d in D_RANGE:
            idx += 1
            tasks.append((uc, d, H_RANGE, T_RANGE, idx, total_tasks))

    cpu_count = multiprocessing.cpu_count()
    print(f"System Cores  : {cpu_count}")
    print(f"Total Blocks  : {total_tasks}")
    print(f"Starting calculation pool...")
    print("-" * 60)
    
    t0 = time.time()
    results = []
    
    # Execute with Progress Tracking
    with multiprocessing.Pool() as pool:
        for i, result in enumerate(pool.imap(process_depth_block, tasks)):
            uc_res, d_res, txt_res = result
            
            percent = ((i + 1) / total_tasks) * 100
            elapsed = time.time() - t0
            
            print(f"[{percent:5.1f}%] Computed Block {i+1}/{total_tasks}: "
                  f"Uc={uc_res:<4.1f} d={d_res:<2} "
                  f"(Elapsed: {elapsed:.1f}s)")
            
            results.append(result)

    t1 = time.time()
    print("-" * 60)
    print(f"Calculation finished in {t1-t0:.2f} seconds.")

    # Write Results to Disk
    print(f"Writing to disk...")
    with open(OUTPUT_FILE, "w") as f:
        current_uc = -999.9
        
        for uc, d, block_text in results:
            # Create a Major Header when Current (Uc) changes
            if uc != current_uc:
                current_uc = uc
                table_idx = UC_VALUES.index(uc) + 1
                f.write("=" * 90 + "\n")
                f.write(f"  TABLE {table_idx}: WAVELENGTH (L) [m]\n")
                f.write(f"  CURRENT (Uc) = {uc:.1f} m/s\n")
                f.write(f"  Legend: 'BREAK' = Wave Breaks, 'FAIL' = No Conv\n")
                f.write("=" * 90 + "\n")
            
            f.write(block_text)
            f.write("\n\n")

    print(f"Success. File '{OUTPUT_FILE}' is ready.")

    # Cleanup temporary Python cache files
    cache_dir = "__pycache__"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleaned up '{cache_dir}' directory.")   
    
if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()