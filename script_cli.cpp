/**
 * ==============================================================================
 * HIGH-PRECISION WAVE HYDRODYNAMICS & STRUCTURAL IMPACT SOLVER (C++ PORT)
 * ==============================================================================
 * MODULE:   script_cli.cpp
 * TYPE:     Nonlinear BVP Solver & Transient Load Calculator
 * METHOD:   Fenton's Fourier Approximation
 * LICENSE:  MIT / Academic Open Source
 * ==============================================================================
 *
 * PROGRAM DESCRIPTION:
 * This software calculates the hydrodynamics (kinematics and dynamics) and 
 * structural loading of steady, finite-amplitude water waves acting on a 
 * vertical cylindrical pile.
 *
 * It implements the "Fourier Approximation Method" for the Nonlinear Stream 
 * Function as developed by J.D. Fenton (1988). Unlike Linear (Airy) theory 
 * or Stokes 5th Order approximations, this numerical method satisfies the 
 * full nonlinear boundary conditions to machine precision (limited only by the 
 * truncation order N).
 *
 * LIMITATIONS:
 * - Restricted to H/d <= 0.6.
 * - Impulsive slamming loads (breaking waves) are NOT calculated.
 *
 * ==============================================================================
 * THEORETICAL MANUAL & NUMERICAL DOCUMENTATION
 * ==============================================================================
 *
 * 1. INTRODUCTION & PHYSICAL SCOPE
 * -----------------------------------------------------------------------------
 * This solver addresses the Boundary Value Problem (BVP) of nonlinear water 
 * waves propagating over a horizontal bed and their interaction with vertical 
 * cylindrical structures. Unlike Linear Wave Theory (Airy), which assumes 
 * infinitesimal wave amplitude (H << d), or Stokes Expansion methods which 
 * diverge in shallow water, the Fourier Approximation Method (Stream Function 
 * Theory) is a numerical method accurate to machine precision for:
 * a. Waves within the H/d <= 0.6 limit.
 * b. Any water depth (Shallow, Intermediate, Deep).
 * c. Nonlinear wave-current interaction (Doppler shifting).
 *
 * The kinematic field calculated here drives the Quasi-Static Loading model:
 * - Morison Equation (Drag + Inertia).
 *
 * 2. MATHEMATICAL FORMULATION: FENTON'S STREAM FUNCTION THEORY
 * -----------------------------------------------------------------------------
 * The method solves for the Stream Function Psi(x,z) in a 2D domain.
 *
 * 2.1 GOVERNING FIELD EQUATIONS
 * Assumption: Fluid is inviscid, incompressible, and irrotational.
 * - Incompressibility: div(u) = 0
 * - Irrotationality:   curl(u) = 0
 * Consequently, a Scalar Potential (Phi) and Stream Function (Psi) exist.
 * The governing equation is the Laplace Equation:
 * * ∇²Psi = (d²Psi/dx²) + (d²Psi/dz²) = 0
 *
 * Velocity relationships: u = dPsi/dz, w = -dPsi/dx.
 *
 * 2.2 COORDINATE SYSTEM & REFERENCE FRAME
 * We utilize a coordinate system moving with the steady wave celerity (C).
 * - (x, z): Stationary frame coordinates (Seabed at z=0).
 * - (X, z): Moving frame coordinates, where X = x - C*t.
 * * In this moving frame, the wave profile is stationary in time, reducing 
 * the problem from unsteady (t-dependent) to steady state.
 *
 * 2.3 THE FOURIER ANSATZ (Analytical Solution Structure)
 * Fenton (1988) defines a truncated Fourier series of order N that 
 * automatically satisfies the Laplace equation and the Bottom Boundary 
 * Condition (w=0 at z=0).
 *
 * Psi(X, z) = -U_bar * z + SUM_{j=1}^{N} [ B_j * sinh(j*k*z)/cosh(j*k*d) * cos(j*k*X) ]
 *
 * Where:
 * - U_bar: Mean fluid speed in the moving frame (related to mass flux).
 * - k:     Wavenumber (2*pi/L).
 * - B_j:   Dimensionless Fourier coefficients (The unknowns).
 * - d:     Water depth.
 *
 * 2.4 BOUNDARY CONDITIONS (The Constraint System)
 * The coefficients B_j must be solved such that they satisfy conditions 
 * at the free surface z = eta(X).
 *
 * A. Kinematic Boundary Condition (KBC):
 * The free surface is a streamline. No flow crosses the interface.
 * Psi(X, eta(X)) = -Q
 * Where Q is the constant volume flux per unit width in the moving frame.
 *
 * B. Dynamic Boundary Condition (DBC - Bernoulli):
 * Pressure is constant (atmospheric) along the free surface.
 * 0.5 * [ (dPsi/dX)^2 + (dPsi/dz)^2 ] + g * eta(X) = R
 * Where R is the Bernoulli constant (Total Energy Head).
 *
 * 3. NUMERICAL SOLVER ALGORITHM (C++ IMPLEMENTATION)
 * -----------------------------------------------------------------------------
 * The problem is recast as a nonlinear optimization problem.
 *
 * 3.1 SYSTEM OF EQUATIONS
 * We discretize the wave phase (0 to pi, due to symmetry) into M nodes.
 * The Unknowns Vector x (Size N + 3) contains:
 * - [B_1 ... B_N]: The stream function coefficients.
 * - k:             The Wavenumber.
 * - Q:             The Mass Flux.
 * - R:             The Bernoulli Constant.
 *
 * 3.2 OPTIMIZATION OBJECTIVE
 * We define a residual vector 'r' combining errors from KBC, DBC, and 
 * geometric definitions (Wave Height H and Mean Water Level d).
 * Minimization target: sum(r^2) -> 0.
 *
 * Solver: Custom Levenberg-Marquardt algorithm (Native C++).
 * Jacobian: Calculated numerically via Complex-Step Differentiation
 * for maximum precision without cancellation errors.
 *
 * 3.3 ADAPTIVE HOMOTOPY (continuation Method)
 * Nonlinear solvers fail if the initial guess is too far from the solution.
 * This script implements a homotopy wrapper:
 * 1. Initialize with Linear Wave Theory (Airy) for H ~ 0.
 * 2. Step H from 0 -> Target_H in 'n' increments (Linear steps).
 * 3. Use the solution of step i as the initial guess for step i+1.
 *
 * 4. FORCE CALCULATION: THE MORISON EQUATION
 * -----------------------------------------------------------------------------
 * Applied when the structure is hydrodynamically transparent (D/L < 0.2).
 * The total force is the superposition of Drag (viscous) and Inertia (mass).
 *
 * dF(z,t) = dF_Drag + dF_Inertia
 *
 * 4.1 DRAG COMPONENT (Velocity Dependent)
 * dF_D = 0.5 * rho * Cd * D * (u + Uc) * |u + Uc| * dz
 * - Nonlinear dependence on u|u|.
 * - Includes steady current Uc in the velocity vector.
 *
 * 4.2 INERTIA COMPONENT (Acceleration Dependent)
 * dF_I = rho * Cm * (pi * D^2 / 4) * du/dt * dz
 * - Dominated by fluid acceleration field (du/dt).
 *
 * ==============================================================================
 * DEPENDENCIES & COMPILATION
 * ==============================================================================
 * PREREQUISITES:
 * - C++17 Compiler (GCC, Clang, or MSVC).
 * - Standard Template Library (STL).
 *
 * COMPILATION INSTRUCTION:
 * $ g++ -O3 -march=native -std=c++17 -Wall -Wextra -static -static-libgcc -static-libstdc++ -o script_cli.exe script_cli.cpp -lm
 *
 * RUNNING INSTRUCTIONS:
 * 1. Interactive Mode: Run `script.exe` and follow prompts.
 * 2. CLI Mode: `script_cli.exe [H] [T] [d] [Uc] [Type] [Dia] [Mg] [Cd] [Cm]`
 *
 * C++ IMPLEMENTATION DETAILS & OPTIMIZATIONS:
 * -----------------------------------------------------------------------------
 * 1. C++ Standard Library (STL)
 * - Role: Core Data Structures & Complex Arithmetic.
 * - Usage: `std::vector` & `std::complex` replace NumPy arrays for dynamic 
 * storage and complex-step differentiation.
 *
 * 2. Custom Numerical Engines (Replacing SciPy)
 * - `FastLinearSolver`: A custom implementation of Crout's LU Decomposition
 * optimized for flattened 1D memory layouts with Iterative Refinement.
 * - `Levenberg-Marquardt`: A native C++ implementation of the TRF/LM 
 * algorithm to solve the nonlinear system of N+3 equations.
 * - `Golden Section Search`: Used to locate the exact phase angle of peak force.
 *
 * ==============================================================================
 * BIBLIOGRAPHY & REFERENCES
 * ==============================================================================
 *
 * *** THEORETICAL BASIS (FENTON: STREAM FUNCTION, KINEMATICS & NUMERICAL METHODS) ***
 * 1.  Fenton, J.D. (1999). "Numerical methods for nonlinear waves." 
 * In P.L.-F. Liu (Ed.), Advances in Coastal and Ocean Engineering (Vol. 5, 
 * pp. 241–324). World Scientific: Singapore.
 * [Primary Source: Comprehensive review of fully-nonlinear methods including 
 * Fourier approximation, Boundary Integral Equation (BIE) methods, and 
 * Local Polynomial Approximation].
 * URL: https://johndfenton.com/Papers/Fenton99Liu-Numerical-methods-for-nonlinear-waves.pdf
 *
 * 2.  Fenton, J.D. (1988). "The numerical solution of steady water wave problems."
 * Computers & Geosciences, 14(3), 357–368.
 * [The core algorithm for high-accuracy Stream Function Theory].
 * URL: https://doi.org/10.1016/0098-3004(88)90066-0
 *
 * 3.  Fenton, J.D. (1985). "A fifth-order Stokes theory for steady waves."
 * Journal of Waterway, Port, Coastal, and Ocean Engineering, 111(2), 216–234.
 * [Standard analytical theory for deep/intermediate water pile design].
 * URL: https://doi.org/10.1061/(ASCE)0733-950X(1985)111:2(216)
 *
 * 4.  Fenton, J.D. (1978). "Wave forces on vertical bodies of revolution."
 * Journal of Fluid Mechanics, 85(2), 241–255.
 * [Foundational diffraction theory for large diameter piles].
 * URL: https://johndfenton.com/Papers/Fenton78-Waves-on-bodies-of-revolution.pdf
 *
 * 5.  Fenton, J.D. (1990). "Nonlinear wave theories." In B. Le Méhauté & 
 * D.M. Hanes (Eds.), The Sea: Ocean Engineering Science (Vol. 9, Part A).
 * John Wiley & Sons.
 * [Comprehensive guide for selecting wave theories: Stokes vs Cnoidal vs Stream].
 * URL: https://www.johndfenton.com/Papers/Fenton90b-Nonlinear-wave-theories.pdf
 *
 * *** HOCINE OUMERACI (BREAKING WAVE IMPACT, SLAMMING & RINGING) ***
 * 6.  Wienke, J., & Oumeraci, H. (2005). "Breaking wave impact force on a vertical 
 * and inclined slender pile—theoretical and large-scale model investigations."
 * Coastal Engineering, 52(5), 435–462.
 * [CRITICAL: Separates quasi-static (Morison) from dynamic (slamming) forces].
 * URL: https://doi.org/10.1016/j.coastaleng.2004.12.008
 *
 * 7.  Irschik, K., Sparboom, U., & Oumeraci, H. (2004). "Breaking wave loads on a 
 * slender pile in shallow water." Proceedings of the 29th ICCE, 4, 3968–3980.
 * [Focuses on shallow water impacts where Stream Function may reach limits].
 * URL: https://www.worldscientific.com/doi/abs/10.1142/9789812701916_0045
 *
 * 8.  Kortenhaus, A., & Oumeraci, H. (1998). "Classification of wave loading on 
 * monolithic coastal structures." Proceedings of the 26th ICCE, 1, 867–879.
 * [Defines transition zones between pulsating and impulsive load regimes].
 * URL: https://icce-ojs-tamu.tdl.org/icce/article/download/5654/5324/0
 *
 * 9.  Muttray, M., & Oumeraci, H. (2005). "Theoretical and experimental study on 
 * wave damping inside a perforated caisson." Ocean Engineering, 32(14), 1803–1818.
 * [Relevant for piles with scour protection or permeable outer layers].
 * URL: https://www.sciencedirect.com/science/article/abs/pii/S0378383905000591
 *
 * *** ENGINEERING MANUALS & STANDARDS ***
 * 10. U.S. Army Corps of Engineers (USACE). (2002). "Coastal Engineering Manual 
 * (CEM)." Engineer Manual 1110-2-1100. Washington, D.C.
 * [The modern successor to the SPM; standard for wave mechanics].
 * URL: https://www.publications.usace.army.mil/USACE-Publications/Engineer-Manuals/u43544q/636F617374616C20656E67696E656572696E67206D616E75616C/
 *
 * 11. U.S. Army Corps of Engineers (USACE). (1984). "Shore Protection Manual 
 * (SPM)." Vol. I & II. 4th Edition. CERC, Vicksburg, MS.
 * [Classic reference; still widely used for historical comparison and empirical data].
 * URL: https://usace.contentdm.oclc.org/digital/collection/p16021coll11/id/1934/
 *
 * 12. CIRIA, CUR, CETMEF. (2007). "The Rock Manual. The Use of Rock in 
 * Hydraulic Engineering." (2nd Edition). C683, CIRIA, London.
 * [Standard for pile scour protection design and rock interaction].
 * URL: https://www.ciria.org/ItemDetail?iProductCode=C683
 *
 * 13. DNV (Det Norske Veritas). (2014). "Environmental Conditions and Environmental 
 * Loads." Recommended Practice DNV-RP-C205.
 * [Industry standard for offshore pile design and Morison coefficients].
 * URL: https://www.dnv.com/energy/standards-guidelines/dnv-rp-c205-environmental-conditions-and-environmental-loads/
 *
 * *** TEXTBOOKS (WAVE MECHANICS & FORCES) ***
 * 14. Sumer, B. M., & Fredsøe, J. (2006). "Hydrodynamics Around Cylindrical 
 * Structures." (Revised Edition). World Scientific.
 * [The 'Bible' for flow around piles, vortex shedding, and scour].
 * URL: https://doi.org/10.1142/6248
 *
 * 15. Sarpkaya, T., & Isaacson, M. (1981). "Mechanics of Wave Forces on 
 * Offshore Structures." Van Nostrand Reinhold.
 * [Classic text on diffraction and inertia/drag regimes].
 * URL: https://www.amazon.com/-/pt/dp/0521896258/
 *
 * 16. Goda, Y. (2010). "Random Seas and Design of Maritime Structures." 
 * (3rd Edition). World Scientific.
 * [Essential for spectral analysis and statistical design of piles].
 * URL: https://doi.org/10.1142/7425
 *
 * 17. Dean, R. G., & Dalrymple, R. A. (1991). "Water Wave Mechanics for 
 * Engineers and Scientists." World Scientific.
 * [Foundational pedagogy for linear and nonlinear wave theory].
 * URL: https://doi.org/10.1142/1232
 *
 * ==============================================================================
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>
#include <complex>
#include <limits>

#ifdef __MINGW32__
#include <stdio.h>
// Fix for static linking with MinGW GCC 15+
extern "C" {
    // Redirect the linker looking for the DLL symbol to the static function
    int (*__imp_fseeko64)(FILE*, _off64_t, int) = &fseeko64;
    _off64_t (*__imp_ftello64)(FILE*) = &ftello64;
}
#endif

// ==============================================================================
//  SECTION 1: TYPE DEFINITIONS & CONSTANTS
// ==============================================================================

namespace Core {
    // Use standard double (matches Python float64). 
    using Real    = double;
    using Integer = int;
    using String  = std::string;
    using Complex = std::complex<Real>;
    using Vector  = std::vector<Real>;
    using VectorC = std::vector<Complex>;
    using Matrix  = std::vector<std::vector<Real>>;
}

using namespace Core;

namespace Defaults {
    constexpr Real   WAVE_HEIGHT    = 3.0000;
    constexpr Real   WAVE_PERIOD    = 9.0000;
    constexpr Real   DEPTH          = 5.0000;
    constexpr Real   CURRENT        = 1.0000;
    const     String CURRENT_TYPE   = "Eulerian";
    constexpr Real   PILE_DIAMETER  = 1.5000;
    constexpr Real   MARINE_GROWTH  = 0.0500;
    constexpr int    SOLVER_ORDER   = 50;
    constexpr int    HOMOTOPY_STEPS = 5;
}

namespace Phys {
    constexpr Real PI          = 3.14159265358979323846;
    constexpr Real RHO         = 1025.0;       
    constexpr Real G_STD       = 9.8066;     
    constexpr Real NU_SEAWATER = 1.05e-6; 
}

// ==============================================================================
//  SECTION 2: MATH & LINEAR ALGEBRA UTILITIES
// ==============================================================================

namespace MathUtils {

    // Vector Operations
    inline Vector operator+(const Vector& a, const Vector& b) {
        Vector r(a.size()); 
        for(size_t i=0; i<a.size(); ++i) r[i] = a[i] + b[i]; 
        return r;
    }

    inline Vector operator-(const Vector& a, const Vector& b) {
        Vector r(a.size()); 
        for(size_t i=0; i<a.size(); ++i) r[i] = a[i] - b[i]; 
        return r;
    }

    inline Vector operator*(Real s, const Vector& a) {
        Vector r(a.size()); 
        for(size_t i=0; i<a.size(); ++i) r[i] = s * a[i]; 
        return r;
    }

    inline Real dot_product(const Vector& a, const Vector& b) {
        Real s = 0; 
        for(size_t i=0; i<a.size(); ++i) s += a[i] * b[i]; 
        return s;
    }

    // High-Performance Linear Solver
    // Uses Crout LU decomposition on a flattened matrix to maximize cache hits.
    class FastLinearSolver {
    public:
        // Core solver (Crout LU)
        static Vector solve_flat(const Matrix& A_2d, const Vector& b) {
            int n = b.size();
            std::vector<Real> LU(n * n);
            for(int i=0; i<n; ++i) 
                for(int j=0; j<n; ++j) 
                    LU[i*n + j] = A_2d[i][j];

            std::vector<int> p(n);
            std::iota(p.begin(), p.end(), 0);

            for (int i = 0; i < n; ++i) {
                int max_row = i;
                Real max_val = std::abs(LU[i*n + i]);
                for (int k = i + 1; k < n; ++k) {
                    Real val = std::abs(LU[k*n + i]);
                    if (val > max_val) {
                        max_val = val;
                        max_row = k;
                    }
                }
                std::swap(p[i], p[max_row]);
                for (int k = 0; k < n; ++k) std::swap(LU[i*n + k], LU[max_row*n + k]);

                Real pivot_val = LU[i*n + i];
                if (std::abs(pivot_val) < 1e-12) pivot_val = (pivot_val >= 0 ? 1e-12 : -1e-12);
                LU[i*n + i] = pivot_val;

                for (int k = i + 1; k < n; ++k) {
                    LU[k*n + i] /= pivot_val;
                    for (int j = i + 1; j < n; ++j) {
                        LU[k*n + j] -= LU[k*n + i] * LU[i*n + j];
                    }
                }
            }

            Vector x(n);
            for (int i = 0; i < n; ++i) {
                Real sum = 0;
                for (int j = 0; j < i; ++j) sum += LU[i*n + j] * x[j];
                x[i] = b[p[i]] - sum;
            }
            for (int i = n - 1; i >= 0; --i) {
                Real sum = 0;
                for (int j = i + 1; j < n; ++j) sum += LU[i*n + j] * x[j];
                x[i] = (x[i] - sum) / LU[i*n + i];
            }
            return x;
        }

        // IMPROVEMENT: Iterative Refinement
        // Solves Ax=b, calculates residual r=b-Ax, solves Ae=r, returns x+e.
        // This recovers precision lost during LU factorization, essential for 1e-16 accuracy.
        static Vector solve_refined(const Matrix& A, const Vector& b) {
            // 1. Initial Solve
            Vector x = solve_flat(A, b);
            
            // 2. Calculate Residual (r = b - A*x)
            // Note: In a perfect world, this would use higher precision, but double is usually sufficient 
            // to recover the last few bits of machine epsilon.
            int n = b.size();
            Vector r(n);
            for(int i=0; i<n; ++i) {
                Real Ax_i = 0.0;
                for(int j=0; j<n; ++j) Ax_i += A[i][j] * x[j];
                r[i] = b[i] - Ax_i;
            }

            // 3. Solve for Error Correction
            Vector correction = solve_flat(A, r);
            
            // 4. Apply Correction
            for(int i=0; i<n; ++i) x[i] += correction[i];
            
            return x;
        }
    };

    // Golden Section Search for finding maximum of a function f(x)
    // Used to precisely locate the phase of maximum total force.
    template <typename Func>
    Real golden_section_search(Func f, Real a, Real b, Real tol = 1e-9) {
        const Real phi = (1.0 + std::sqrt(5.0)) / 2.0;
        const Real resphi = 2.0 - phi;
        
        Real c = a + resphi * (b - a);
        Real d = b - resphi * (b - a);
        Real fc = f(c);
        Real fd = f(d);
        
        while (std::abs(c - d) > tol) {
            // We want to MAXIMIZE force magnitude |F|
            if (fc > fd) {
                b = d;
                d = c;
                fd = fc;
                c = a + resphi * (b - a);
                fc = f(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = b - resphi * (b - a);
                fd = f(d);
            }
        }
        return (c + d) / 2.0;
    }
}

using namespace MathUtils;

// ==============================================================================
//  SECTION 3: I/O UTILITIES & LOGGING
// ==============================================================================

class Logger {
    std::ofstream file;
public:
    explicit Logger(const String& filename) { file.open(filename); }
    ~Logger() { if(file.is_open()) file.close(); }
    
    template <typename T> 
    void print(const T& val) { 
        std::cout << val; 
        if(file.is_open()) file << val; 
    }
    
    template <typename T> 
    void println(const T& val) { 
        std::cout << val << "\n"; 
        if(file.is_open()) file << val << "\n"; 
    }
    
    void newline() { 
        std::cout << "\n"; 
        if(file.is_open()) file << "\n"; 
    }
    
    void print_data_row(const String& desc, Real val, const String& unit) {
        std::stringstream ss; 
        ss << std::fixed << std::setprecision(4) << val;
        print_data_str(desc, ss.str(), unit);
    }
    
    void print_data_str(const String& desc, const String& val, const String& unit) {
        std::stringstream line;
        line << std::left << std::setw(41) << desc << "| " 
             << std::setw(15) << val << " | " << unit;
        println(line.str());
    }

    void print_force_row(const String& desc, Real val, const String& unit) {
        std::stringstream ss; 
        ss << std::fixed << std::setprecision(4) << val;
        std::stringstream line;
        line << " " << std::left << std::setw(39) << desc << " | " 
             << std::setw(15) << ss.str() << " | " << unit;
        println(line.str());
    }
};

Real get_input(const String& prompt, Real def) {
    std::cout << std::left << std::setw(45) << prompt << " [" << def << "]: ";
    String line; std::getline(std::cin, line);
    if(line.empty()) return def;
    try { return std::stod(line); } catch(...) { return def; }
}

String get_input_str(const String& prompt, const String& def) {
    std::cout << std::left << std::setw(45) << prompt << " [" << def << "]: ";
    String line; std::getline(std::cin, line);
    return line.empty() ? def : line;
}

// ==============================================================================
//  SECTION 4: CORE SOLVER LOGIC (FENTON STREAM FUNCTION)
// ==============================================================================

class FentonWave {
public:
    // Core Parameters
    Real H_target, T, d, Uc;
    String current_type;
    int N;
    
    // Solution State
    Real k, L, c, Q, R;
    Vector Bj, eta_nodes, Ej;
    
    // Integral Properties
    Real prop_KE, prop_PE, prop_I, prop_Sxx, prop_F, prop_ub2;
    Real prop_q, prop_S, prop_u1, prop_u2, prop_U_frame, prop_r, prop_R;

    // Status Tracking
    bool converged;
    struct StepLog { String type; Real h; Real err; String status; };
    std::vector<StepLog> history;

private:
    // Optimization: Flattened tables for basis functions
    // Stores cos(j * m * pi / N) and sin(j * m * pi / N)
    std::vector<Real> table_cos;
    std::vector<Real> table_sin;

public:
    FentonWave(Real _H, Real _T, Real _d, Real _Uc, String _ct) 
        : H_target(_H), T(_T), d(_d), Uc(_Uc), current_type(_ct), N(Defaults::SOLVER_ORDER) {
        
        // -- CRITICAL LIMIT CHECK: H/d <= 0.6 --
        if (d > 0 && (_H / d) > 0.6) {
             std::cout << "\n[!] LIMIT EXCEEDED: H/d = " << std::fixed << std::setprecision(3) 
                       << (_H/d) << " > 0.6. Calculation aborted." << std::endl;
             converged = false;
             history.push_back({"LIMIT", _H, 0.0, "ABORT"});
             k = 0.0;
             return; 
        }

        converged = false;
        Bj.resize(N, 0.0); Ej.resize(N, 0.0); eta_nodes.resize(N + 1, d);
        precompute_basis();
    }

    // Precompute basis functions (Performance Boost)
    void precompute_basis() {
        table_cos.resize((N + 1) * N);
        table_sin.resize((N + 1) * N);

        Real pi_div_N = Phys::PI / (Real)N;

        for (int m = 0; m <= N; ++m) {
            for (int j = 1; j <= N; ++j) {
                Real arg = j * m * pi_div_N;
                size_t idx = m * N + (j - 1);
                table_cos[idx] = std::cos(arg);
                table_sin[idx] = std::sin(arg);
            }
        }
    }

    void solve() { 
        // Abort if limit check failed in constructor
        if (k == 0.0 && !history.empty() && history.back().status == "ABORT") return;
        
        solve_adaptive(); 
        if(converged) calculate_integral_properties(); 
    }

    Vector get_kinematics(Real z_bed, Real x) {
        // This function calculates kinematics at arbitrary (x,z)
        // z_bed is the coordinate where 0 = seabed
        Real kd = k * d;
        Real sc = std::sqrt(Phys::G_STD / std::pow(k, 3));
        Real u_p = 0, v_p = 0, dup_dx = 0, dup_dz = 0, dwp_dx = 0, dwp_dz = 0;
        
        for (int j = 1; j <= N; ++j) {
            Real arg = j * k * z_bed;
            Real ch = 0, sh = 0;
            Real jkd = j * kd;
            
            if (jkd > 20.0) {
                Real term = std::exp(j * k * (z_bed - d));
                ch = term; sh = term;
            } else {
                Real denom = std::cosh(jkd);
                ch = std::cosh(arg) / denom;
                sh = std::sinh(arg) / denom;
            }
            
            Real cx = std::cos(j * k * x);
            Real sx = std::sin(j * k * x);
            Real jk = j * k;
            Real B = Bj[j-1];
            
            u_p += B * jk * ch * cx;
            v_p += B * jk * sh * sx; 
            
            dup_dx += B * std::pow(jk, 2) * ch * (-sx);
            dup_dz += B * std::pow(jk, 2) * sh * cx;
            dwp_dx += B * std::pow(jk, 2) * sh * cx;
            dwp_dz += B * std::pow(jk, 2) * ch * sx;
        }
        
        u_p *= sc; v_p *= sc;
        dup_dx *= sc; dup_dz *= sc; dwp_dx *= sc; dwp_dz *= sc;
        
        Real u_fix = (c - prop_U_frame) + u_p;
        Real w_fix = v_p; 
        Real ax = (u_fix - c) * dup_dx + w_fix * dup_dz;
        Real az = (u_fix - c) * dwp_dx + w_fix * dwp_dz;
        
        Real u_w = prop_U_frame - u_p;
        Real p = Phys::RHO * (R - Phys::G_STD * z_bed - 0.5 * (u_w*u_w + v_p*v_p));
        
        return {u_fix, w_fix, ax, az, p};
    }
    
    Real get_eta(Real x) {
        Real y = d; 
        for(int iter=0; iter<50; ++iter) { 
             Real psi_p = 0, u_p = 0;
             Real sc = std::sqrt(Phys::G_STD/std::pow(k,3));
             
             for(int j=1; j<=N; ++j) {
                 Real term = (j*k*d > 20) ? std::exp(j*k*(y-d)) : std::sinh(j*k*y)/std::cosh(j*k*d);
                 Real term_c = (j*k*d > 20) ? std::exp(j*k*(y-d)) : std::cosh(j*k*y)/std::cosh(j*k*d);
                 psi_p += Bj[j-1] * term * std::cos(j*k*x);
                 u_p   += Bj[j-1] * (j*k) * term_c * std::cos(j*k*x);
             }
             psi_p *= sc; u_p *= sc;
             Real f = -prop_U_frame * y + psi_p + Q;
             Real df = -prop_U_frame + u_p;
             if (std::abs(df) < 1e-20) df = 1e-20; 
             Real dy = f/df;
             y -= dy;
             if(std::abs(dy) < 1e-15) break;
        }
        return y;
    }

private:
    Vector pack_state() {
        Vector x; x.reserve(1 + eta_nodes.size() + Bj.size() + 2);
        x.push_back(k);
        for(Real e : eta_nodes) x.push_back(e);
        for(Real b : Bj) x.push_back(b);
        x.push_back(Q); x.push_back(R);
        return x;
    }

    void unpack_state(const Vector& x) {
        int idx = 0; k = x[idx++];
        for(int i=0; i<=N; ++i) eta_nodes[i] = x[idx++];
        for(int i=0; i<N; ++i) Bj[i] = x[idx++];
        Q = x[idx++]; R = x[idx++];
    }

    // --- OPTIMIZED TEMPLATED RESIDUALS ---
    // Uses precomputed tables and reduced power calls
    template <typename Scalar>
    void residuals_internal_optimized(const std::vector<Scalar>& x_state, Real current_H, std::vector<Scalar>& res_out) {
        // Unpack state
        Scalar tk = x_state[0];
        const Scalar* t_eta = &x_state[1];
        const Scalar* t_B = &x_state[1 + N + 1];
        Scalar tQ = x_state[x_state.size()-2];
        Scalar tR = x_state[x_state.size()-1];

        // Constants cast to Scalar type
        Scalar t_g = (Scalar)Phys::G_STD; 
        Scalar t_d = (Scalar)d; 
        Scalar t_pi = (Scalar)Phys::PI;
        Scalar t_T = (Scalar)T; 
        Scalar t_Uc = (Scalar)Uc; 
        Scalar t_H = (Scalar)current_H;

        // Derived variables
        Scalar tc = (2.0 * t_pi) / (tk * t_T);
        Scalar tU_frame = (current_type == "Eulerian") ? (tc - t_Uc) : (tQ / t_d);
        
        // Optimize: tk^3 via multiplication
        Scalar tk2 = tk*tk; 
        Scalar tk3 = tk2*tk;
        Scalar sc = std::sqrt(t_g / tk3);
        Scalar inv_g_d = 1.0 / (t_g * t_d);
        Scalar inv_sqrt_gd_d = 1.0 / (std::sqrt(t_g * t_d) * t_d);

        // 1 Current + 2 Geom + 2*(N+1) Field Equations
        size_t expected_size = 3 + 2 * (N + 1);
        if (res_out.size() != expected_size) res_out.resize(expected_size);
        
        size_t r_idx = 0;

        // 1. Current Constraint
        if (current_type != "Eulerian") 
            res_out[r_idx++] = ((tc - tQ/t_d) - t_Uc) / std::sqrt(t_g*t_d);
        else 
            res_out[r_idx++] = Scalar(0.0);

        // 2. Geometric Constraints
        res_out[r_idx++] = (t_eta[0] - t_eta[N] - t_H) / t_d; 

        Scalar mean_eta = Scalar(0.0); 
        for(int i=0; i<=N; ++i) mean_eta += t_eta[i];
        mean_eta -= 0.5*(t_eta[0] + t_eta[N]);
        res_out[r_idx++] = (mean_eta/(Real)N - t_d)/t_d; 

        // 3. Field Equation Loop (The Hot Path)
        Scalar kd = tk * t_d;

        for (int m = 0; m <= N; ++m) {
            Scalar z_nd = t_eta[m];
            Scalar psi_p = 0, u_p = 0, v_p = 0;
            
            size_t table_offset = m * N;

            for (int j = 1; j <= N; ++j) {
                // Hyperbolic optimization
                Scalar j_scalar = (Real)j;
                Scalar arg = j_scalar * tk * z_nd;
                Scalar jkd = j_scalar * kd;
                
                Scalar S, C;
                // Use std::real to check magnitude for stability in complex steps
                if (std::real(jkd) > 20.0) {
                    Scalar term = std::exp(j_scalar * tk * (z_nd - t_d));
                    S = term; C = term;
                } else {
                    Scalar ch_denom = std::cosh(jkd);
                    Scalar inv_denom = 1.0 / ch_denom;
                    S = std::sinh(arg) * inv_denom;
                    C = std::cosh(arg) * inv_denom;
                }
                
                // Use precomputed Real trig tables
                Real cx_real = table_cos[table_offset + (j - 1)];
                Real sx_real = table_sin[table_offset + (j - 1)];
                
                Scalar B_val = t_B[j-1];
                psi_p += B_val * S * cx_real;
                
                Scalar B_jk = B_val * j_scalar * tk;
                u_p   += B_jk * C * cx_real;
                v_p   += B_jk * S * (-sx_real); 
            }
            
            psi_p *= sc; u_p *= sc; v_p *= sc;
            
            // Kinematic BC
            res_out[r_idx++] = (-tU_frame * z_nd + psi_p + tQ) * inv_sqrt_gd_d;
            
            // Dynamic BC (Bernoulli)
            Scalar u_tot = tU_frame - u_p;
            res_out[r_idx++] = (0.5 * (u_tot*u_tot + v_p*v_p) + t_g * z_nd - tR) * inv_g_d;
        }
    }
    
    // Wrapper for real-valued calls
    Vector residuals(const Vector& x, Real h) { 
        Vector res;
        residuals_internal_optimized<Real>(x, h, res);
        return res; 
    }

    // --- OPTIMIZED COMPLEX-STEP JACOBIAN ---
    // Uses single allocation and buffer reuse
    Matrix compute_jacobian_complex(const Vector& x_val, Real target_h) {
        int n = x_val.size();
        
        VectorC x_c(n);
        for(int i=0; i<n; ++i) x_c[i] = Complex(x_val[i], 0.0);
        
        Vector r_base = residuals(x_val, target_h);
        int m = r_base.size();
        
        Matrix J(m, Vector(n));
        Real h_step = 1.0e-20; 
        
        std::vector<Complex> r_c_buffer;
        r_c_buffer.reserve(m);

        for(int j=0; j<n; ++j) {
            x_c[j].imag(h_step);
            residuals_internal_optimized<Complex>(x_c, target_h, r_c_buffer);
            
            Real inv_h = 1.0 / h_step;
            for(int i=0; i<m; ++i) {
                J[i][j] = std::imag(r_c_buffer[i]) * inv_h;
            }
            x_c[j].imag(0.0);
        }
        return J;
    }

    // --- ROBUST SOLVER STRATEGY (Adaptive Levenberg-Marquardt) ---
    void levenberg_marquardt(Real target_h, Real tol = 1e-15, int max_iter = 200) {
        Vector x = pack_state();
        Real lambda = 1e-3;
        Real v = 2.0;       

        for (int iter = 0; iter < max_iter; ++iter) {
            Vector r = residuals(x, target_h);
            Real err_sq = dot_product(r, r); 
            
            if (std::sqrt(err_sq/r.size()) < tol) break;

            Matrix J = compute_jacobian_complex(x, target_h);
            
            int n_vars = x.size(); 
            int n_res = r.size();
            Matrix JtJ(n_vars, Vector(n_vars, 0.0));
            Vector Jtr(n_vars, 0.0);

            for(int i=0; i<n_res; ++i) {
                for(int j=0; j<n_vars; ++j) {
                    Jtr[j] -= J[i][j] * r[i]; 
                    for(int k=j; k<n_vars; ++k) {
                        Real val = J[i][j] * J[i][k];
                        JtJ[j][k] += val;
                        if(j!=k) JtJ[k][j] += val;
                    }
                }
            }

            Matrix A = JtJ;
            for(int i=0; i<n_vars; ++i) {
                A[i][i] += lambda * A[i][i]; 
                if(A[i][i] < 1e-12) A[i][i] = 1e-12; 
            }

            // Calls the FastLinearSolver with Iterative Refinement
            // Crucial step for achieving 1e-16 machine epsilon convergence
            Vector delta = FastLinearSolver::solve_refined(A, Jtr);

            Vector x_new = x + delta;
            Vector r_new = residuals(x_new, target_h);
            Real err_new_sq = dot_product(r_new, r_new);
            
            Real predicted_red = 0;
            for(int i=0; i<n_vars; ++i) predicted_red += delta[i] * (lambda*delta[i]*JtJ[i][i] + Jtr[i]);
            
            Real rho = (err_sq - err_new_sq) / std::abs(predicted_red + 1e-20);

            if (err_new_sq < err_sq) {
                x = x_new;
                lambda *= std::max(0.33, 1.0 - std::pow(2.0*rho - 1.0, 3));
                v = 2.0;
            } else {
                lambda *= v;
                v *= 2.0;
            }
            
            if (lambda > 1e10) lambda = 1e10; 
            if (lambda < 1e-16) lambda = 1e-16;
        }
        unpack_state(x);
    }

    // --- SIMPLIFIED LINEAR HOMOTOPY ---
    void solve_adaptive() {
        // 1. Initial Guess (Linear Theory)
        Real L0 = (Phys::G_STD * T*T) / (2*Phys::PI);
        Real k0 = (d/L0 < 0.05) ? (2*Phys::PI / (T * std::sqrt(Phys::G_STD*d))) : (2*Phys::PI / L0);
        Real u_dop = (current_type == "Eulerian") ? Uc : 0.0;
        
        // Iterative Linear Solution for k0
        for(int i=0; i<50; ++i) {
            Real sig = 2*Phys::PI/T - k0*u_dop;
            if (sig <= 0) sig = 1e-5; // Protection against excessive counter-current
            Real next_k0 = 0.5*k0 + 0.5*(sig*sig / (Phys::G_STD * std::tanh(k0*d)));
            if (std::abs(next_k0 - k0) < 1e-15) { k0 = next_k0; break; }
            k0 = next_k0;
        }
        k = k0;
        
        for(int i=0; i<=N; ++i) eta_nodes[i] = d + (0.01/2.0)*std::cos(k * i * (Phys::PI/k)/N);
        std::fill(Bj.begin(), Bj.end(), 0.0);
        Q = (2*Phys::PI/k/T - Uc)*d; R = 0.5*std::pow(Q/d, 2) + Phys::G_STD*d;
        
        history.clear();

        // 2. Simple Linear Stepping Loop
        int n_steps = std::max(3, Defaults::HOMOTOPY_STEPS);
        Real h_start = 0.01;
        
        // Initialize at small amplitude
        levenberg_marquardt(h_start, 1e-3, 200);
        history.push_back({"Init", h_start, 0.0, "Init"});

        std::cout << "-----------------------------------------------------------------" << std::endl;
        std::cout << "   Type         | Height (H)   | Error        | Status" << std::endl;
        std::cout << "-----------------------------------------------------------------" << std::endl;
        
        String final_status = "FAIL";

        for(int i=1; i<=n_steps; ++i) {
            Real ratio = (Real)i / n_steps;
            Real h_target = h_start + (H_target - h_start) * ratio;
            bool is_last = (i == n_steps);
            
            // Standard LM Solve with MACHINE EPSILON PRECISION
            // Tolerance set to 4 * epsilon (~8.8e-16) to ensure convergence near limit
            Real tol = 4.0 * std::numeric_limits<Real>::epsilon(); 
            int max_iter = 2000;
            levenberg_marquardt(h_target, tol, max_iter); 
            
            // Verify
            Vector r = residuals(pack_state(), h_target);
            Real err_sq = 0; for(Real v : r) err_sq += v*v;
            Real err = std::sqrt(err_sq/r.size());
            
            String status = "OK";
            if (err > 1e-5) status = "FAIL"; // Tightened failure check
            
            if (is_last) {
                // Convergence classification logic updated for 1e-16 target
                if (err < 1e-14) final_status = "CONVERGED"; 
                else if (err < 2e-3) final_status = "ACCEPTED";
                else final_status = "DRIFT";
                status = final_status;
            }

            String label = is_last ? "Final" : ("Step " + std::to_string(i));
            history.push_back({label, h_target, err, status});
            
            std::cout << "   " << std::left << std::setw(13) << label << "| " 
                      << std::setw(13) << std::fixed << std::setprecision(3) << h_target << "| "
                      << std::setw(13) << std::scientific << std::setprecision(1) << err << "| " 
                      << status << std::endl;
        }

        // 3. Final Polish if needed
        // If we haven't hit strictly < 1e-14, force a deeper polish
        if (final_status != "CONVERGED") {
             Vector x_old = pack_state();
             Real err_old = history.back().err;
             
             // Attempt deep polish with strict machine epsilon
             levenberg_marquardt(H_target, std::numeric_limits<Real>::epsilon(), 5000); 
             
             Vector r = residuals(pack_state(), H_target);
             Real err_new = 0; for(Real v : r) err_new += v*v;
             err_new = std::sqrt(err_new/r.size());
             
             if (err_new < err_old) {
                 final_status = (err_new < 1e-14) ? "CONVERGED" : "ACCEPTED";
                 history.back().err = err_new;
                 history.back().status = final_status;
                 std::cout << "   " << std::left << std::setw(13) << "Final Polish" << "| " 
                           << std::setw(13) << std::fixed << std::setprecision(3) << H_target << "| "
                           << std::setw(13) << std::scientific << std::setprecision(1) << err_new << "| RECOVERED" << std::endl;
             } else {
                 unpack_state(x_old); 
             }
        }

        converged = (final_status == "CONVERGED" || final_status == "ACCEPTED");
        L = 2*Phys::PI/k; c = L/T;
    }

    void calculate_integral_properties() {
        if(current_type == "Eulerian") {
            prop_u1 = Uc; prop_U_frame = c - Uc; prop_u2 = c - Q/d;
        } else {
            prop_u2 = Uc; prop_u1 = c - Q/d; prop_U_frame = Q/d;
        }
        prop_q = prop_U_frame * d - Q;
        prop_I = Phys::RHO * (c * d - Q);
        
        for(int j=1; j<=N; ++j) {
            Real sum = 0;
            for(int m=0; m<=N; ++m) {
                Real val = (eta_nodes[m] - d) * std::cos(j * m * Phys::PI / N);
                Real wt = (m==0 || m==N) ? 0.5 : 1.0;
                sum += val * wt;
            }
            Ej[j-1] = (2.0/N) * sum;
        }
        
        Real sum_E2 = 0; for(Real e : Ej) sum_E2 += e*e;
        prop_PE = 0.25 * Phys::RHO * Phys::G_STD * sum_E2;
        prop_KE = 0.5 * (c * prop_I - prop_u1 * Q * Phys::RHO);
        
        // 1. Keep ub2_calculation as it was (Integral Method)
        Real ub2_sum = 0; int pts = N * 4;
        for(int i=0; i<pts; ++i) {
            Real x = i * (L/pts);
            auto kin = get_kinematics(0, x);
            ub2_sum += kin[0]*kin[0];
        }
        prop_ub2 = ub2_sum / pts;

        // 2. Sxx and F (using Algebraic ub2 for consistency with Fenton)
        // Fenton uses 2(R-gd) - c^2 for these specific flux/power calculations
        Real ub2_alg = 2.0 * (R - Phys::G_STD * d) - c * c;
        
        // Use ub2_alg ONLY for Sxx and F
        prop_Sxx = 4.0*prop_KE - 3.0*prop_PE + ub2_alg*(Phys::RHO*d) + 2.0*prop_u1*Phys::RHO*Q;
        prop_F = c*(3.0*prop_KE - 2.0*prop_PE) + 0.5*ub2_alg*(prop_I + Phys::RHO*c*d) + c*prop_u1*Phys::RHO*Q;
        
        prop_S = prop_Sxx - 2.0*c*prop_I + Phys::RHO*(c*c + 0.5*Phys::G_STD*d)*d;
        prop_r = R - Phys::G_STD*d; prop_R = R;
    }
};

// ==============================================================================
//  SECTION 5: FORCE CALCULATION (Smart Grid Integration)
// ==============================================================================

struct ForceResult {
    Real F_max, M_max, M_max_true;
    Real phase_max;
    Real F_drag, F_inertia;
    Real max_local_F, max_local_z;
    struct Node { Real z, u, ax, p, fd, fi, ft; };
    std::vector<Node> profile;
};

void calculate_forces(FentonWave& wave, Real D, Real mg, Real Cd, Real Cm, ForceResult& res) {
    if (!wave.converged) return;

    Real D_eff = D + 2*mg;
    
    // Helper to calculate total force at a specific phase
    auto get_force = [&](Real ph) {
        Real x_loc = -ph / wave.k; 
        Real eta = wave.get_eta(x_loc) - wave.d;
        
        std::vector<Real> zs;
        int base_steps = 2000; // Increased precision to match Python's Quad Integration
        
        for(int i=0; i<=base_steps; ++i) {
            zs.push_back(-wave.d + i*(wave.d + eta)/base_steps);
        }
        
        std::sort(zs.begin(), zs.end());
        auto last = std::unique(zs.begin(), zs.end());
        zs.erase(last, zs.end());
        
        Real F = 0, M = 0, Fd = 0, Fi = 0;
        
        for(size_t i=0; i<zs.size()-1; ++i) {
            Real z1 = zs[i];
            Real z2 = zs[i+1];
            
            if (z2 > eta) z2 = eta;
            if (z1 >= z2) continue;
            
            Real z_mid = (z1 + z2) / 2.0;
            Real dz = z2 - z1;
            
            // get_kinematics takes z_bed (0 is bed). z_mid is relative to MWL (-d is bed).
            auto k = wave.get_kinematics(z_mid + wave.d, x_loc);
            
            Real u = k[0]; 
            Real ax = k[2]; 
            
            Real fd_local = 0.5 * Phys::RHO * Cd * D_eff * u * std::abs(u);
            Real fi_local = Phys::RHO * Cm * (Phys::PI * D_eff*D_eff / 4.0) * ax;
            Real ft_local = fd_local + fi_local;
            
            F += ft_local * dz;
            // Moment about mudline: lever arm is (z_mid + d)
            M += ft_local * dz * (z_mid + wave.d); 
            Fd += fd_local * dz;
            Fi += fi_local * dz;
        }
        
        return std::vector<Real>{F, M, Fd, Fi};
    };
    
    Real best_ph = 0, max_F = 0, max_M_true = 0;
    
    // 1. Coarse Scan
    for(int i=0; i<360; ++i) {
        Real ph = i * (2*Phys::PI/360.0);
        auto f = get_force(ph);
        if(std::abs(f[0]) > max_F) { 
            max_F = std::abs(f[0]); 
            best_ph = ph; 
        }
        if(std::abs(f[1]) > max_M_true) max_M_true = std::abs(f[1]);
    }
    
    // 2. Fine Optimization: Golden Section Search
    auto optim_target = [&](Real x) {
        return std::abs(get_force(x)[0]); // Maximize |F|
    };
    
    // Search window +/- 0.3 rad around coarse peak
    Real a = best_ph - 0.3;
    Real b = best_ph + 0.3;
    best_ph = golden_section_search(optim_target, a, b, 1e-9); // Precision updated to 1e-9
    
    auto f_final = get_force(best_ph);
    res.F_max = f_final[0]; 
    res.M_max = f_final[1]; 
    res.F_drag = f_final[2]; 
    res.F_inertia = f_final[3]; 
    res.phase_max = best_ph; 
    res.M_max_true = max_M_true;
    
    // Generate Output Profile
    Real eta_final = wave.get_eta(-best_ph/wave.k) - wave.d;
    
    res.max_local_F = 0;
    res.max_local_z = 0;
    res.profile.clear();
    
    int pts = 50;
    for(int i=pts; i>=0; --i) { 
        Real z = -wave.d + i*(wave.d + eta_final)/pts;
        
        auto kin = wave.get_kinematics(z + wave.d, -best_ph/wave.k);
        
        Real fd = 0.5 * Phys::RHO * Cd * D_eff * kin[0] * std::abs(kin[0]);
        Real fi = Phys::RHO * Cm * (Phys::PI * D_eff*D_eff / 4.0) * kin[2];
        Real ft = fd + fi;
        
        if(std::abs(ft) > res.max_local_F) {
            res.max_local_F = std::abs(ft); 
            res.max_local_z = z;
        }

        // kin[4] returns Total Pressure (Hydrostatic + Dynamic). 
        // We must subtract Hydrostatic (-rho*g*z) to get Dynamic Pressure.
        Real p_total = kin[4];
        Real p_dyn = p_total + Phys::RHO * Phys::G_STD * z;

        res.profile.push_back({z, kin[0], std::abs(kin[2]), p_dyn, fd, std::abs(fi), ft});
    }
}

// ==============================================================================
//  SECTION 6: MAIN & REPORTING
// ==============================================================================

void get_morison_coefficients(Real mg, Real& Cd, Real& Cm, String& Source) {
    std::cout << "\n" << std::string(80, '=') << "\n MORISON COEFFICIENTS SELECTION\n" << std::string(80, '=') << "\n";
    std::cout << "1. BS 6349-1\n";
    std::cout << "2. USACE (CEM)\n";
    std::cout << "3. DNV-RP-C205 (North Sea)\n";
    std::cout << "4. API RP 2A-WSD\n";
    std::cout << "5. ISO 19902\n";
    std::cout << "6. User Defined (Manual Input)\n";

    String choice = get_input_str("Select Method [1-6]", "1");
    bool is_rough = (mg > 0.001);

    std::cout << "   -> Detected Surface State: " << (is_rough ? "ROUGH" : "SMOOTH")
              << " (mg = " << std::fixed << std::setprecision(3) << mg << "m)\n";

    if (choice == "6") {
        Cd = get_input("Enter Drag Coeff (Cd)", 1.30);
        Cm = get_input("Enter Inertia Coeff (Cm)", 2.00);
        Source = "User Defined";
        return;
    }

    if (choice == "1") { Source = "BS 6349-1";       if(is_rough) { Cd=1.3; Cm=2.0; } else { Cd=0.7; Cm=2.0; } }
    else if (choice == "2") { Source = "USACE (CEM)";     if(is_rough) { Cd=1.2; Cm=1.5; } else { Cd=0.7; Cm=1.5; } }
    else if (choice == "3") { Source = "DNV-RP-C205";     if(is_rough) { Cd=1.15; Cm=1.3; } else { Cd=0.65; Cm=1.6; } }
    else if (choice == "4") { Source = "API RP 2A-WSD";   if(is_rough) { Cd=1.05; Cm=1.2; } else { Cd=0.65; Cm=1.6; } }
    else if (choice == "5") { Source = "ISO 19902";       if(is_rough) { Cd=1.05; Cm=1.2; } else { Cd=0.65; Cm=1.6; } }
    else { 
         Source = "BS 6349-1"; if(is_rough) { Cd=1.3; Cm=2.0; } else { Cd=0.7; Cm=2.0; }
    }
}

int main(int argc, char* argv[]) {
    Logger log("output.txt");

    log.println(std::string(80, '='));
    log.println(" WAVE FORCE CALCULATOR - EXECUTIVE SUMMARY");
    log.println(std::string(80, '='));

    // 1. Define variables with defaults
    Real H = Defaults::WAVE_HEIGHT;
    Real T = Defaults::WAVE_PERIOD;
    Real d = Defaults::DEPTH;
    Real Uc = Defaults::CURRENT;
    String type = Defaults::CURRENT_TYPE;
    Real Dia = Defaults::PILE_DIAMETER;
    Real Mg = Defaults::MARINE_GROWTH;
    Real Cd = 1.3, Cm = 2.0; 
    String Src = "BS 6349-1";

    // 2. Argument Parsing Logic
    if (argc >= 10) {
        // --- Command Line Mode ---
        try {
            H = std::stod(argv[1]);
            T = std::stod(argv[2]);
            d = std::stod(argv[3]);
            Uc = std::stod(argv[4]);
            type = argv[5];
            Dia = std::stod(argv[6]);
            Mg = std::stod(argv[7]);
            Cd = std::stod(argv[8]);
            Cm = std::stod(argv[9]);
            Src = "Command Line Input";

            std::cout << "--- CLI MODE ACTIVATED ---" << std::endl;
            std::cout << "Inputs loaded from arguments successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
            std::cerr << "Usage: " << argv[0] << " [H] [T] [d] [Uc] [Type] [Dia] [Mg] [Cd] [Cm]" << std::endl;
            return 1;
        }
    } else {
        // --- Interactive Mode (Original Behavior) ---
        H = get_input("Wave Height (H)", Defaults::WAVE_HEIGHT);
        T = get_input("Wave Period (T)", Defaults::WAVE_PERIOD);
        d = get_input("Water Depth (d)", Defaults::DEPTH);
        Uc = get_input("Current Velocity (Uc)", Defaults::CURRENT);
        type = "Eulerian";
        Dia = get_input("Pile Diameter", Defaults::PILE_DIAMETER);
        Mg = get_input("Marine Growth", Defaults::MARINE_GROWTH);
        
        get_morison_coefficients(Mg, Cd, Cm, Src);
    }

    std::cout << "\nRunning Solver..." << std::endl;
    FentonWave wave(H, T, d, Uc, type);
    wave.solve();
    std::cout << "Done." << std::endl;

    if (!wave.converged) {
        log.println("\n [!] SOLVER FAILED OR ABORTED: SKIPPING QUANTITATIVE REPORT.");
        return 0;
    }

    ForceResult res;
    calculate_forces(wave, Dia, Mg, Cd, Cm, res);

    Real err = wave.history.back().err;
    
    std::stringstream ss;
    ss << " SOLVER STATUS:        " << (wave.converged ? "CONVERGED" : "FAILED") 
       << " (Final Residual: " << std::scientific << std::setprecision(1) << err << ")";
    log.println(ss.str());
    
    log.println(" ALGORITHM:            Fenton Fourier Stream Function (Order 50)");
    
    Real L = wave.L;
    Real Ur = (L > 0) ? (H*L*L/(d*d*d)) : 0;
    
    String regime = "SHALLOW";
    Real d_L = (L > 0) ? d/L : 0;
    if(d_L > 0.05 && d_L < 0.5) regime = "INTERMEDIATE";
    if(d_L >= 0.5) regime = "DEEP WATER";
    
    ss.str("");
    ss << " HYDRODYNAMICS:        " << regime << " WATER\n";
    ss << "                       d/L = " << std::fixed << std::setprecision(4) << d_L
       << "  |  H/L = " << ((L>0)?H/L:0)
       << "  |  Ur = " << std::fixed << std::setprecision(1) << Ur;
    log.println(ss.str());
    
    Real H_limit = (L > 0) ? (0.142 * L * std::tanh(wave.k * d)) : 0;
    String brk_msg = "STABLE (No Breaking)";
    if (H > H_limit) brk_msg = "CAUTION: WAVE NEAR BREAKING LIMIT";
    
    // UPDATED: Now includes (H/d) value
    std::stringstream ss_stab;
    ss_stab << " STABILITY CHECK:      " << brk_msg 
            << " (H/d = " << std::fixed << std::setprecision(3) << (H/d) << ")";
    log.println(ss_stab.str());
    
    ss.str("");
    ss << "                       (Limit H ~ " << std::fixed << std::setprecision(2) << H_limit 
       << " m based on Miche Criterion)";
    log.println(ss.str());
    log.println(std::string(80, '-'));

    Real lever = (std::abs(res.F_max) > 1e-4) ? (res.M_max/res.F_max) : 0;
    Real cob = lever; 

    log.print_force_row("MAX. BASE SHEAR:", res.F_max/1000.0, "kN");
    log.print_force_row("  |-> Drag Comp.:", res.F_drag/1000.0, "kN");
    log.print_force_row("  |-> Inertia Comp.:", res.F_inertia/1000.0, "kN");
    
    log.print_force_row("MAX. OTM (MUDLINE):", res.M_max_true/1000.0, "kNm");
    log.print_force_row("EFFECTIVE LEVER ARM:", lever, "m (Height from Seabed)"); 

    log.newline();
    log.println(std::string(80, '='));
    log.println(" 1. ENVIRONMENTAL & STRUCTURE DATA");
    log.println(std::string(80, '='));
    log.print_data_row("Wave Height (H)", H, "m");
    log.print_data_row("Wave Period (T)", T, "s");
    log.print_data_row("Water Depth (d)", d, "m");
    log.print_data_row("Current Velocity (Uc)", Uc, "m/s");
    log.print_data_str("Current Definition", type, "-");
    log.print_data_row("Local Gravity (g)", Phys::G_STD, "m/s2");
    log.print_data_row("Kinematic Viscosity (nu)", Phys::NU_SEAWATER*1e6, "10^-6 m2/s");
    log.println("---------------------------------------------------------------------------");
    Real Deff = Dia + 2*Mg;
    log.print_data_row("Pile Diameter", Dia, "m");
    log.print_data_row("Marine Growth", Mg, "m");
    log.print_data_row("Effective Diameter (D)", Deff, "m");
    log.print_data_row("Roughness Ratio (2*mg/D)", 2*Mg/Dia, "-");
    log.println("---------------------------------------------------------------------------");
    
    Real eta0 = wave.get_eta(0);
    Real u_tot_crest = wave.get_kinematics(eta0, 0)[0]; 
    Real u_orb = u_tot_crest - wave.prop_u1;            
    
    log.print_data_row("Calculated KC Number", u_orb*T/Deff, "-");
    log.print_data_row("Reynolds Number (Re)", u_tot_crest*Deff/Phys::NU_SEAWATER/1e6, "10^6 -");
    log.print_data_str("Surface State", (Mg > 0.001 ? "ROUGH" : "SMOOTH"), "-");
    log.print_data_str("Coefficient Source", Src, "-");
    log.print_data_row("Drag Coefficient (Cd)", Cd, "-");
    log.print_data_row("Inertia Coefficient (Cm)", Cm, "-");
    
    log.newline();
    log.println(std::string(80, '='));
    log.println(" 2. FENTON STREAM FUNCTION SOLUTION (ORDER 50)");
    log.println(std::string(80, '='));
    log.println("\n   --- SOLVER CONVERGENCE HISTORY ---");
    log.println("   STEP TYPE    | TARGET H (m) | MEAN ERROR   | STATUS");
    log.println("   -------------------------------------------------------");
    for(const auto& h : wave.history) {
        std::stringstream ss;
        ss << "   " << std::left << std::setw(13) << h.type << "| " 
           << std::setw(13) << std::fixed << std::setprecision(3) << h.h << "| "
           << std::setw(13) << std::scientific << std::setprecision(1) << h.err << "| "
           << h.status;
        log.println(ss.str());
    }
    log.println("   -------------------------------------------------------");
    
    log.println("\n----------------------------------------------------------------------------------------------------");
    log.println("   # INTEGRAL QUANTITIES - FENTON (1988) DEFINITIONS");
    log.println("   # (1) Quantity  (2) Symbol  (3) Dimensionless/(g,k)  (4) Dimensionless/(g,d)");
    log.println("----------------------------------------------------------------------------------------------------");
    
    auto pr_fen = [&](const String& n, const String& sym, Real v, Real sk, Real sd) {
        std::stringstream ss;
        ss << std::left << std::setw(36) << n << "| " << std::setw(11) << sym << "| "
           << std::setw(16) << std::fixed << std::setprecision(8) << v*sk << "| "
           << std::setw(16) << v*sd;
        log.println(ss.str());
    };
    
    Real k = wave.k, g = Phys::G_STD;
    // Scale Factors
    Real sl_k = k, sl_d = 1.0/d;
    Real st_k = std::sqrt(g*k), st_d = std::sqrt(g/d);
    Real sv_k = 1.0/std::sqrt(g/k), sv_d = 1.0/std::sqrt(g*d);
    Real sq_k = pow(k, 1.5)/sqrt(g), sq_d = 1.0/sqrt(g*pow(d,3));
    Real se_k = k/g, se_d = 1.0/(g*d);
    Real sp_k = k*k/(Phys::RHO*g), sp_d = 1.0/(Phys::RHO*g*d*d);
    Real sm_k = pow(k, 1.5)/(Phys::RHO*sqrt(g)), sm_d = 1.0/(Phys::RHO*d*sqrt(g*d));
    Real sw_k = pow(k, 2.5)/(Phys::RHO*pow(g, 1.5)), sw_d = 1.0/(Phys::RHO * pow(g, 1.5) * pow(d, 2.5));
    Real su_k = k/g, su_d = 1.0/(g*d);

    pr_fen("Water depth", "(d)", d, sl_k, sl_d);
    pr_fen("Wave length", "(lambda)", wave.L, sl_k, sl_d);
    pr_fen("Wave height", "(H)", H, sl_k, sl_d);
    pr_fen("Wave period", "(tau)", T, st_k, st_d);
    pr_fen("Wave speed", "(c)", wave.c, sv_k, sv_d);
    pr_fen("Eulerian current", "(u1_)", wave.prop_u1, sv_k, sv_d);
    pr_fen("Stokes current", "(u2_)", wave.prop_u2, sv_k, sv_d);
    pr_fen("Mean fluid speed in frame", "(U_)", wave.prop_U_frame, sv_k, sv_d);
    pr_fen("Volume flux due to waves", "(q)", wave.prop_q, sq_k, sq_d);
    pr_fen("Bernoulli constant (Excess)", "(r)", wave.prop_r, se_k, se_d);
    pr_fen("Volume flux (Total)", "(Q)", wave.Q, sq_k, sq_d);
    pr_fen("Bernoulli constant (Total)", "(R)", wave.prop_R, se_k, se_d);
    pr_fen("Momentum flux (Total)", "(S)", wave.prop_S, sp_k, sp_d);
    pr_fen("Impulse", "(I)", wave.prop_I, sm_k, sm_d);
    pr_fen("Kinetic energy", "(T)", wave.prop_KE, sp_k, sp_d);
    pr_fen("Potential energy", "(V)", wave.prop_PE, sp_k, sp_d);
    pr_fen("Mean sq bed velocity", "(ub2_)", wave.prop_ub2, su_k, su_d);
    pr_fen("Radiation stress", "(Sxx)", wave.prop_Sxx, sp_k, sp_d);
    pr_fen("Wave power", "(F)", wave.prop_F, sw_k, sw_d);

    log.newline();
    log.println(std::string(80, '='));
    log.println(" 3. FOURIER COEFFICIENTS (Bj & Ej)");
    log.println(std::string(80, '='));
    std::stringstream ss_kn; ss_kn << "   Wavenumber k = " << std::fixed << std::setprecision(6) << k << " rad/m";
    log.println(ss_kn.str());
    std::stringstream ss_Ln; ss_Ln << "   Wave Length L= " << std::fixed << std::setprecision(4) << wave.L << " m";
    log.println(ss_Ln.str());
    log.println("   -----------------------------------------------------------------");
    log.println("   j     | B[j] (Stream)      | E[j] (Elevation)  ");
    log.println("   -----------------------------------------------------------------");
    for(int i=0; i<wave.N; ++i) {
        std::stringstream ss;
        ss << "   " << std::left << std::setw(6) << (i+1) << "| " 
           << std::setw(19) << std::scientific << std::setprecision(8) << wave.Bj[i] << "| "
           << std::setw(19) << wave.Ej[i] * wave.k;
        log.println(ss.str());
    }
    log.println("   -----------------------------------------------------------------");

    log.newline();
    log.println(std::string(80, '='));
    log.println(" 4. FORCE & MOMENT CALCULATION RESULTS");
    log.println(std::string(80, '='));

    auto print_sect4 = [&](const String& desc, Real val, const String& unit) {
        std::stringstream ss_val;
        ss_val << std::fixed << std::setprecision(4) << val;
        
        std::stringstream ss_line;
        ss_line << std::left << std::setw(41) << desc << "| " 
                << std::setw(15) << ss_val.str() << " | " << unit;
        log.println(ss_line.str());
    };

    print_sect4("Maximum Total Force (Base Shear)", res.F_max/1000.0, "kN");
    print_sect4("Phase of Max Force", res.phase_max * 180/Phys::PI, "deg");
    print_sect4("Time of Max Force", res.phase_max / wave.k / wave.c, "s");
    log.println("------------------------------------------------------------");
    print_sect4("Max Overturning Moment (Sync)", res.M_max/1000.0, "kNm");
    print_sect4("Max Overturning Moment (True)", res.M_max_true/1000.0, "kNm");
    print_sect4("Center of Effort (from Bed)", cob, "m");
    print_sect4("Center of Effort (from MSL)", cob - d, "m");
    log.println("------------------------------------------------------------");
    print_sect4("Drag Component @ Max Load", res.F_drag/1000.0, "kN");
    print_sect4("Inertia Component @ Max Load", res.F_inertia/1000.0, "kN");
    log.println("------------------------------------------------------------");
    print_sect4("Max Local Force Density", res.max_local_F/1000.0, "kN/m");
    print_sect4("Elevation of Max Local Load", res.max_local_z, "m");
    
    log.println("\n   --- FORCE DISTRIBUTION PROFILE AT MAX LOAD PHASE ---");
    log.println("   ----------------------------------------------------------------------------------------------------");
    log.println("   Elev Z(m)    | Vel(m/s)     | Acc(m/s2)    | P_dyn(kPa)   | Fd (kN/m)    | Fi (kN/m)    | Ftot (kN/m) ");
    log.println("   ----------------------------------------------------------------------------------------------------");
    
    for(const auto& p : res.profile) {
        std::stringstream ss;
        ss << "   " << std::left << std::setw(12) << std::fixed << std::setprecision(3) << p.z << " | "
           << std::setw(12) << p.u << " | " << std::setw(12) << p.ax << " | "
           << std::setw(12) << p.p/1000.0 << " | " << std::setw(12) << p.fd/1000.0 << " | "
           << std::setw(12) << p.fi/1000.0 << " | "
           << std::setw(12) << p.ft/1000.0;
        log.println(ss.str());
    }
    log.println("   ----------------------------------------------------------------------------------------------------");

    return 0;
}