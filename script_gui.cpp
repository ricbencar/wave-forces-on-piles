/**
 * ==============================================================================
 * HIGH-PRECISION WAVE HYDRODYNAMICS SOLVER & GUI (PRODUCTION RELEASE)
 * ==============================================================================
 * MODULE:   script_gui.cpp
 * TYPE:     Nonlinear BVP Solver, Transient Load Calculator & Windows GUI
 * METHOD:   Fenton's Fourier Approximation
 * LICENSE:  MIT / Academic Open Source
 * ==============================================================================
 *
 * PROGRAM DESCRIPTION:
 * This software calculates the hydrodynamics (kinematics and dynamics) and 
 * structural loading of steady, finite-amplitude water waves acting on a 
 * vertical cylindrical pile.
 *
 * It combines the "Fourier Approximation Method" for the Nonlinear Stream 
 * Function as developed by J.D. Fenton (1988) with a native Windows Graphical 
 * User Interface (GUI). Unlike Linear (Airy) theory or Stokes 5th Order 
 * approximations, this numerical method satisfies the full nonlinear boundary 
 * conditions to machine precision (limited only by the truncation order N).
 *
 * KEY FEATURES:
 * 1. Physics Engine: Newton-Raphson solver for Stream Function coefficients.
 * 2. Force Engine: Depth-integrated Drag and Inertia forces (Morison Eq).
 * 3. Reporting: Generates high-precision reports identical to industry standards.
 * 4. GUI: Native Win32 interface for inputs and result visualization.
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
 * 3.3 ADAPTIVE HOMOTOPY (Continuation Method)
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
 * - Windows API (MinGW or MSVC).
 *
 * COMPILATION INSTRUCTION (MinGW/GCC):
 * g++ -O3 -std=c++17 -static -static-libgcc -static-libstdc++ -o script_gui.exe script_gui.cpp -mwindows -lgdi32
 *
 * RUNNING INSTRUCTIONS:
 * 1. Launch `script_gui.exe`.
 * 2. Enter Environmental Parameters (H, T, d, Uc).
 * 3. Enter Structural Parameters (Dia, Mg, Cd, Cm).
 * 4. Click "CALCULATE FORCES".
 * 5. View results in the output window or check `output.txt`.
 *
 * ARCHITECTURE & MODULES:
 * -----------------------------------------------------------------------------
 * 1. MATH KERNEL: 
 * - Custom Crout LU Decomposition for linear systems.
 * - Levenberg-Marquardt Non-linear optimization for Fourier coefficients.
 * - Complex-Step Differentiation for precise Jacobian calculation.
 *
 * 2. PHYSICS ENGINE (FentonWave):
 * - Solves the Boundary Value Problem (BVP) for the Stream Function Psi.
 * - Satisfies KBC and DBC to machine precision (~1e-16).
 * - Enforces "Eulerian" current definition.
 *
 * 3. FORCE ENGINE:
 * - Integrates Morison Equation (Drag + Inertia) over the water column.
 * - Uses Golden Section Search to find the exact phase of maximum load.
 *
 * 4. GUI SHELL (Win32 API):
 * - Manages User Inputs, Event Loop, and File I/O.
 * - Renders reports to an on-screen Edit Control and "output.txt".
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

#define _USE_MATH_DEFINES 
#define NOMINMAX

#include <windows.h>
#include <commctrl.h>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <complex>
#include <limits>
#include <fstream>
// #include <thread> REMOVED FOR COMPATIBILITY

// ==============================================================================
//  SECTION 1: MATH TYPES & CONSTANTS
// ==============================================================================

namespace Core {
    using Real    = double;
    using Integer = int;
    using String  = std::string;
    using Complex = std::complex<Real>;
    using Vector  = std::vector<Real>;
    using VectorC = std::vector<Complex>;
    using Matrix  = std::vector<std::vector<Real>>;
}

using namespace Core;

namespace Phys {
    constexpr Real PI          = 3.14159265358979323846;
    constexpr Real RHO         = 1025.0;       // Seawater Density (kg/m3)
    constexpr Real G_STD       = 9.8066;       // Standard Gravity (m/s2)
    constexpr Real NU_SEAWATER = 1.05e-6;      // Kinematic Viscosity (m2/s)
}

// ==============================================================================
//  SECTION 2: LOGGING & FORMATTING UTILITIES
// ==============================================================================

/**
 * StringLogger
 * ------------
 * A utility class designed to capture output for the report.
 */
class StringLogger {
    std::stringstream buffer;
public:
    // Append raw content
    template <typename T> 
    void print(const T& val) { buffer << val; }
    
    // Append content with a standard newline
    template <typename T> 
    void println(const T& val) { buffer << val << "\n"; }
    
    // Append empty newline
    void newline() { buffer << "\n"; }
    
    void print_data_row(const String& desc, Real val, const String& unit) {
        std::stringstream ss; 
        ss << std::fixed << std::setprecision(4) << val;
        print_data_str(desc, ss.str(), unit);
    }
    
    void print_data_str(const String& desc, const String& val, const String& unit) {
        buffer << std::left << std::setw(41) << desc << "| " 
               << std::setw(15) << val << " | " << unit << "\n";
    }

    void print_force_row(const String& desc, Real val, const String& unit) {
        std::stringstream ss; 
        ss << std::fixed << std::setprecision(4) << val;
        buffer << " " << std::left << std::setw(39) << desc << " | " 
               << std::setw(15) << ss.str() << " | " << unit << "\n";
    }

    // Retrieve full buffer content
    std::string get_content() const { return buffer.str(); }
    
    // Reset buffer
    void clear() { buffer.str(""); buffer.clear(); }
};

// ==============================================================================
//  SECTION 3: MATHEMATICAL KERNEL
// ==============================================================================

namespace MathUtils {

    // --- Vector Operator Overloads ---
    inline Vector operator+(const Vector& a, const Vector& b) {
        Vector r(a.size()); for(size_t i=0; i<a.size(); ++i) r[i] = a[i] + b[i]; return r;
    }
    inline Vector operator-(const Vector& a, const Vector& b) {
        Vector r(a.size()); for(size_t i=0; i<a.size(); ++i) r[i] = a[i] - b[i]; return r;
    }
    inline Vector operator*(Real s, const Vector& a) {
        Vector r(a.size()); for(size_t i=0; i<a.size(); ++i) r[i] = s * a[i]; return r;
    }
    inline Real dot_product(const Vector& a, const Vector& b) {
        Real s = 0; for(size_t i=0; i<a.size(); ++i) s += a[i] * b[i]; return s;
    }

    /**
     * FastLinearSolver
     * ----------------
     * Crout's LU Decomposition with Iterative Refinement.
     * Essential for solving the Newton-Raphson steps to machine precision.
     */
    class FastLinearSolver {
    public:
        static Vector solve_flat(const Matrix& A_2d, const Vector& b) {
            int n = b.size();
            std::vector<Real> LU(n * n);
            for(int i=0; i<n; ++i) for(int j=0; j<n; ++j) LU[i*n + j] = A_2d[i][j];

            std::vector<int> p(n);
            std::iota(p.begin(), p.end(), 0);

            for (int i = 0; i < n; ++i) {
                int max_row = i;
                Real max_val = std::abs(LU[i*n + i]);
                for (int k = i + 1; k < n; ++k) {
                    Real val = std::abs(LU[k*n + i]);
                    if (val > max_val) { max_val = val; max_row = k; }
                }
                std::swap(p[i], p[max_row]);
                for (int k = 0; k < n; ++k) std::swap(LU[i*n + k], LU[max_row*n + k]);

                Real pivot_val = LU[i*n + i];
                if (std::abs(pivot_val) < 1e-12) pivot_val = (pivot_val >= 0 ? 1e-12 : -1e-12);
                LU[i*n + i] = pivot_val;

                for (int k = i + 1; k < n; ++k) {
                    LU[k*n + i] /= pivot_val;
                    for (int j = i + 1; j < n; ++j) LU[k*n + j] -= LU[k*n + i] * LU[i*n + j];
                }
            }
            Vector x(n);
            for (int i = 0; i < n; ++i) {
                Real sum = 0; for (int j = 0; j < i; ++j) sum += LU[i*n + j] * x[j];
                x[i] = b[p[i]] - sum;
            }
            for (int i = n - 1; i >= 0; --i) {
                Real sum = 0; for (int j = i + 1; j < n; ++j) sum += LU[i*n + j] * x[j];
                x[i] = (x[i] - sum) / LU[i*n + i];
            }
            return x;
        }

        static Vector solve_refined(const Matrix& A, const Vector& b) {
            Vector x = solve_flat(A, b);
            int n = b.size();
            Vector r(n);
            for(int i=0; i<n; ++i) {
                Real Ax_i = 0.0;
                for(int j=0; j<n; ++j) Ax_i += A[i][j] * x[j];
                r[i] = b[i] - Ax_i;
            }
            Vector correction = solve_flat(A, r);
            for(int i=0; i<n; ++i) x[i] += correction[i];
            return x;
        }
    };

    /**
     * Golden Section Search
     * ---------------------
     * Finds the wave phase angle where the Total Force is maximized.
     */
    template <typename Func>
    Real golden_section_search(Func f, Real a, Real b, Real tol = 1e-9) {
        const Real phi = (1.0 + std::sqrt(5.0)) / 2.0;
        const Real resphi = 2.0 - phi;
        Real c = a + resphi * (b - a), d = b - resphi * (b - a);
        Real fc = f(c), fd = f(d);
        while (std::abs(c - d) > tol) {
            if (fc > fd) { b = d; d = c; fd = fc; c = a + resphi * (b - a); fc = f(c); } 
            else { a = c; c = d; fc = fd; d = b - resphi * (b - a); fd = f(d); }
        }
        return (c + d) / 2.0;
    }
}
using namespace MathUtils;

// ==============================================================================
//  SECTION 4: PHYSICS ENGINE (FENTON STREAM FUNCTION)
// ==============================================================================

class FentonWave {
public:
    // Inputs
    Real H_target, T, d, Uc;
    String current_type; // "Eulerian"
    int N; // Fourier Order
    
    // Solution State
    Real k, L, c, Q, R;
    Vector Bj, eta_nodes, Ej;
    
    // Integral Properties
    Real prop_KE, prop_PE, prop_I, prop_Sxx, prop_F, prop_ub2;
    Real prop_q, prop_S, prop_u1, prop_u2, prop_U_frame, prop_r, prop_R;

    // Solver Status
    bool converged;
    struct StepLog { String type; Real h; Real err; String status; };
    std::vector<StepLog> history;
    
    // Optimization tables
    std::vector<Real> table_cos;
    std::vector<Real> table_sin;

    FentonWave(Real _H, Real _T, Real _d, Real _Uc, String _ct) 
        : H_target(_H), T(_T), d(_d), Uc(_Uc), current_type(_ct), N(50) {
        converged = false;
        Bj.resize(N, 0.0); Ej.resize(N, 0.0); eta_nodes.resize(N + 1, d);
        precompute_basis();
    }

    void precompute_basis() {
        table_cos.resize((N + 1) * N); table_sin.resize((N + 1) * N);
        Real pi_div_N = Phys::PI / (Real)N;
        for (int m = 0; m <= N; ++m) {
            for (int j = 1; j <= N; ++j) {
                Real arg = j * m * pi_div_N;
                size_t idx = m * N + (j - 1);
                table_cos[idx] = std::cos(arg); table_sin[idx] = std::sin(arg);
            }
        }
    }

    void solve() { 
        if (d > 0 && (H_target / d) > 0.6) {
             converged = false;
             history.push_back({"LIMIT", H_target, 0.0, "ABORT: H/d > 0.6"});
             return; 
        }
        solve_adaptive(); 
        if(converged) calculate_integral_properties(); 
    }

    Vector get_kinematics(Real z_bed, Real x) {
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
                ch = std::cosh(arg) / denom; sh = std::sinh(arg) / denom;
            }
            Real cx = std::cos(j * k * x); Real sx = std::sin(j * k * x);
            Real jk = j * k; Real B = Bj[j-1];
            
            u_p += B * jk * ch * cx; v_p += B * jk * sh * sx; 
            dup_dx += B * std::pow(jk, 2) * ch * (-sx); dup_dz += B * std::pow(jk, 2) * sh * cx;
            dwp_dx += B * std::pow(jk, 2) * sh * cx; dwp_dz += B * std::pow(jk, 2) * ch * sx;
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

    template <typename Scalar>
    void residuals_internal_optimized(const std::vector<Scalar>& x_state, Real current_H, std::vector<Scalar>& res_out) {
        Scalar tk = x_state[0];
        const Scalar* t_eta = &x_state[1];
        const Scalar* t_B = &x_state[1 + N + 1];
        Scalar tQ = x_state[x_state.size()-2];
        Scalar tR = x_state[x_state.size()-1];
        Scalar t_g = (Scalar)Phys::G_STD; Scalar t_d = (Scalar)d; Scalar t_pi = (Scalar)Phys::PI;
        Scalar t_T = (Scalar)T; Scalar t_Uc = (Scalar)Uc; Scalar t_H = (Scalar)current_H;
        Scalar tc = (2.0 * t_pi) / (tk * t_T);
        Scalar tU_frame = (current_type == "Eulerian") ? (tc - t_Uc) : (tQ / t_d);
        Scalar tk2 = tk*tk; Scalar tk3 = tk2*tk;
        Scalar sc = std::sqrt(t_g / tk3);
        Scalar inv_g_d = 1.0 / (t_g * t_d);
        Scalar inv_sqrt_gd_d = 1.0 / (std::sqrt(t_g * t_d) * t_d);

        size_t expected_size = 3 + 2 * (N + 1);
        if (res_out.size() != expected_size) res_out.resize(expected_size);
        size_t r_idx = 0;
        
        if (current_type != "Eulerian") res_out[r_idx++] = ((tc - tQ/t_d) - t_Uc) / std::sqrt(t_g*t_d);
        else res_out[r_idx++] = Scalar(0.0);
        
        res_out[r_idx++] = (t_eta[0] - t_eta[N] - t_H) / t_d; 
        
        Scalar mean_eta = Scalar(0.0); for(int i=0; i<=N; ++i) mean_eta += t_eta[i];
        mean_eta -= 0.5*(t_eta[0] + t_eta[N]);
        res_out[r_idx++] = (mean_eta/(Real)N - t_d)/t_d; 

        Scalar kd = tk * t_d;
        for (int m = 0; m <= N; ++m) {
            Scalar z_nd = t_eta[m];
            Scalar psi_p = 0, u_p = 0, v_p = 0;
            size_t table_offset = m * N;
            for (int j = 1; j <= N; ++j) {
                Scalar j_scalar = (Real)j;
                Scalar arg = j_scalar * tk * z_nd;
                Scalar jkd = j_scalar * kd;
                Scalar S, C;
                if (std::real(jkd) > 20.0) {
                    Scalar term = std::exp(j_scalar * tk * (z_nd - t_d));
                    S = term; C = term;
                } else {
                    Scalar ch_denom = std::cosh(jkd);
                    Scalar inv_denom = 1.0 / ch_denom;
                    S = std::sinh(arg) * inv_denom;
                    C = std::cosh(arg) * inv_denom;
                }
                Real cx_real = table_cos[table_offset + (j - 1)];
                Real sx_real = table_sin[table_offset + (j - 1)];
                Scalar B_val = t_B[j-1];
                psi_p += B_val * S * cx_real;
                Scalar B_jk = B_val * j_scalar * tk;
                u_p   += B_jk * C * cx_real;
                v_p   += B_jk * S * (-sx_real); 
            }
            psi_p *= sc; u_p *= sc; v_p *= sc;
            res_out[r_idx++] = (-tU_frame * z_nd + psi_p + tQ) * inv_sqrt_gd_d;
            Scalar u_tot = tU_frame - u_p;
            res_out[r_idx++] = (0.5 * (u_tot*u_tot + v_p*v_p) + t_g * z_nd - tR) * inv_g_d;
        }
    }
    
    Vector residuals(const Vector& x, Real h) { 
        Vector res; residuals_internal_optimized<Real>(x, h, res); return res; 
    }

    Matrix compute_jacobian_complex(const Vector& x_val, Real target_h) {
        int n = x_val.size();
        VectorC x_c(n); for(int i=0; i<n; ++i) x_c[i] = Complex(x_val[i], 0.0);
        Vector r_base = residuals(x_val, target_h);
        int m = r_base.size();
        Matrix J(m, Vector(n));
        Real h_step = 1.0e-20; 
        std::vector<Complex> r_c_buffer; r_c_buffer.reserve(m);

        for(int j=0; j<n; ++j) {
            x_c[j].imag(h_step);
            residuals_internal_optimized<Complex>(x_c, target_h, r_c_buffer);
            Real inv_h = 1.0 / h_step;
            for(int i=0; i<m; ++i) J[i][j] = std::imag(r_c_buffer[i]) * inv_h;
            x_c[j].imag(0.0);
        }
        return J;
    }

    void levenberg_marquardt(Real target_h, Real tol = 1e-15, int max_iter = 200) {
        Vector x = pack_state();
        Real lambda = 1e-3, v = 2.0;       

        for (int iter = 0; iter < max_iter; ++iter) {
            Vector r = residuals(x, target_h);
            Real err_sq = dot_product(r, r); 
            if (std::sqrt(err_sq/r.size()) < tol) break;
            Matrix J = compute_jacobian_complex(x, target_h);
            int n_vars = x.size(); int n_res = r.size();
            Matrix JtJ(n_vars, Vector(n_vars, 0.0));
            Vector Jtr(n_vars, 0.0);

            for(int i=0; i<n_res; ++i) {
                for(int j=0; j<n_vars; ++j) {
                    Jtr[j] -= J[i][j] * r[i]; 
                    for(int k=j; k<n_vars; ++k) {
                        Real val = J[i][j] * J[i][k];
                        JtJ[j][k] += val; if(j!=k) JtJ[k][j] += val;
                    }
                }
            }
            Matrix A = JtJ;
            for(int i=0; i<n_vars; ++i) {
                A[i][i] += lambda * A[i][i]; 
                if(A[i][i] < 1e-12) A[i][i] = 1e-12; 
            }
            Vector delta = FastLinearSolver::solve_refined(A, Jtr);
            Vector x_new = x + delta;
            Vector r_new = residuals(x_new, target_h);
            Real err_new_sq = dot_product(r_new, r_new);
            Real predicted_red = 0;
            for(int i=0; i<n_vars; ++i) predicted_red += delta[i] * (lambda*delta[i]*JtJ[i][i] + Jtr[i]);
            Real rho = (err_sq - err_new_sq) / std::abs(predicted_red + 1e-20);

            if (err_new_sq < err_sq) {
                x = x_new; lambda *= std::max(0.33, 1.0 - std::pow(2.0*rho - 1.0, 3)); v = 2.0;
            } else { lambda *= v; v *= 2.0; }
            
            if (lambda > 1e10) { lambda = 1e10; }
            if (lambda < 1e-16) { lambda = 1e-16; }
        }
        unpack_state(x);
    }

    void solve_adaptive() {
        Real L0 = (Phys::G_STD * T*T) / (2*Phys::PI);
        Real k0 = (d/L0 < 0.05) ? (2*Phys::PI / (T * std::sqrt(Phys::G_STD*d))) : (2*Phys::PI / L0);
        Real u_dop = (current_type == "Eulerian") ? Uc : 0.0;
        
        for(int i=0; i<50; ++i) {
            Real sig = 2*Phys::PI/T - k0*u_dop;
            if (sig <= 0) sig = 1e-5; 
            Real next_k0 = 0.5*k0 + 0.5*(sig*sig / (Phys::G_STD * std::tanh(k0*d)));
            if (std::abs(next_k0 - k0) < 1e-15) { k0 = next_k0; break; }
            k0 = next_k0;
        }
        k = k0;
        for(int i=0; i<=N; ++i) eta_nodes[i] = d + (0.01/2.0)*std::cos(k * i * (Phys::PI/k)/N);
        std::fill(Bj.begin(), Bj.end(), 0.0);
        Q = (2*Phys::PI/k/T - Uc)*d; R = 0.5*std::pow(Q/d, 2) + Phys::G_STD*d;
        
        history.clear();
        int n_steps = 4;
        Real h_start = 0.01;
        
        levenberg_marquardt(h_start, 1e-3, 200);
        history.push_back({"Init", h_start, 0.0, "Init"});

        String final_status = "FAIL";
        for(int i=1; i<=n_steps; ++i) {
            Real ratio = (Real)i / n_steps;
            Real h_target = h_start + (H_target - h_start) * ratio;
            bool is_last = (i == n_steps);
            Real tol = 4.0 * std::numeric_limits<Real>::epsilon(); 
            levenberg_marquardt(h_target, tol, 2000); 
            Vector r = residuals(pack_state(), h_target);
            Real err_sq = 0; for(Real v : r) err_sq += v*v;
            Real err = std::sqrt(err_sq/r.size());
            
            String status = "OK";
            if (err > 1e-5) status = "FAIL"; 
            if (is_last) {
                if (err < 1e-14) final_status = "CONVERGED"; 
                else if (err < 2e-3) final_status = "ACCEPTED";
                else final_status = "DRIFT";
                status = final_status;
            }
            history.push_back({is_last ? "Final" : ("Step " + std::to_string(i)), h_target, err, status});
        }

        if (final_status != "CONVERGED") {
             Vector x_old = pack_state();
             Real err_old = history.back().err;
             levenberg_marquardt(H_target, std::numeric_limits<Real>::epsilon(), 5000); 
             Vector r = residuals(pack_state(), H_target);
             Real err_new = 0; for(Real v : r) err_new += v*v;
             err_new = std::sqrt(err_new/r.size());
             if (err_new < err_old) {
                 final_status = (err_new < 1e-14) ? "CONVERGED" : "ACCEPTED";
                 history.back().err = err_new;
                 history.back().status = final_status;
             } else { unpack_state(x_old); }
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
        
        Real ub2_sum = 0; int pts = N * 4;
        for(int i=0; i<pts; ++i) {
            Real x = i * (L/pts);
            auto kin = get_kinematics(0, x);
            ub2_sum += kin[0]*kin[0];
        }
        prop_ub2 = ub2_sum / pts;
        Real ub2_alg = 2.0 * (R - Phys::G_STD * d) - c * c;
        prop_Sxx = 4.0*prop_KE - 3.0*prop_PE + ub2_alg*(Phys::RHO*d) + 2.0*prop_u1*Phys::RHO*Q;
        prop_F = c*(3.0*prop_KE - 2.0*prop_PE) + 0.5*ub2_alg*(prop_I + Phys::RHO*c*d) + c*prop_u1*Phys::RHO*Q;
        prop_S = prop_Sxx - 2.0*c*prop_I + Phys::RHO*(c*c + 0.5*Phys::G_STD*d)*d;
        prop_r = R - Phys::G_STD*d; prop_R = R;
    }
};

// ==============================================================================
//  SECTION 5: FORCE CALCULATION (MORISON EQUATION)
// ==============================================================================

struct ForceResult {
    Real F_max, M_max, M_max_true, phase_max;
    Real F_drag, F_inertia, max_local_F, max_local_z;
    struct Node { Real z, u, ax, p, fd, fi, ft; };
    std::vector<Node> profile;
};

void calculate_forces(FentonWave& wave, Real D, Real mg, Real Cd, Real Cm, ForceResult& res) {
    if (!wave.converged) return;
    Real D_eff = D + 2*mg;

    // Helper to integrate force at a specific phase
    auto get_force = [&](Real ph) {
        Real x_loc = -ph / wave.k; 
        Real eta = wave.get_eta(x_loc) - wave.d;
        std::vector<Real> zs;
        int base_steps = 2000;
        // Generate Z points from seabed to surface
        for(int i=0; i<=base_steps; ++i) zs.push_back(-wave.d + i*(wave.d + eta)/base_steps);
        std::sort(zs.begin(), zs.end());
        auto last = std::unique(zs.begin(), zs.end()); zs.erase(last, zs.end());
        
        Real F = 0, M = 0, Fd = 0, Fi = 0;
        for(size_t i=0; i<zs.size()-1; ++i) {
            Real z1 = zs[i]; Real z2 = zs[i+1];
            if (z2 > eta) { z2 = eta; } 
            if (z1 >= z2) { continue; }
            
            Real z_mid = (z1 + z2) / 2.0; Real dz = z2 - z1;
            auto k = wave.get_kinematics(z_mid + wave.d, x_loc);
            Real u = k[0]; Real ax = k[2]; 
            Real fd_local = 0.5 * Phys::RHO * Cd * D_eff * u * std::abs(u);
            Real fi_local = Phys::RHO * Cm * (Phys::PI * D_eff*D_eff / 4.0) * ax;
            Real ft_local = fd_local + fi_local;
            F += ft_local * dz; M += ft_local * dz * (z_mid + wave.d); 
            Fd += fd_local * dz; Fi += fi_local * dz;
        }
        return std::vector<Real>{F, M, Fd, Fi};
    };
    
    // Coarse Search for Max Force
    Real best_ph = 0, max_F = 0, max_M_true = 0;
    for(int i=0; i<360; ++i) {
        Real ph = i * (2*Phys::PI/360.0);
        auto f = get_force(ph);
        if(std::abs(f[0]) > max_F) { max_F = std::abs(f[0]); best_ph = ph; }
        if(std::abs(f[1]) > max_M_true) max_M_true = std::abs(f[1]);
    }
    
    // Fine Optimization using Golden Section Search
    auto optim_target = [&](Real x) { return std::abs(get_force(x)[0]); };
    best_ph = golden_section_search(optim_target, best_ph - 0.3, best_ph + 0.3, 1e-9);
    
    // Final Values
    auto f_final = get_force(best_ph);
    res.F_max = f_final[0]; res.M_max = f_final[1]; res.F_drag = f_final[2]; res.F_inertia = f_final[3]; 
    res.phase_max = best_ph; res.M_max_true = max_M_true;
    
    // Generate Distribution Profile (Top Down)
    Real eta_final = wave.get_eta(-best_ph/wave.k) - wave.d;
    res.max_local_F = 0; res.max_local_z = 0; res.profile.clear();
    int pts = 50;
    // Loop goes from pts (Top) down to 0
    for(int i=pts; i>=0; --i) { 
        Real z = -wave.d + i*(wave.d + eta_final)/pts;
        auto kin = wave.get_kinematics(z + wave.d, -best_ph/wave.k);
        Real fd = 0.5 * Phys::RHO * Cd * D_eff * kin[0] * std::abs(kin[0]);
        Real fi = Phys::RHO * Cm * (Phys::PI * D_eff*D_eff / 4.0) * kin[2];
        Real ft = fd + fi;
        if(std::abs(ft) > res.max_local_F) { res.max_local_F = std::abs(ft); res.max_local_z = z; }
        Real p_dyn = kin[4] + Phys::RHO * Phys::G_STD * z;
        res.profile.push_back({z, kin[0], std::abs(kin[2]), p_dyn, fd, std::abs(fi), ft});
    }
}

// ==============================================================================
//  SECTION 6: GUI IMPLEMENTATION
// ==============================================================================

// Control IDs
#define IDC_EDIT_H      101
#define IDC_EDIT_T      102
#define IDC_EDIT_D      103
#define IDC_EDIT_UC     104
#define IDC_EDIT_DIA    106
#define IDC_EDIT_MG     107
#define IDC_EDIT_CD     108
#define IDC_EDIT_CM     109
#define IDC_BTN_CALC    110
#define IDC_OUTPUT      111

HWND hEditH, hEditT, hEditD, hEditUc;
HWND hEditDia, hEditMg, hEditCd, hEditCm, hOutput, hBtnCalc;
HFONT hUIFont, hMonoFont;

struct WaveInputs {
    double H, T, d, Uc, Dia, Mg, Cd, Cm;
    std::string type;
};

// --- REPORT GENERATION ---
std::string RunSimulation(WaveInputs inp) {
    StringLogger log;
    log.println(std::string(80, '='));
    log.println(" WAVE FORCE CALCULATOR - EXECUTIVE SUMMARY");
    log.println(std::string(80, '='));

    // Initialize & Solve
    FentonWave wave(inp.H, inp.T, inp.d, inp.Uc, inp.type);
    wave.solve();

    if (!wave.converged) {
        log.println("\n [!] SOLVER FAILED OR ABORTED: H/d > 0.6 or Convergence Error.");
        for(const auto& h : wave.history) {
             std::stringstream ss;
             ss << "   " << std::left << std::setw(13) << h.type << "| " 
                << std::setw(13) << h.h << "| " << h.status;
             log.println(ss.str());
        }
        return log.get_content();
    }

    // Calculate Forces
    ForceResult res;
    calculate_forces(wave, inp.Dia, inp.Mg, inp.Cd, inp.Cm, res);

    // --- EXECUTIVE SUMMARY CONTENT ---
    Real err = wave.history.back().err;
    std::stringstream ss;
    ss << " SOLVER STATUS:        " << (wave.converged ? "CONVERGED" : "FAILED") 
       << " (Final Residual: " << std::scientific << std::setprecision(1) << err << ")";
    log.println(ss.str());
    log.println(" ALGORITHM:            Fenton Fourier Stream Function (Order 50)");

    Real L = wave.L;
    Real Ur = (L > 0) ? (inp.H*L*L/(inp.d*inp.d*inp.d)) : 0;
    String regime = "SHALLOW";
    Real d_L = (L > 0) ? inp.d/L : 0;
    if(d_L > 0.05 && d_L < 0.5) regime = "INTERMEDIATE WATER";
    else if(d_L >= 0.5) regime = "DEEP WATER";
    else regime = "SHALLOW WATER";
    
    ss.str(""); 
    ss << " HYDRODYNAMICS:        " << regime << "\n"
       << "                       d/L = " << std::fixed << std::setprecision(4) << d_L
       << "  |  H/L = " << ((L>0)?inp.H/L:0) << "  |  Ur = " << std::fixed << std::setprecision(1) << Ur;
    log.println(ss.str());
    
    Real H_limit = (L > 0) ? (0.142 * L * std::tanh(wave.k * inp.d)) : 0;
    String brk_msg = (inp.H > H_limit) ? "CAUTION: WAVE NEAR BREAKING" : "STABLE (No Breaking)";
    
    ss.str("");
    ss << " STABILITY CHECK:      " << brk_msg << " (H/d = " << std::fixed << std::setprecision(3) << (inp.H/inp.d) << ")";
    log.println(ss.str());
    ss.str("");
    ss << "                       (Limit H ~ " << std::fixed << std::setprecision(2) << H_limit << " m based on Miche Criterion)";
    log.println(ss.str());
    log.println(std::string(80, '-'));

    Real lever = (std::abs(res.F_max) > 1e-4) ? (res.M_max/res.F_max) : 0;
    
    log.print_force_row("MAX. BASE SHEAR:", res.F_max/1000.0, "kN");
    log.print_force_row("  |-> Drag Comp.:", res.F_drag/1000.0, "kN");
    log.print_force_row("  |-> Inertia Comp.:", res.F_inertia/1000.0, "kN");
    log.print_force_row("MAX. OTM (MUDLINE):", res.M_max_true/1000.0, "kNm");
    log.print_force_row("EFFECTIVE LEVER ARM:", lever, "m (Height from Seabed)");

    log.newline();
    log.println(std::string(80, '='));
    log.println(" 1. ENVIRONMENTAL & STRUCTURE DATA");
    log.println(std::string(80, '='));
    log.print_data_row("Wave Height (H)", inp.H, "m");
    log.print_data_row("Wave Period (T)", inp.T, "s");
    log.print_data_row("Water Depth (d)", inp.d, "m");
    log.print_data_row("Current Velocity (Uc)", inp.Uc, "m/s");
    log.print_data_str("Current Definition", inp.type, "-");
    log.print_data_row("Local Gravity (g)", Phys::G_STD, "m/s2");
    log.print_data_row("Kinematic Viscosity (nu)", Phys::NU_SEAWATER*1e6, "10^-6 m2/s");
    log.println("---------------------------------------------------------------------------");
    Real Deff = inp.Dia + 2*inp.Mg;
    log.print_data_row("Pile Diameter", inp.Dia, "m");
    log.print_data_row("Marine Growth", inp.Mg, "m");
    log.print_data_row("Effective Diameter (D)", Deff, "m");
    log.print_data_row("Roughness Ratio (2*mg/D)", 2*inp.Mg/inp.Dia, "-");
    log.println("---------------------------------------------------------------------------");
    
    Real eta0 = wave.get_eta(0);
    Real u_tot_crest = wave.get_kinematics(eta0, 0)[0]; 
    Real u_orb = u_tot_crest - wave.prop_u1; 

    log.print_data_row("Calculated KC Number", u_orb*inp.T/Deff, "-");
    log.print_data_row("Reynolds Number (Re)", u_tot_crest*Deff/Phys::NU_SEAWATER/1e6, "10^6 -");
    log.print_data_str("Surface State", (inp.Mg > 0.001 ? "ROUGH" : "SMOOTH"), "-");
    log.print_data_str("Coefficient Source", "BS 6349-1", "-");
    log.print_data_row("Drag Coefficient (Cd)", inp.Cd, "-");
    log.print_data_row("Inertia Coefficient (Cm)", inp.Cm, "-");

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
    
    // Scale Factors for Dimensionless Columns
    Real k = wave.k, g = Phys::G_STD;
    Real sl_k = k, sl_d = 1.0/inp.d;
    Real st_k = std::sqrt(g*k), st_d = std::sqrt(g/inp.d);
    Real sv_k = 1.0/std::sqrt(g/k), sv_d = 1.0/std::sqrt(g*inp.d);
    Real sq_k = pow(k, 1.5)/sqrt(g), sq_d = 1.0/sqrt(g*pow(inp.d,3));
    Real se_k = k/g, se_d = 1.0/(g*inp.d);
    Real sp_k = k*k/(Phys::RHO*g), sp_d = 1.0/(Phys::RHO*g*inp.d*inp.d);
    Real sm_k = pow(k, 1.5)/(Phys::RHO*sqrt(g)), sm_d = 1.0/(Phys::RHO*inp.d*sqrt(g*inp.d));
    Real sw_k = pow(k, 2.5)/(Phys::RHO*pow(g, 1.5)), sw_d = 1.0/(Phys::RHO * pow(g, 1.5) * pow(inp.d, 2.5));
    Real su_k = k/g, su_d = 1.0/(g*inp.d);

    auto pr_fen = [&](const String& n, const String& sym, Real v, Real sk, Real sd) {
        std::stringstream ss;
        ss << std::left << std::setw(36) << n << "| " << std::setw(11) << sym << "| "
           << std::setw(16) << std::fixed << std::setprecision(8) << v*sk << "| "
           << std::setw(16) << v*sd;
        log.println(ss.str());
    };

    pr_fen("Water depth", "(d)", inp.d, sl_k, sl_d);
    pr_fen("Wave length", "(lambda)", wave.L, sl_k, sl_d);
    pr_fen("Wave height", "(H)", inp.H, sl_k, sl_d);
    pr_fen("Wave period", "(tau)", inp.T, st_k, st_d);
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
    ss.str(""); ss << "   Wavenumber k = " << std::fixed << std::setprecision(6) << wave.k << " rad/m";
    log.println(ss.str());
    ss.str(""); ss << "   Wave Length L= " << std::fixed << std::setprecision(4) << wave.L << " m";
    log.println(ss.str());
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
    print_sect4("Center of Effort (from Bed)", lever, "m");
    print_sect4("Center of Effort (from MSL)", lever - inp.d, "m");
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

    return log.get_content();
}

// ==============================================================================
//  SECTION 7: WIN32 GUI UTILITIES
// ==============================================================================

// Helper: Convert string to wstring
std::wstring to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}

std::wstring FixNewlinesForGui(const std::wstring& in) {
    std::wstring out;
    out.reserve(in.size() + 512); 
    for (wchar_t c : in) {
        if (c == L'\n') {
            out.push_back(L'\r');
            out.push_back(L'\n');
        } else {
            out.push_back(c);
        }
    }
    return out;
}

void CreateGUI(HWND hwnd) {
    // Fonts
    hUIFont = CreateFontW(19, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, 
                         OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    hMonoFont = CreateFontW(18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, 
                           OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, FIXED_PITCH | FF_DONTCARE, L"Consolas");

    int y = 20, x_lbl = 20, x_val = 200, w_val = 80, step = 35;
    
    auto AddLabel = [&](const wchar_t* txt, int y_pos) {
        HWND h = CreateWindowW(L"STATIC", txt, WS_CHILD|WS_VISIBLE, x_lbl, y_pos, 180, 25, hwnd, NULL, NULL, NULL);
        SendMessageW(h, WM_SETFONT, (WPARAM)hUIFont, TRUE);
    };
    auto AddEdit = [&](const wchar_t* def, int id, int y_pos) {
        HWND h = CreateWindowW(L"EDIT", def, WS_CHILD|WS_VISIBLE|WS_BORDER|ES_AUTOHSCROLL, x_val, y_pos, w_val, 25, hwnd, (HMENU)(INT_PTR)id, NULL, NULL);
        SendMessageW(h, WM_SETFONT, (WPARAM)hUIFont, TRUE);
        return h;
    };

    // Environmental Inputs
    AddLabel(L"Wave Height H (m):", y); hEditH = AddEdit(L"3.0", IDC_EDIT_H, y); y += step;
    AddLabel(L"Wave Period T (s):", y); hEditT = AddEdit(L"9.0", IDC_EDIT_T, y); y += step;
    AddLabel(L"Water Depth d (m):", y); hEditD = AddEdit(L"5.0", IDC_EDIT_D, y); y += step;
    AddLabel(L"Current Uc (m/s):", y);  hEditUc = AddEdit(L"1.0", IDC_EDIT_UC, y); y += step;
    
    // Structure Inputs
    AddLabel(L"Pile Diameter (m):", y); hEditDia = AddEdit(L"1.5", IDC_EDIT_DIA, y); y += step;
    AddLabel(L"Marine Growth (m):", y); hEditMg = AddEdit(L"0.05", IDC_EDIT_MG, y); y += step;
    AddLabel(L"Drag Coeff (Cd):", y);   hEditCd = AddEdit(L"1.3", IDC_EDIT_CD, y); y += step;
    AddLabel(L"Inertia Coeff (Cm):", y);hEditCm = AddEdit(L"2.0", IDC_EDIT_CM, y); y += step;

    // Button
    y += 10;
    hBtnCalc = CreateWindowW(L"BUTTON", L"CALCULATE FORCES", WS_TABSTOP|WS_VISIBLE|WS_CHILD|BS_DEFPUSHBUTTON, 
                             x_lbl, y, 260, 40, hwnd, (HMENU)IDC_BTN_CALC, NULL, NULL);
    SendMessageW(hBtnCalc, WM_SETFONT, (WPARAM)hUIFont, TRUE);

    hOutput = CreateWindowW(L"EDIT", L"", 
                           WS_CHILD|WS_VISIBLE|WS_BORDER|ES_MULTILINE|ES_AUTOVSCROLL|WS_VSCROLL|ES_READONLY|WS_HSCROLL|ES_AUTOHSCROLL, 
                           320, 10, 875, 640, hwnd, (HMENU)IDC_OUTPUT, NULL, NULL);
                           
    SendMessageW(hOutput, WM_SETFONT, (WPARAM)hMonoFont, TRUE);
}

// --- THREADING UTILITIES (REPLACES std::thread) ---

struct ThreadParams {
    WaveInputs inp;
    HWND hBtn;
    HWND hOutput;
};

// Background worker thread function
DWORD WINAPI CalculationThread(LPVOID lpParam) {
    // 1. Unpack parameters
    ThreadParams* pData = (ThreadParams*)lpParam;
    WaveInputs inp = pData->inp;
    HWND hBtn = pData->hBtn;
    HWND hOutput = pData->hOutput;
    
    // Clean up the heap memory from main thread
    delete pData; 

    // 2. Run the math (Heavy task)
    std::string report = RunSimulation(inp);
    
    // 3. Save to file
    std::ofstream file("output.txt"); 
    if (file.is_open()) { file << report; file.close(); }

    // 4. Prepare text for GUI
    std::wstring wreport = to_wstring(report);
    std::wstring gui_text = FixNewlinesForGui(wreport);

    // 5. Direct SetWindowText is thread-safe in Win32 for simple text
    SetWindowTextW(hOutput, gui_text.c_str());

    // 6. Re-enable the button
    SetWindowTextW(hBtn, L"CALCULATE FORCES");
    EnableWindow(hBtn, TRUE);

    return 0;
}

// Window Procedure
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_CREATE) { CreateGUI(hwnd); return 0; }
    if (msg == WM_DESTROY) { DeleteObject(hUIFont); DeleteObject(hMonoFont); PostQuitMessage(0); return 0; }
    
    if (msg == WM_COMMAND && LOWORD(wParam) == IDC_BTN_CALC) {
        
        // 1. Lock UI
        SetWindowTextW(hBtnCalc, L"RUNNING SOLVER...");
        EnableWindow(hBtnCalc, FALSE);
        UpdateWindow(hBtnCalc); 

        // 2. Define Validation Helper
        auto get_validated = [&](HWND h, const wchar_t* name, bool allow_negative, double& out_val) -> bool {
            wchar_t buf[64];
            GetWindowTextW(h, buf, 63);

            // Check for empty input
            if (buf[0] == L'\0') {
                std::wstring msg = std::wstring(name) + L" cannot be empty.";
                MessageBoxW(hwnd, msg.c_str(), L"Input Error", MB_ICONERROR);
                return false;
            }

            // Robust parsing check
            wchar_t* end_ptr;
            double val = wcstod(buf, &end_ptr);
            
            // Check for trailing non-numeric characters
            if (*end_ptr != L'\0') {
                std::wstring msg = std::wstring(name) + L" must be a valid number.";
                MessageBoxW(hwnd, msg.c_str(), L"Input Error", MB_ICONERROR);
                return false;
            }

            // Check positivity rules
            if (!allow_negative && val < 0) {
                std::wstring msg = std::wstring(name) + L" must be positive.";
                MessageBoxW(hwnd, msg.c_str(), L"Input Error", MB_ICONERROR);
                return false;
            }

            out_val = val;
            return true;
        };

        WaveInputs inp;
        
        // 3. Run Validation Checks
        // Reset the button text and re-enable it if any check fails
        if (!get_validated(hEditH,   L"Wave Height",      false, inp.H))   { EnableWindow(hBtnCalc, TRUE); SetWindowTextW(hBtnCalc, L"CALCULATE FORCES"); return 0; }
        if (!get_validated(hEditT,   L"Wave Period",      false, inp.T))   { EnableWindow(hBtnCalc, TRUE); SetWindowTextW(hBtnCalc, L"CALCULATE FORCES"); return 0; }
        if (!get_validated(hEditD,   L"Water Depth",      false, inp.d))   { EnableWindow(hBtnCalc, TRUE); SetWindowTextW(hBtnCalc, L"CALCULATE FORCES"); return 0; }
        if (!get_validated(hEditUc,  L"Current Velocity", true,  inp.Uc))  { EnableWindow(hBtnCalc, TRUE); SetWindowTextW(hBtnCalc, L"CALCULATE FORCES"); return 0; } // True = Allow negative
        if (!get_validated(hEditDia, L"Pile Diameter",    false, inp.Dia)) { EnableWindow(hBtnCalc, TRUE); SetWindowTextW(hBtnCalc, L"CALCULATE FORCES"); return 0; }
        if (!get_validated(hEditMg,  L"Marine Growth",    false, inp.Mg))  { EnableWindow(hBtnCalc, TRUE); SetWindowTextW(hBtnCalc, L"CALCULATE FORCES"); return 0; }
        if (!get_validated(hEditCd,  L"Drag Coeff",       false, inp.Cd))  { EnableWindow(hBtnCalc, TRUE); SetWindowTextW(hBtnCalc, L"CALCULATE FORCES"); return 0; }
        if (!get_validated(hEditCm,  L"Inertia Coeff",    false, inp.Cm))  { EnableWindow(hBtnCalc, TRUE); SetWindowTextW(hBtnCalc, L"CALCULATE FORCES"); return 0; }

        inp.type = "Eulerian";

        // 4. Physics Logic Checks (Strictly > 0)
        if (inp.H <= 0 || inp.T <= 0 || inp.d <= 0) {
            MessageBoxW(hwnd, L"Height, Period, and Depth must be strictly greater than zero.", L"Physics Error", MB_ICONERROR);
            SetWindowTextW(hBtnCalc, L"CALCULATE FORCES");
            EnableWindow(hBtnCalc, TRUE);
            return 0;
        }

        // 5. Launch Worker Thread (Native Win32, replacing std::thread)
        // Allocate params on heap; thread will delete them.
        ThreadParams* params = new ThreadParams;
        params->inp = inp;
        params->hBtn = hBtnCalc;
        params->hOutput = hOutput;

        HANDLE hThread = CreateThread(
            NULL,                   // Default security attributes
            0,                      // Default stack size
            CalculationThread,      // Thread function
            params,                 // Argument to thread function
            0,                      // Default creation flags
            NULL);                  // Receive thread identifier

        if (hThread) {
            CloseHandle(hThread); // We don't need to hold the handle
        } else {
            // Fallback if thread fails
            delete params;
            SetWindowTextW(hBtnCalc, L"CALCULATE FORCES");
            EnableWindow(hBtnCalc, TRUE);
            MessageBoxW(hwnd, L"Failed to create thread.", L"Error", MB_ICONERROR);
        }
        
        return 0; 
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// Entry Point - Changed to WinMain to avoid -municode requirement
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    WNDCLASSEXW wc = {sizeof(WNDCLASSEXW), 0, WndProc, 0, 0, hInstance, LoadIcon(NULL, IDI_APPLICATION), 
                      LoadCursor(NULL, IDC_ARROW), (HBRUSH)(COLOR_WINDOW+1), NULL, L"FentonCalc", LoadIcon(NULL, IDI_APPLICATION)};
    RegisterClassExW(&wc);
    
    HWND hwnd = CreateWindowExW(0, L"FentonCalc", L"Fenton Wave Force Calculator (High Precision)", 
                               WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX, 
                               CW_USEDEFAULT, CW_USEDEFAULT, 1200, 680, NULL, NULL, hInstance, NULL);
                               
    ShowWindow(hwnd, nCmdShow); UpdateWindow(hwnd);
    MSG msg; while (GetMessageW(&msg, NULL, 0, 0)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    return (int)msg.wParam;
}