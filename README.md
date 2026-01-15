# WAVE FORCES ON CYLINDRICAL PILES CALCULATOR

## 1. Introduction and Engineering Context

The expansion of resilient nearshore infrastructure, particularly pile-supported structures such as open-piled quays, jetties, and mooring pontoons requires the rigorous analysis of hydrodynamic loading on these assets, which are a cornerstone of coastal and port engineering and directly determine their operational safety, structural longevity, and economic viability.

Namelly in nearshore environments, waves interact complexly with the seabed and structural foundations. The interaction between steep, shallow-water nonlinear waves and vertical cylindrical piles supporting the referred structures demands a level of mathematical fidelity that far exceeds the capabilities of traditional linear wave theories.

This document serves as the scientific manual and operational guide for the **Wave Forces on Cylindrical Piles Calculator** (referred to herein as the Calculator). This computational tool provides a high-precision solution for the fluid-structure interaction of vertical cylindrical piles which are essential to modern port and harbor facilities.

### 1.1 The Limitations of Linear Theory in Extreme Design

Historically, preliminary design and fatigue analysis have relied on **Linear (Airy) Wave Theory**, which assumes wave amplitudes are infinitesimally small relative to water depth. While computationally efficient, this assumption breaks down catastrophically in the regime of **Ultimate Limit State (ULS)** design.

Real ocean waves during storm conditions are inherently nonlinear. As the wave height ($H$) increases relative to the water depth ($d$) or wavelength ($L$), the wave profile distorts: crests become sharper and higher, while troughs become flatter.

This asymmetry has profound implications for structural loading:

* **Kinematic Underestimation:** Linear theory significantly underestimates particle velocities in the crest region—precisely where the lever arm acting on the pile is maximized.
* **Splash Zone Integration:** Linear theory integrates forces only up to the Mean Water Level (MWL). In reality, the "splash zone" (the area between MWL and the wave crest) contributes a disproportionately large amount of the total base shear and overturning moment.
* **Mass Transport:** Nonlinear waves exhibit a net mass transport (drift) that linear theory fails to capture, altering the drag forces acting on the structure.

### 1.2 The Computational Solution

The Software addresses these limitations by abandoning perturbative expansions (like Stokes 5th order) in favor of a numerical solution to the full nonlinear boundary value problem. By integrating **Fenton’s Fourier Approximation Method** (see https://johndfenton.com/Steady-waves/Fourier.html), the software solves the stream function $\psi(x,z)$ to machine precision, satisfying the nonlinear free surface boundary conditions exactly.

This kinematic fidelity is coupled with robust force formulation of the **The Morison Equation:** used for slender piles ($D/L < 0.2$), calculating the superposition of viscous drag and inertial forces.

---

## 2. Theoretical Framework: The Nonlinear Boundary Value Problem

The hydrodynamic foundation of the Software is the **Stream Function Theory**. Unlike velocity potential formulations, the stream function approach naturally satisfies the continuity equation for incompressible flow and offers a convenient mechanism for satisfying the bottom boundary condition in constant depth.

### 2.1 Governing Equations

We consider a two-dimensional, periodic gravity wave propagating in the positive $x$-direction over a horizontal bed. The fluid is assumed to be inviscid and incompressible. The flow is irrotational, meaning fluid particles do not rotate about their own axes, a valid assumption for wave propagation outside the immediate viscous boundary layer.

A Cartesian coordinate system is defined with $x$ horizontal and $z$ vertical, measured upwards from the seabed ($z=0$). The fluid velocity vector $\mathbf{u} = (u, w)$ is derived from the scalar stream function $\psi(x, z)$ as follows:

$$
u = \frac{\partial \psi}{\partial z}, \quad w = -\frac{\partial \psi}{\partial x}
$$

The condition of irrotationality requires that the vorticity $\omega$ be zero everywhere:

$$
\omega = \frac{\partial w}{\partial x} - \frac{\partial u}{\partial z} = -\frac{\partial^2 \psi}{\partial x^2} - \frac{\partial^2 \psi}{\partial z^2} = -\nabla^2 \psi = 0
$$

Thus, the governing field equation is the Laplace Equation:

$$
\nabla^2 \psi = 0 \quad \text{for} \quad 0 \le z \le \eta(x)
$$

This elliptic partial differential equation must be satisfied throughout the fluid domain bounded by the seabed and the instantaneous free surface $\eta(x)$.

### 2.2 The Moving Reference Frame

A critical simplification in steady wave theory is the transformation to a reference frame moving with the wave celerity $c$.

* In the physical (fixed) frame $(x, z, t)$, the wave profile is unsteady.
* However, in a frame $(X, z)$ moving at speed $c$, where $X = x - ct$, the wave appears stationary.



**Stationary Frame Variables:** Denoted by uppercase (e.g., $U, W$). The flow appears as a steady current moving in the negative $X$ direction under a frozen wave profile.

**Physical Frame Variables:** Denoted by lowercase (e.g., $u, w$).

$$
u(x, z, t) = U(X, z) + c, \quad w(x, z, t) = W(X, z)
$$

This transformation reduces the problem from a time-dependent evolution to a steady-state boundary value problem, significantly reducing computational complexity without loss of generality.

### 2.3 Boundary Conditions

The physics of the problem are constrained by boundary conditions at the seabed and the free surface.

#### 2.3.1 Bottom Boundary Condition (BBC)
The seabed is impermeable and horizontal. In the stationary frame, this implies that the vertical velocity $W$ must be zero at $z=0$. In terms of the stream function:

$$
\psi(X, 0) = 0
$$

This defines the seabed as the "zero" streamline.

#### 2.3.2 Free Surface Boundary Conditions
The free surface $z = \eta(X)$ is the interface between the water and the atmosphere. It is defined by two nonlinear conditions:

**1. The Kinematic Boundary Condition (KBC):**
The free surface is a material boundary; no fluid can cross it. In the stationary frame, this means the free surface is a streamline. If $Q$ is the volume flux per unit width (the discharge) under the wave, the value of the stream function at the surface is constant:

$$
\psi(X, \eta(X)) = -Q
$$

This equation links the unknown surface elevation $\eta$ to the stream function field.

**2. The Dynamic Boundary Condition (DBC):**
The pressure at the free surface is constant (equal to atmospheric pressure). Applying the steady Bernoulli equation along the surface streamline:

$$
\frac{1}{2} \left[ \left( \frac{\partial \psi}{\partial X} \right)^2 + \left( \frac{\partial \psi}{\partial z} \right)^2 \right] + g\eta(X) = R
$$

Here, $R$ is the Bernoulli Constant, representing the total energy head of the flow. This equation provides the coupling between the flow kinematics and the potential energy of the wave. The simultaneous satisfaction of these two conditions determines the shape of the wave and the internal flow field.

---

## 3. Numerical Solution: Fenton's Fourier Approximation

Analytical solutions to the nonlinear BVP (like Stokes theory) rely on perturbation expansions in terms of wave steepness. These series diverge for high waves. To overcome this, the Software implements the **Stream Function Fourier Approximation Method** described by Fenton (1988), which assumes a solution form that satisfies the field equations exactly and solves for the coefficients numerically.

The Stream Function is a scalar field defined for two-dimensional flows. It exists under the assumption that the fluid is **incompressible** (density is constant) and **irrotational**. Its primary utility is that its spatial derivatives directly provide the fluid's velocity components:

* **Horizontal Velocity ($u$):** $u = \frac{\partial \psi}{\partial z}$.
* **Vertical Velocity ($w$):** $w = - \frac{\partial \psi}{\partial x}$.

By defining velocity this way, the condition of incompressibility (Conservation of Mass) is automatically satisfied.

* **Streamlines:** A line along which the value of $\psi$ is constant is called a *streamline*. In steady flow, streamlines represent the actual trajectories of fluid particles.
* **Impermeable Boundaries:** Since velocity is tangent to streamlines, fluid cannot cross a streamline. This property is used to define boundaries:
    * **Seabed:** The bottom of the fluid domain ($z=0$) is modeled as a streamline where vertical velocity is zero.
    * **Free Surface:** The water surface ($z = \eta$) is also a streamline, implying that no water particles leave the surface to enter the air, effectively acting as a "lid" for the fluid domain.
* **Volume Flux:** The difference in the value of the stream function between two streamlines represents the volume flux (flow rate per unit width) passing between them. In this software, the surface streamline is defined as $\psi = -Q$, where $Q$ is the total volume flux.

In the context of the provided software, the Stream Function is the core variable solved by the algorithm.

* **Governing Equation:** The solver searches for a $\psi$ field that satisfies the **Laplace Equation** ($\nabla^2\psi = 0$) throughout the water column.
* **Fourier Approximation:** Instead of solving for $\psi$ at every grid point, the software approximates it using a truncated **Fourier series** of order $N$. The solver adjusts the coefficients ($B_j$) of this series until the stream function satisfies the physical boundary conditions (Bernoulli's principle) at the free surface.

### 3.1 The Fourier Ansatz

The stream function is approximated by a truncated Fourier series of order $N$. The specific mathematical form is chosen to satisfy the Laplace equation ($\nabla^2\psi=0$) and the Bottom Boundary Condition ($\psi(X,0)=0$) analytically:

$$
\psi(X, z) = -(\bar{u} + c)z + \sqrt{\frac{g}{k^3}} \sum_{j=1}^{N} B_j \frac{\sinh(jkz)}{\cosh(jkd)} \cos(jkX)
$$

**Term-by-Term Analysis:**
* $-(\bar{u} + c)z$: Represents the uniform current component. In the moving frame, the apparent current is the superposition of the Eulerian current $\bar{u}$ and the frame velocity $c$.
* $\sqrt{g/k^3}$: A scaling factor introduced to render the Fourier coefficients $B_j$ dimensionless, improving the conditioning of the numerical system.
* $\sinh(jkz)$: Hyperbolic sine function. It is zero at $z=0$, satisfying the BBC, and grows exponentially with depth, capturing the depth-decay of wave orbitals.
* $\cos(jkX)$: Trigonometric term representing the periodic variation of the wave in the horizontal direction.
* $B_j$: The $N$ dimensionless coefficients that define the unique wave solution.

By using this ansatz, the dimensionality of the problem is reduced from a 2D field calculation to finding the finite set of coefficients $B_j$ and wave parameters ($k, Q, R$) that minimize errors at the free surface.

### 3.2 The System of Nonlinear Equations

The continuous boundary conditions are discretized at $M$ points (nodes) equally spaced over half a wavelength, taking advantage of the wave's symmetry about the crest.

$$
X_m = \frac{m L}{2 M}, \quad m = 0, 1, \dots, M
$$

The unknowns to be solved are vector $\mathbf{Z}$:

$$
\mathbf{Z} = [k, R, Q, B_1, B_2, \dots, B_N]^T
$$

This vector contains $N+3$ variables. At each node $m$, the surface elevation $\eta_m$ is implicitly defined by the KBC:

$$
\psi(X_m, \eta_m; \mathbf{Z}) + Q = 0
$$

For a given guess of $\mathbf{Z}$, the Software solves this equation numerically (using a sub-iteration) to find the corresponding $\eta_m$. Once $\eta_m$ is known, the error in the Dynamic Boundary Condition (DBC) can be computed:

$$
E_m = \frac{1}{2} (U_m^2 + W_m^2) + g\eta_m - R
$$

Where velocities $U_m, W_m$ are analytical derivatives of the stream function ansatz evaluated at $(X_m, \eta_m)$.

To close the system, additional integral constraints are enforced:

**Mean Water Level:** The spatial average of surface elevations must equal the water depth $d$.
$$
\frac{1}{L} \int_0^L \eta(X) dX - d = 0
$$

**Wave Height:** The difference between crest and trough elevations must equal the specified height $H$.
$$
\eta(0) - \eta(L/2) - H = 0
$$

This formulation results in a system of $M + 2$ nonlinear equations for the $N+3$ unknowns. Typically, $M$ is chosen such that the system is overdetermined ($M \ge N+1$), and solved in a least-squares sense.

### 3.3 Limitations of Analytical Methods

It is crucial to emphasize why this method supersedes Stokes theory. Stokes expansions are power series in terms of wave steepness $\epsilon = kH/2$.

$$
\eta(x) \approx \epsilon \cos(kx) + \epsilon^2 B_{22} \cos(2kx) + \dots
$$

For steep waves, $\epsilon$ is not small, and the series convergence is slow or non-existent. In shallow water, the "Ursell Paradox" means contributions from higher-order terms become comparable to the leading order, causing perturbation methods to diverge. Fenton's method, being spectral, does not rely on a small parameter expansion; its accuracy depends only on the number of Fourier modes $N$, making it robust for both deep-water breaking waves and shallow-water cnoidal-like waves.

---

## 4. Algorithmic Implementation: The Nonlinear Solver

The numerical core of the Calculator departs from the legacy Fortran implementations of the 1980s by utilizing modern optimization algorithms from the **SciPy** library. This shift significantly enhances stability and maintainability.

### 4.1 Optimization Engine: Trust Region Reflective (TRF)

The original Fenton (1988) algorithm employed a custom Newton-Raphson solver. While fast, Newton-Raphson is sensitive to the initial guess; if the starting point is outside the "basin of attraction," the solver can diverge, predicting physically impossible waves (e.g., negative depths).

The Calculator employs the **Trust Region Reflective (TRF)** algorithm via `scipy.optimize.least_squares`.

**Mechanism of TRF:**
Unlike line-search methods that determine a direction and then a step size, TRF defines a region around the current iterate within which the local quadratic model of the objective function is "trusted."

* **Approximation:** At step $k$, the cost function $F(\mathbf{Z})$ is approximated by a quadratic model $m_k(p)$.
* **Constraint:** The step $p$ is computed to minimize $m_k(p)$ subject to $||p|| \le \Delta_k$, where $\Delta_k$ is the trust region radius.
* **Evaluation:** If the actual reduction in $F(\mathbf{Z}+p)$ agrees well with the predicted reduction, the step is accepted and $\Delta_k$ may be increased. If not, the region is shrunk.
* **Reflection:** The "Reflective" aspect handles bound constraints (e.g., $k > 0$, $d > 0$) by reflecting the search path off the boundaries rather than truncating it, allowing for efficient exploration of the feasible space.

This method is particularly advantageous for the wave problem, where the solution space is bounded by physical breaking limits.

### 4.2 Jacobian Estimation: Finite Difference

The Jacobian matrix $\mathbf{J}$ contains the partial derivatives of the residuals with respect to the unknowns:

$$
J_{ij} = \frac{\partial E_i}{\partial Z_j}
$$

Fenton (1988) derived analytical expressions for these derivatives. While elegant, implementing analytical Jacobians for high-order Fourier series is error-prone and rigid. The Software uses a **2-point finite difference scheme** to estimate $\mathbf{J}$ numerically.

$$
J_{ij} \approx \frac{E_i(\mathbf{Z} + \delta \mathbf{e}_j) - E_i(\mathbf{Z})}{\delta}
$$

The step size $\delta$ is adaptively chosen based on the machine epsilon to balance truncation error and floating-point round-off error. This allows the solver to treat the stream function evaluation as a "black box," facilitating future upgrades to the physics (e.g., adding variable bathymetry terms) without rewriting the solver logic.

### 4.3 Homotopy (Continuation) Strategy

A common failure mode in nonlinear wave solvers is the "cold start" problem: attempting to solve for a steep, near-breaking wave using linear theory as an initial guess often fails because the physics are too dissimilar.

The Software implements a **Homotopy (Continuation) Method**:

1.  **Linear Seed:** The solver starts with a wave of very small height ($H_{start} \approx H_{target}/10$), for which Linear Theory provides an excellent guess.
2.  **Incremental Stepping:** The solution for height $H_i$ is used as the initial guess for height $H_{i+1} = H_i + \Delta H$.
3.  **Path Following:** This process traces the solution curve through the parameter space, gently guiding the solver from the linear regime into the strongly nonlinear regime.

The number of steps $n$ is a user-configurable parameter, allowing trade-offs between execution speed and robustness.

### 4.4 Solver Outputs

Upon convergence, the solver returns the state vector $\mathbf{Z}$. Key derived quantities include:

* **Wavenumber ($k$):** Determining the true nonlinear wavelength $L = 2\pi/k$.
* **Bernoulli Constant ($R$):** Indicating the energy level of the flow.
* **Flux ($Q$):** The mass transport rate.
* **Coefficients ($B_j$):** The spectral content of the wave.

The residual norm (sum of squared errors) is logged to `output.txt` to verify the solution quality, typically achieving values $< 10^{-10}$.

---

## 5. Hydrodynamic Loading I: The Morison Equation

Once the kinematic field (velocities and accelerations) is fully resolved, the Software calculates structural loads. For slender cylindrical piles, where the diameter $D$ is less than 20% of the wavelength ($D/L < 0.2$), diffraction effects are negligible. The flow remains attached or separates locally without globally altering the wave field. In this regime, the **Morison Equation** is the industry standard.

### 5.1 Formulation of Force Components

The Morison equation superimposes two force mechanisms: a Drag force (viscous) and an Inertia force (inviscid/mass). The force per unit height $dF(z)$ is given by:

$$
dF(z, t) = dF_D + dF_I
$$

#### 5.1.1 The Drag Force
The drag component arises from the pressure differential caused by flow separation and the formation of a wake downstream of the pile. It is proportional to the square of the velocity:

$$
dF_D = \frac{1}{2} \rho C_D D (u |u|) dz
$$

* **Nonlinearity:** The term $u|u|$ preserves the sign, ensuring drag opposes the instantaneous velocity vector.
* **Currents:** The velocity $u$ represents the **total** velocity ($u_{wave} + U_c$). The presence of a current $U_c$ introduces a quadratic coupling (e.g., $(u_{wave} + U_c)^2 = u_{wave}^2 + 2u_{wave}U_c + U_c^2$), significantly amplifying the load compared to waves alone.
* **Coefficient $C_D$:** The drag coefficient depends on the Reynolds number ($Re$) and the Keulegan-Carpenter number ($KC$). While the Software allows for a constant user-defined $C_D$, engineers must select this value based on the expected flow regime (typically 0.7 for smooth piles, up to 1.3 for marine-fouled piles).

#### 5.1.2 The Inertia Force
The inertia component acts on the fluid mass displaced by the cylinder and the "added mass" of fluid surrounding it. It is proportional to acceleration:

$$
dF_I = \rho C_M \frac{\pi D^2}{4} \dot{u} dz
$$

* **Froude-Krylov Force:** The force required to accelerate the fluid that *would* have occupied the pile's volume ($\rho V \dot{u}$).
* **Added Mass:** The force required to accelerate the surrounding fluid entrained by the cylinder's presence ($\rho C_a V \dot{u}$).
* **Coefficient $C_M$:** The inertia coefficient is $C_M = 1 + C_a$. For a circular cylinder in potential flow, $C_a = 1.0$, yielding $C_M = 2.0$. This value is generally stable but decreases slightly in drag-dominated regimes ($high KC$).

### 5.2 Kinematic Reconstruction

The Software calculates $u$ and $\dot{u}$ analytically from the Fourier series.

**Velocity ($u$):**
$$
u(x, z) = -(\bar{u} + c) + \sqrt{\frac{g}{k^3}} \sum_{j=1}^{N} jkB_j \frac{\cosh(jkz)}{\cosh(jkd)} \cos(jkX)
$$

**Acceleration ($a_x$):**
A critical detail often missed in simple solvers is the convective acceleration. In the Eulerian frame:
$$
a_x = \frac{Du}{Dt} = \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + w\frac{\partial u}{\partial z}
$$
For a steady wave traveling at speed $c$, the local time derivative is $\partial u / \partial t = -c (\partial u / \partial x)$. Thus:
$$
a_x = (u - c) \frac{\partial u}{\partial x} + w \frac{\partial u}{\partial z}
$$
The Software computes $\partial u / \partial x$ and $\partial u / \partial z$ by differentiating the cosine/cosh terms in the Fourier series, avoiding the discretization errors associated with finite difference differentiation of a velocity grid.

### 5.3 Exact Surface Integration

A major source of error in legacy codes is the "stretching" approximation, where kinematics are extrapolated from the Mean Water Level (MWL) to the crest. The Calculator performs Exact Surface Integration:

$$
F_{total}(t) = \int_{seabed}^{\eta(t)} \left( dF_D(z, t) + dF_I(z, t) \right) dz
$$

The integration limit $\eta(t)$ changes at every time step. This captures the large periodic loads in the splash zone, which are essentially "turned on and off" as the crest passes. Neglecting this zone can underestimate base shear by 15-20% in steep waves.

---

## 6. Hydrodynamic Loading II: Impulsive Breaking Wave Loads

While Morison's equation handles non-breaking waves well, it fails to predict the massive, short-duration impact loads generated when a wave breaks directly onto the pile. These "slamming" events are critical for ULS design as they can cause local buckling or excite high-frequency structural ringing modes.

### 6.1 Breaking Criteria

The Software automatically checks if the simulated wave is breaking using two criteria:

1.  **Geometric Limit:** Based on McCowan’s criterion for shallow water, if $H/d > 0.78$.
2.  **Kinematic Limit:** If the horizontal particle velocity at the crest exceeds the wave celerity ($u_{crest} > c$).

Notice the Calculator does NOT implements the **Wienke & Oumeraci (2005) Slamming Model**, which is necessary under breaking waves conditons, because the restriction H/d <= 0.6 is applied by the software.

### 6.2 Force Magnitude

The impact force should be modeled based on the von Karman and Wagner theories of water entry. The maximum slamming force $F_{slam\_max}$ occurs at the instant of impact ($t=0$):

$$
F_{slam\_max} = \lambda \eta_b C_s \rho R C^2
$$

**Parameters:**
* $\eta_b$: The breaking wave crest elevation.
* $C$: The wave celerity (impact velocity).
* $R$: Cylinder radius ($D/2$).
* $\lambda$ (Curling Factor): This empirical factor defines the vertical extent of the "flat" wave front that impacts the pile. Based on the large-scale experiments of Wienke & Oumeraci, $\lambda = 0.46$ for plunging breakers.
* $C_s$ (Slamming Coefficient): This coefficient relates the dynamic pressure to the impact velocity. Theoretical values are $\pi$ (von Karman, no pile-up) and $2\pi$ (Wagner, with pile-up). Experimental data shows scatter between 3.0 and 6.0.

### 6.3 Time History and Duration

The slamming force is an impulse. Its duration $T_{dur}$ is governed by the time it takes for the cylinder to be fully immersed by the passing wave front.

$$
T_{dur} = \frac{13 R}{32 C}
$$

This relationship, derived specifically for cylindrical piles by Wienke & Oumeraci, predicts durations on the order of milliseconds.

The time history is modeled as a Linear Decay:

$$
F_s(t) = F_{slam\_max} \left( 1 - \frac{t}{T_{dur}} \right) \quad \text{for} \quad 0 \le t \le T_{dur}
$$

$$
F_s(t) = 0 \quad \text{for} \quad t > T_{dur}
$$

In breaking conditions this triangular pulse should be superimposed onto the Morison force time series. The resulting load profile would feature a sharp, high-frequency spike near the crest phase, which is the signature of breaking wave impact.

---

### 6.4 Total Force and Moment

The precise determination of the total hydrodynamic load acting on a cylindrical pile is the culmination of the kinematic analysis. While local force densities ($f(z)$) provide insight into stress distribution, the global design of the foundation—specifically the **Base Shear ($F_{tot}$)** and **Overturning Moment ($M_{tot}$)**—requires a rigorous integration of these forces over the entire submerged length of the structure.

This software does the **Exact Surface Integration**, by integrating forces from the seabed ($z=0$) exactly up to the instantaneous free surface $\eta(t)$. This captures the significant contribution of the "splash zone"—the area between the MWL and the wave crest—where particle velocities and drag forces are at their maximum.

#### 6.4.1 Vertical Integration of Force Components

At any given instant in time $t$ (or phase $\phi$), the total horizontal force $F_{tot}(t)$ acting on the pile is the integral of the distributed force density $dF(z,t)$ over the wetted height of the pile.

The integration domain extends from the seabed ($z=0$) to the instantaneous water surface elevation $\eta(t)$:

$$
F_{tot}(t) = \int_{0}^{\eta(t)} \left[ dF_D(z, t) + dF_I(z, t) \right] dz
$$

Where:
* $z=0$ represents the seabed.
* $\eta(t)$ is the time-varying free surface elevation derived from the Fenton Stream Function solution.
* $dF_D$ and $dF_I$ are the Drag and Inertia force densities per unit vertical length ($N/m$), defined by the Morison equation.

**Breaking Down the Components:**

Substituting the Morison formulations (Eq. 5.1.1 and 5.1.2) into the integral yields the explicit form used by the solver:

$$
F_{tot}(t) = \int_{0}^{\eta(t)} \left[ \underbrace{\frac{1}{2} \rho C_D D_{eff} (u |u|)}_{\text{Drag Density}} + \underbrace{\rho C_M \frac{\pi D_{eff}^2}{4} \dot{u}}_{\text{Inertia Density}} \right] dz
$$

**Key Computational Details:**

1.  **Effective Diameter ($D_{eff}$):** The integration uses the hydrodynamically effective diameter, which accounts for marine growth thickness ($t_{mg}$):
    $$D_{eff} = D_{pile} + 2 \cdot t_{mg}$$
2.  **Velocity ($u$):** This is the **total horizontal fluid velocity** at height $z$, including both the wave orbital velocity and any steady current $U_c$. The term $u|u|$ ensures that the drag force correctly opposes the direction of fluid motion, even in reversing flows.
3.  **Acceleration ($\dot{u}$):** This represents the **total horizontal fluid acceleration**, incorporating both local ($\partial u / \partial t$) and convective ($(u \cdot \nabla)u$) components, which are significant in steep nonlinear waves.

#### 6.2.4 Calculation of Overturning Moment

The Overturning Moment ($M_{tot}$) is the tendency of the hydrodynamic forces to rotate the pile about a specific pivot point. For foundation design, this is typically calculated about the **mudline** (seabed, $z=0$).

The moment is calculated by integrating the force density multiplied by its lever arm (the vertical distance $z$ from the seabed):

$$
M_{tot}(t) = \int_{0}^{\eta(t)} z \cdot \left[ dF_D(z, t) + dF_I(z, t) \right] dz
$$

Expanding this using the Morison terms:

$$
M_{tot}(t) = \int_{0}^{\eta(t)} z \cdot \left[ \frac{1}{2} \rho C_D D_{eff} (u |u|) + \rho C_M \frac{\pi D_{eff}^2}{4} \dot{u} \right] dz
$$

**Why Exact Integration Matters for Moments:**

The contribution of the wave crest to the overturning moment is disproportionately large because of the **lever arm effect**. Forces generated in the splash zone (near $\eta_{max}$) act at the maximum possible distance from the seabed ($z_{max} \approx d + H/2$).
* **Linear Theory:** Often neglects the lever arm above $z=d$, significantly underestimating the moment.
* **Fenton's Method:** By integrating to $\eta(t)$, this solver captures the full moment arm, often resulting in design moments 20-30% higher than linear predictions for shallow, steep waves.

---

# 7. Software Operational Manual

This repository contains a high-precision hydrodynamics suite designed for analyzing nonlinear steady water waves and their interaction with cylindrical structures. The suite implements **Fenton’s Fourier Approximation Method (1988)**, providing machine-precision solutions to the nonlinear boundary value problem.

The suite consists of four distinct modules:
1.  **`script_cli.cpp`**: High-performance C++ CLI solver for batch processing.
2.  **`script.py`**: Reference Python implementation with graphical reporting.
3.  **`script_gui.cpp`**: Native Windows GUI for interactive analysis.
4.  **`fenton.py`**: Utility for calculating nonlinear wave properties.

---

## 7.1 Prerequisites & Dependencies

### For C++ Modules (`script_cli.cpp`, `script_gui.cpp`)
* **Compiler:** A C++17 compliant compiler (GCC, Clang, or MSVC).
* **OS:** Windows is required for `script_gui.cpp` (Win32 API). `script_cli.cpp` is cross-platform (Linux/macOS/Windows).
* **Libraries:** Standard Template Library (STL). No external math libraries (e.g., Eigen, Boost) are required; all linear algebra kernels are self-contained.

### For Python Modules (`script.py`, `fenton.py`)
* **Interpreter:** Python 3.8 or higher.
* **Libraries:**
    ```bash
    pip install numpy scipy matplotlib numba
    ```
    * **NumPy:** Core linear algebra and vectorization.
    * **SciPy:** Nonlinear optimization (`least_squares`) and root finding.
    * **Matplotlib:** Generation of PDF graphical reports.
	* **Numba:** JIT (Just-In-Time) compilation for accelerating mathematical kernels.

---

## 7.2 Module 1: C++ CLI Solver (`script_cli.cpp`)

This is the computational engine of the suite. It uses a custom Levenberg-Marquardt solver and complex-step differentiation to achieve convergence on the order of machine epsilon ($10^{-16}$).

### 7.2.1 Compilation Instructions

**Using g++ (MinGW/Linux):**
Optimize for the host architecture to ensure maximum speed during the Jacobian matrix inversion.
```bash
g++ -O3 -march=native -std=c++17 -Wall -Wextra -static -static-libgcc -static-libstdc++ -o script_cli.exe script_cli.cpp -lm
```


**Using MSVC (Windows):**
```cmd
cl /O2 /std:c++17 /EHsc script_cli.cpp
```

### 7.2.2 Operational Modes

The executable supports two modes of operation:

**A. Interactive Mode:**
Run the executable without arguments. The program will prompt for inputs sequentially.
```bash
./script_cli.exe
```

**B. Command Line Argument Mode (Batch):**
Pass parameters directly for automated batch processing or sensitivity analysis.
```bash
./script_cli.exe [H] [T] [d] [Uc] [Type] [Dia] [Mg] [Cd] [Cm]
```

| Parameter | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `H` | float | Wave Height (m) | `3.0` |
| `T` | float | Wave Period (s) | `9.0` |
| `d` | float | Water Depth (m) | `5.0` |
| `Uc` | float | Current Velocity (m/s) | `1.0` |
| `Type` | string | Current Type ("Eulerian" only) | `Eulerian` |
| `Dia` | float | Pile Diameter (m) | `1.5` |
| `Mg` | float | Marine Growth Thickness (m) | `0.05` |
| `Cd` | float | Drag Coefficient | `1.3` |
| `Cm` | float | Inertia Coefficient | `2.0` |

### 7.2.3 Outputs
* **Console Output:** Real-time convergence logs and final force summaries.
* **output.txt:** A detailed log file containing the Fourier coefficients ($B_j$), derived integral properties (Energy, Power, Impulse), and the full force distribution profile.

---

## 7.3 Module 2: Python Reference Solver (`script.py`)

This script mirrors the C++ logic but utilizes `scipy.optimize` and `numpy` for implementation ease. Its distinct feature is the generation of vector-graphic reports.

### 7.3.1 Usage
Execute the script via the terminal:
```bash
python script.py
```
Follow the interactive prompts to define the wave environment and Morison coefficients.

### 7.3.2 Output Files
The script generates two artifacts in the working directory:

**1. `output.txt` (Data Log)**
* **Convergence History:** Tracks the residual error at each homotopy step.
* **Integral Quantities:** Lists calculated properties like Radiation Stress ($S_{xx}$), Group Velocity ($C_g$), and Energy Flux ($P$).
* **Force Profile:** Tabular data of velocity $u(z)$, acceleration $\dot{u}(z)$, and force density $f(z)$ at the phase of maximum load.

**2. `plots.pdf` (Visual Report)**
A multi-page PDF containing:
* **Plot 1:** Wave Kinematics (Surface Elevation and Velocity profiles ).
* **Plot 2:** Wave Dynamics (Acceleration envelopes and Dynamic Pressure profiles).
* **Plot 3:** Solver Validation (Comparison of Nonlinear Fenton vs. Linear Airy profiles).
* **Plot 4:**: 2D Field Visualization (Contour plots of Velocity magnitude and Dynamic Pressure).
* **Plot 5:** Force Spectrum (FFT analysis) and Harmonic Reconstruction.
* **Plot 6:** Vertical Pressure Profiles (Comparison of Hydrostatic/Dynamic Fluid Pressures vs. Equivalent Drag/Inertia Force Pressures).
* **Plot 7:** Time-histories of Total Base Shear and Overturning Moment (including individual Drag/Inertia components).

---

## 7.4 Module 3: Windows GUI (`script_gui.cpp`)

A standalone Windows application that wraps the C++ solver in a user-friendly interface using the Win32 API.

### 7.4.1 Compilation Instructions (MinGW-w64)
You must link against the Windows subsystem and Unicode libraries.
```bash
g++ -O3 -std=c++17  -Wall -Wextra -static -static-libgcc -static-libstdc++ -o script_gui.exe script_gui.cpp -mwindows -lgdi32

```


### 7.4.2 Usage Guide
1.  **Launch:** Execute `script_gui.exe`.
2.  **Input:** Fill in the Environmental and Structural fields.
    * *Note:* All fields must be positive numbers (except Current, which can be negative for opposing flows).
3.  **Execute:** Click the **"CALCULATE FORCES"** button.
4.  **Results:** A complete text report is rendered in the right-hand output window. This content is simultaneously saved to `output.txt`.

---

## 7.5 Module 4: Linear vs. Nonlinear Utility (`fenton.py`)

This utility is strictly for hydrodynamic verification. It runs the Fenton solver alongside exact analytical Linear Wave Theory to quantify the "Nonlinearity Error".

### 7.5.1 Usage
```bash
python fenton.py
```
The script prompts for $H$, $T$, $d$, and $U_c$.

### 7.5.2 Output Interpretation
The script prints a side-by-side comparison table to the console (and `output.txt`) contrasting:
* **Wavelength ($L$):** Nonlinear waves are generally longer than linear waves in shallow water.
* **Crest Elevation ($\eta_{crest}$):** Nonlinear crests are higher and sharper ($> H/2$).
* **Trough Elevation ($\eta_{trough}$):** Nonlinear troughs are flatter and shallower ($< H/2$).
* **Breaking Status:** Evaluates stability against the Miche Criterion:
    $$\frac{H}{L} \le 0.142 \tanh(kd)$$

---

## 7.6 Troubleshooting & Theory Limits

### 7.6.1 Convergence Failure
If the solver fails to converge (Residual $> 10^{-5}$):
1.  **Check Steepness:** The wave may be physically unstable (breaking). Ensure $H/d \le 0.6$ (the software's hard limit) and $H/L \le 0.14$.
2.  **Increase Homotopy Steps:** Increase `DEF_HOMOTOPY_STEPS` (default 5) to 10 or 20. This forces the solver to take smaller increments in wave height.
3.  **Check Current:** High opposing currents ($U_c < 0$) can cause singularity in the wavenumber calculation. Ensure the wave can physically propagate against the current.

### 7.6.2 Physics Interpretation
* **Residual Norm:** Found in `output.txt`. It should be $< 10^{-10}$ for a valid solution.
* **Stokes Drift:** If "Mean Stokes Drift" is non-zero, mass is being transported. This is a key difference from Linear Theory and vital for sediment transport analysis.
* **Force Asymmetry:** In shallow water, Drag forces ($F_D \propto u|u|$) increase drastically at the crest due to particle velocity asymmetry. Linear theory will significantly underpredict this load.

---

## 8. Discussion on Accuracy and Limitations

### 8.1 Comparison with Other Theories

* **Vs. Linear Theory:** The Calculator yields significantly higher velocities in the crest (up to 30-50% higher for steep waves) due to the proper handling of nonlinear boundary conditions. This results in much higher drag forces.
* **Vs. Stokes 5th:** Fenton's method is superior in shallow water where Stokes theory often creates spurious oscillations ("wiggles") in the profile. Fenton's method remains monotonic and physically consistent.

### 8.2 Known Limitations

* **Regular Waves Only:** The stream function approach assumes a periodic, steady wave train. It cannot model transient focused wave groups (e.g., "NewWave") or irregular sea states directly, although it is often used to model the largest wave in a train.
* **Vertical Piles Only:** The integrated force algorithms are specific to vertical cylinders. The Software does not currently support inclined piles, scour analysis, or wave-in-deck (Kaplan) loads, despite their theoretical relevance to the broader field.
* **Diffraction:** For large diameter structures (e.g., Gravity Based Structures where $D/L > 0.2$), the Morison equation is invalid. A diffraction solver (MacCamy-Fuchs or BEM) would be required.

---

## 9. Conclusion

The development of the **Wave Forces on Cylindrical Piles Calculator** represents an advancement in the precision of structural analysis for offshore monopiles. By deliberately transcending the simplifications of linear wave theory, this tool adopts the **Fenton Fourier Approximation** to accurately model the non-linear kinematics inherent in high-energy ocean environments. This transition is critical for capturing essential physical phenomena—such as crest peaking, mass transport, and flow asymmetry—that define **extreme sea states** where traditional methods often underestimate fluid velocities.

A key achievement of this Calculator is the integration of the **Trust Region Reflective numerical solver**. This algorithm addresses the critical challenge of computational stability often associated with high-order stream function theories. By navigating the complex optimization landscape with high efficiency, the solver prevents numerical divergence where standard Newton-Raphson methods typically fail, ensuring no loss of data during critical design scenarios.

Ultimately, the Calculator above described provides engineers with a **rigorous basis for design**, offering a more realistic load profile than Airy theory, particularly in shallow water regimes. By combining high-fidelity physical modeling with a stable numerical framework, the tool allows for the adequate prediction of forces bellow the **wave-breaking singularity**, facilitating safer, more cost-efficient, and optimized monopile designs under **maximum load conditions**.

---

## Parameters Glossary

This glossary provides a definition of all variables and parameters used in the software. The variables are categorized by their function: **Input Parameters** (user-defined), **Internal State Variables** (solver-specific), **Derived Kinematics**, and **Force Calculation Variables**.

### 1. Input Parameters (User Configuration)
These variables are located in the main execution block and define the physical simulation case.

**Water Depth** ($d$) [m]
The vertical distance from the Mean Water Level (MWL) to the seabed. It defines the domain height for the boundary value problem. In the code, the seabed is treated as $z=0$ and the MWL as $z=d$.

**Design Wave Height** ($H$) [m]
The vertical distance between the wave trough and the wave crest. This is the primary driver of nonlinearity; as $H$ increases, the wave profile distorts, becoming sharper at the crest.

**Wave Period** ($T$) [s]
The time interval between two successive crests passing a fixed point. It determines the fundamental frequency of the wave ($\omega = 2\pi/T$) and is used to calculate the celerity.

**Eulerian Current** ($U_c$) [m/s]
A steady, uniform background current superimposed on the wave field.
* Positive values indicate a current following the wave (Doppler shift increases apparent period).
* Negative values indicate an opposing current.

**Current Type**
Defines the reference frame for the background current. 'Eulerian' (default) superimposes a uniform velocity profile defined at a fixed point, while 'Stokes' implies a mass transport velocity.

**Fluid Density** ($\rho$) [kg/m³]
The mass per unit volume of the water. Default is typically 1025 kg/m³ for seawater. This scales all calculated forces linearly.

**Pile Diameter** ($D$) [m]
The outer diameter of the cylindrical structure. It is used to calculate the frontal area ($D$) for drag and the displaced volume ($\pi D^2/4$) for inertia loads.

**Marine Growth** $t_{mg}$ [m]
Bio-fouling thickness. Increases Effective Diameter ($D+2t_{mg}$) and Surface Roughness ($k/D$).

**Gravitational Acceleration** ($g$) [m/s²]
A global constant, typically set to 9.81 m/s². It provides the restoring force for the gravity waves.

### 2. Numerical & Solver Parameters
These parameters control the accuracy, stability, and execution of the **Fenton Fourier Approximation** solver.

**Fourier Truncation Order** ($N$)
The number of Fourier modes (cosine terms) used to approximate the stream function $\psi$.
* **Low $N$ (5-10):** Sufficient for deep water ($d/L > 1/2$).
* **High $N$ (20-50):** Required for shallow water ($d/L < 1/20$) to capture sharp crests without "ringing" (Gibbs phenomenon).

**Number of Surface Nodes** ($M$)
The number of discrete points along the free surface (from crest to trough) where the boundary conditions are enforced. Typically set to $M \ge N+1$ to create an overdetermined system, solved via least-squares.

**Homotopy Steps** ($n$)
The number of incremental steps used to ramp up the wave height from a linear seed ($H \approx 0$) to the target height $H$. This "continuation method" prevents solver divergence by ensuring the initial guess for step $i+1$ is always within the "basin of attraction" of the previous solution.

**State Vector** ($\mathbf{Z}$)
The primary unknown vector optimized by the `scipy.optimize.least_squares` solver. It contains variables required to define the wave field:
1.  `Z[0]`: Wavenumber $k$.
2.  `Z[1:M+2]`: Surface elevations $\eta$ at the discrete nodes.
3.  `Z[M+2:-2]`: The array of $N$ dimensionless Fourier coefficients ($B_j$).
4.  `Z[-2]`: Flux $Q$ (mass transport).
5.  `Z[-1]`: Bernoulli Constant $R$ (energy head).

### 3. Hydrodynamic Force Coefficients
Empirical factors used to calibrate the **Morison Equation**.

**Drag Coefficient** ($C_D$) [Typical: 0.7 - 1.3]
Scales the viscous force component. It depends on surface roughness (marine growth) and the Reynolds number. Higher values are used for rough/fouled piles.

**Inertia Coefficient** ($C_M$) [Typical: 2.0]
Scales the mass-acceleration force. $C_M = 1 + C_a$, where $C_a$ is the added mass coefficient. For a cylinder in potential flow, $C_a=1.0$, so $C_M=2.0$.

### 4. Calculated Internal Variables
Variables computed during the solution process (Solver Output).

**Wave Length** ($L$)
The horizontal distance between two successive crests, derived from the wavenumber ($L = 2\pi/k$).

**Wavenumber** ($k$)
Calculated as $2\pi/L$. It relates the spatial scale of the wave to the water depth ($kd$) and determines the decay rate of velocities with depth.

**Celerity** ($c$)
The phase speed of the wave. Calculated from the solution variables to satisfy the dispersion relation. Crucial for determining if a wave is breaking ($u_{crest} > c$).

**Bernoulli Constant** ($R$)
The total energy head of the flow, constant along the free surface in the steady frame.

**Mass Flux** ($Q$)
The volume flux per unit width (discharge) under the wave in the moving reference frame.

**Free Surface Elevation** ($\eta$)
Array representing the elevation at each time step. It defines the instantaneous wetted length of the pile.

**Particle Velocities** ($u, w$)
Arrays representing the **Horizontal** and **Vertical particle velocities** at a specific grid point $(x,z)$. Derived analytically from the Stream Function $\psi$ derivatives.

**Accelerations** ($a_x, a_z$)
Particle accelerations. Included is the convective acceleration term ($u \partial u/\partial x$), which is significant in steep waves.

**Keulegan-Carpenter Number** ($KC$)
A dimensionless parameter ($u_{orbital} T / D_{eff}$) quantifying the relative importance of drag over inertia for oscillatory flow.

**Reynolds Number** ($Re$)
A dimensionless ratio of inertial forces to viscous forces ($u_{crest} D_{eff} / \nu$), used to determine the drag coefficient.

**Ursell Number** ($Ur$)
A dimensionless parameter ($H L^2 / d^3$) indicating the degree of nonlinearity in shallow water.

### 5. Output Force Variables
The final results of the simulation.

**Drag Force** (`F_drag`)
The quasi-static drag force time series, proportional to $u|u|$.

**Inertia Force** (`F_inertia`)
The quasi-static inertia force time series, proportional to $\dot{u}$ (local + convective).

**Total Force** (`F_total`)
The sum of Drag and Inertia (calculator does NOT includes Slamming) forces at each time step.

**Base Shear** (`Base_Shear`)
The integral of `F_total` over the submerged length of the pile.

**Overturning Moment** (`Overturning_Moment`)
The integral of `F_total * z` (force $\times$ lever arm) from the seabed.

**Center of Effort**
The effective elevation (relative to the seabed or MSL) at which the total hydrodynamic force is assumed to act.

**Max Local Force Density**
The maximum force per unit length (kN/m) occurring at any specific elevation along the pile.

---

## References

**THEORETICAL BASIS (FENTON: STREAM FUNCTION & KINEMATICS)**

1.  Fenton, J.D. (1999). "Numerical methods for nonlinear waves." In P.L.-F. Liu (Ed.), *Advances in Coastal and Ocean Engineering* (Vol. 5, pp. 241–324). World Scientific: Singapore. [Primary Source: Comprehensive review of fully-nonlinear methods including Fourier approximation]. URL: https://johndfenton.com/Papers/Fenton99Liu-Numerical-methods-for-nonlinear-waves.pdf
2.  Fenton, J.D. (1988). "The numerical solution of steady water wave problems." *Computers & Geosciences*, 14(3), 357–368. [The core algorithm for high-accuracy Stream Function Theory]. URL: https://doi.org/10.1016/0098-3004(88)90066-0
3.  Fenton, J.D. (1985). "A fifth-order Stokes theory for steady waves." *Journal of Waterway, Port, Coastal, and Ocean Engineering*, 111(2), 216–234. [Standard analytical theory for deep/intermediate water pile design]. URL: https://doi.org/10.1061/(ASCE)0733-950X(1985)111:2(216)
4.  Fenton, J.D. (1978). "Wave forces on vertical bodies of revolution." *Journal of Fluid Mechanics*, 85(2), 241–255. [Foundational diffraction theory for large diameter piles]. URL: https://johndfenton.com/Papers/Fenton78-Waves-on-bodies-of-revolution.pdf
5.  Fenton, J.D. (1990). "Nonlinear wave theories." In B. Le Méhauté & D.M. Hanes (Eds.), *The Sea: Ocean Engineering Science* (Vol. 9, Part A). John Wiley & Sons. [Comprehensive guide for selecting wave theories: Stokes vs Cnoidal vs Stream]. URL: https://www.johndfenton.com/Papers/Fenton90b-Nonlinear-wave-theories.pdf

**HOCINE OUMERACI (BREAKING WAVE IMPACT, SLAMMING & RINGING)**

6.  Wienke, J., & Oumeraci, H. (2005). "Breaking wave impact force on a vertical and inclined slender pile—theoretical and large-scale model investigations." *Coastal Engineering*, 52(5), 435–462. [CRITICAL: Separates quasi-static (Morison) from dynamic (slamming) forces]. URL: https://doi.org/10.1016/j.coastaleng.2004.12.008
7.  Irschik, K., Sparboom, U., & Oumeraci, H. (2004). "Breaking wave loads on a slender pile in shallow water." *Proceedings of the 29th ICCE*, 4, 3968–3980. [Focuses on shallow water impacts where Stream Function may reach limits]. URL: https://www.worldscientific.com/doi/abs/10.1142/9789812701916_0045?srsltid=AfmBOoqJdBABjd1aTI4ZExsN095TJCbKX79yiG0ve1zBoRd1IW1VhYfS
8.  Kortenhaus, A., & Oumeraci, H. (1998). "Classification of wave loading on monolithic coastal structures." *Proceedings of the 26th ICCE*, 1, 867–879. [Defines transition zones between pulsating and impulsive load regimes]. URL: https://icce-ojs-tamu.tdl.org/icce/article/download/5654/5324/0
9.  Muttray, M., & Oumeraci, H. (2005). "Theoretical and experimental study on wave damping inside a perforated caisson." *Ocean Engineering*, 32(14), 1803–1818. [Relevant for piles with scour protection or permeable outer layers]. URL: https://www.sciencedirect.com/science/article/abs/pii/S0378383905000591

**ENGINEERING MANUALS & STANDARDS**

10. U.S. Army Corps of Engineers (USACE). (2002). "Coastal Engineering Manual (CEM)." *Engineer Manual 1110-2-1100*. Washington, D.C. [The modern successor to the SPM; standard for wave mechanics]. URL: https://www.publications.usace.army.mil/USACE-Publications/Engineer-Manuals/u43544q/636F617374616C20656E67696E656572696E67206D616E75616C/
11. U.S. Army Corps of Engineers (USACE). (1984). "Shore Protection Manual (SPM)." Vol. I & II. 4th Edition. CERC, Vicksburg, MS. [Classic reference; still widely used for historical comparison and empirical data]. URL: https://usace.contentdm.oclc.org/digital/collection/p16021coll11/id/1934/
12. CIRIA, CUR, CETMEF. (2007). "The Rock Manual. The Use of Rock in Hydraulic Engineering." (2nd Edition). *C683, CIRIA, London*. [Standard for pile scour protection design and rock interaction]. URL: https://www.ciria.org/ItemDetail?iProductCode=C683
13. DNV (Det Norske Veritas). (2014). "Environmental Conditions and Environmental Loads." *Recommended Practice DNV-RP-C205*. [Industry standard for offshore pile design and Morison coefficients]. URL: https://www.dnv.com/energy/standards-guidelines/dnv-rp-c205-environmental-conditions-and-environmental-loads/

**TEXTBOOKS (WAVE MECHANICS & FORCES)**

14. Sumer, B. M., & Fredsøe, J. (2006). *Hydrodynamics Around Cylindrical Structures*. (Revised Edition). World Scientific. [The 'Bible' for flow around piles, vortex shedding, and scour]. URL: https://doi.org/10.1142/6248
15. Sarpkaya, T., & Isaacson, M. (1981). *Mechanics of Wave Forces on Offshore Structures*. Van Nostrand Reinhold. [Classic text on diffraction and inertia/drag regimes]. URL: https://www.amazon.com/-/pt/dp/0521896258/
16. Goda, Y. (2010). *Random Seas and Design of Maritime Structures*. (3rd Edition). World Scientific. [Essential for spectral analysis and statistical design of piles]. URL: https://doi.org/10.1142/7425
17. Dean, R. G., & Dalrymple, R. A. (1991). *Water Wave Mechanics for Engineers and Scientists*. World Scientific. [Foundational pedagogy for linear and nonlinear wave theory]. URL: https://doi.org/10.1142/1232