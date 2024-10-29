import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
pi = np.pi

# Inverse power potential parameter
n = 4

# Packing fractions
eta_cp = 0.7405  # Close-packed density for fcc lattice

# Define functions for the fluid phase (gas and liquid) as before

def Z_HS(eta):
    """Compressibility factor from Carnahan-Starling equation."""
    return (1 + eta + eta**2 - eta**3) / (1 - eta)**3

def pressure(eta, T_star):
    """Total pressure for fluid phases."""
    Z_hs = Z_HS(eta)
    P_hs = eta * T_star * Z_hs
    a = (2 * pi) / (n - 3)
    P_attr = - a * eta**2
    return P_hs + P_attr

def chemical_potential(eta, T_star):
    """Chemical potential for fluid phases."""
    mu_id = T_star * (np.log(eta) + 1)
    f_ex = T_star * eta * (4 - 3 * eta) / (1 - eta)**2
    mu_ex = f_ex + T_star * eta * (8 - 9 * eta + 3 * eta**2) / ( (1 - eta)**3 )
    a = (2 * pi) / (n - 3)
    fA = - a * eta
    mu_attr = fA - a * eta
    return mu_id + mu_ex + mu_attr

# Functions for the solid phase

def Z_HS_solid(eta):
    """Compressibility factor for hard-sphere solid."""
    return (1 + eta + eta**2 - eta**3) / (1 - eta)**3

def pressure_solid(eta, T_star):
    """Total pressure for solid phase."""
    Z_hs = Z_HS_solid(eta)
    P_hs = eta * T_star * Z_hs
    a_solid = (2 * pi) / (n - 3)
    P_attr = - a_solid * eta**2
    return P_hs + P_attr

def chemical_potential_solid(eta, T_star):
    """Chemical potential for solid phase."""
    mu_id = T_star * (np.log(eta) + 1)
    f_ex = T_star * eta * (4 - 3 * eta) / (1 - eta)**2
    mu_ex = f_ex + T_star * eta * (8 - 9 * eta + 3 * eta**2) / ( (1 - eta)**3 )
    a_solid = (2 * pi) / (n - 3)
    fA = - a_solid * eta
    mu_attr = fA - a_solid * eta
    return mu_id + mu_ex + mu_attr

# Temperature range
T_min = 0.3
T_max = 1.0
T_star_range = np.linspace(T_min, T_max, 50)

# Arrays to store coexistence data
eta_solidus = []
eta_fluidus = []
T_solidus = []

for T_star in T_star_range:
    # Function to find solid-fluid coexistence
    def solid_fluid_coexistence(e):
        eta_f, eta_s = e
        P_f = pressure(eta_f, T_star)
        mu_f = chemical_potential(eta_f, T_star)
        P_s = pressure_solid(eta_s, T_star)
        mu_s = chemical_potential_solid(eta_s, T_star)
        return [P_f - P_s, mu_f - mu_s]
    
    # Initial guesses
    eta_f_guess = 0.5  # Starting near the liquid density
    eta_s_guess = 0.6  # Starting in the solid density range
    e_guess = [eta_f_guess, eta_s_guess]
    
    try:
        e_sol = fsolve(solid_fluid_coexistence, e_guess)
        eta_f_sol, eta_s_sol = e_sol
        # Ensure solutions are within physical bounds
        if 0.4 < eta_f_sol < 0.6 and 0.5 < eta_s_sol < eta_cp:
            eta_fluidus.append(eta_f_sol)
            eta_solidus.append(eta_s_sol)
            T_solidus.append(T_star)
    except Exception:
        pass

# Convert lists to arrays
T_solidus = np.array(T_solidus)
eta_fluidus = np.array(eta_fluidus)
eta_solidus = np.array(eta_solidus)

# Plotting the phase diagram
plt.figure(figsize=(8,6))

# Plot solid-fluid coexistence
plt.plot(eta_fluidus, T_solidus, 'm-', label='Fluid Phase at Solid-Fluid Coexistence')
plt.plot(eta_solidus, T_solidus, 'g-', label='Solid Phase at Solid-Fluid Coexistence')

# Existing code for gas-liquid coexistence
# Density range for fluid
eta_fluid = np.linspace(0.01, 0.5, 2000)

# Arrays to store gas-liquid coexistence data
eta_gas = []
eta_liquid = []
T_coexist = []

for T_star in T_star_range:
    # Calculate pressure and chemical potential for fluid densities
    P_fluid = pressure(eta_fluid, T_star)
    mu_fluid = chemical_potential(eta_fluid, T_star)
    
    # Identify the gas-liquid coexistence as before
    dP_deta = np.gradient(P_fluid, eta_fluid)
    inflection_points = np.where(np.diff(np.sign(dP_deta)))[0]
    
    if len(inflection_points) >= 2:
        idx1 = inflection_points[0]
        idx2 = inflection_points[-1]
        
        def coexistence(e):
            e1, e2 = e
            P1 = pressure(e1, T_star)
            mu1 = chemical_potential(e1, T_star)
            P2 = pressure(e2, T_star)
            mu2 = chemical_potential(e2, T_star)
            return [P1 - P2, mu1 - mu2]
        
        e_guess = [eta_fluid[idx1], eta_fluid[idx2]]
        
        try:
            e_sol = fsolve(coexistence, e_guess)
            if 0 < e_sol[0] < e_sol[1] < 0.5:
                eta_gas.append(e_sol[0])
                eta_liquid.append(e_sol[1])
                T_coexist.append(T_star)
        except Exception:
            pass

# Convert lists to arrays and sort
T_coexist = np.array(T_coexist)
eta_gas = np.array(eta_gas)
eta_liquid = np.array(eta_liquid)
T_critical = T_coexist[-1]
# Corresponding densities
eta_gas_critical = eta_gas[-1]
eta_liquid_critical = eta_liquid[-1]
# Compute midpoint
eta_critical = (eta_gas_critical + eta_liquid_critical) / 2
# Plot gas-liquid coexistence
plt.plot(eta_gas, T_coexist, 'b-', label='Gas Phase')
plt.plot(eta_liquid, T_coexist, 'r-', label='Liquid Phase')
plt.plot([eta_critical], [T_critical], 'ko', label='Critical Point', markersize=10)

# Mark the solid region
plt.fill_betweenx(T_solidus, eta_solidus, eta_cp, color='gray', alpha=0.3, label='Solid Phase')

plt.xlabel('Reduced Density, $\\eta$')
plt.ylabel('Reduced Temperature, $T^*$')
plt.title('Phase Diagram including Solid Phase for $n=4$ Inverse Power Potential')
plt.legend()
plt.grid(True)
plt.show()
