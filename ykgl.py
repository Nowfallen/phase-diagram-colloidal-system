import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline

# Constants
pi = np.pi
kappa = 1.0  # Adjusted inverse screening length

# Define functions for the fluid phase (gas and liquid)
def Z_HS(eta):
    """Compressibility factor from Carnahan-Starling equation."""
    return (1 + eta + eta**2 - eta**3) / (1 - eta)**3

def pressure(eta, T_star):
    """Total pressure for fluid phases with Yukawa potential."""
    Z_hs = Z_HS(eta)
    P_hs = eta * T_star * Z_hs
    a = (4 * pi) / kappa**2
    P_attr = - (a / 2) * eta**2
    return P_hs + P_attr

def chemical_potential(eta, T_star):
    """Chemical potential for fluid phases with Yukawa potential."""
    mu_id = T_star * (np.log(eta) + 1)
    f_ex = T_star * eta * (4 - 3 * eta) / (1 - eta)**2
    mu_ex = f_ex + T_star * eta * (8 - 9 * eta + 3 * eta**2) / ((1 - eta)**3)
    a = (4 * pi) / kappa**2
    mu_attr = - a * eta
    return mu_id + mu_ex + mu_attr

# Temperature range
T_min = 0.5
T_max = 1.5
T_star_range = np.linspace(T_min, T_max, 100)  # Increased resolution
eta_fluid = np.linspace(0.01, 0.5, 2000)

# Arrays to store gas-liquid coexistence data
eta_gas = []
eta_liquid = []
T_coexist = []

for T_star in T_star_range:
    P_fluid = pressure(eta_fluid, T_star)
    mu_fluid = chemical_potential(eta_fluid, T_star)
    dP_deta = np.gradient(P_fluid, eta_fluid)
    inflection_points = np.where(np.diff(np.sign(dP_deta)))[0]
    inflection_points = inflection_points[(inflection_points > 0) & (inflection_points < len(eta_fluid)-1)]

    if len(inflection_points) >= 2:
        idx1, idx2 = inflection_points[0], inflection_points[-1]

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

# Convert lists to arrays
T_coexist = np.array(T_coexist)
eta_gas = np.array(eta_gas)
eta_liquid = np.array(eta_liquid)

if len(T_coexist) > 0:
    sorted_indices = np.argsort(T_coexist)
    T_coexist = T_coexist[sorted_indices]
    eta_gas = eta_gas[sorted_indices]
    eta_liquid = eta_liquid[sorted_indices]

    T_critical = T_coexist[-1]
    eta_critical = (eta_gas[-1] + eta_liquid[-1]) / 2
    spline_gas = UnivariateSpline(T_coexist, eta_gas, k=3, s=0)
    spline_liquid = UnivariateSpline(T_coexist, eta_liquid, k=3, s=0)
    T_fit = np.linspace(T_coexist[0], T_critical, 500)

    plt.figure(figsize=(8, 6))
    plt.plot(eta_gas, T_coexist, 'bo', markersize=3)
    plt.plot(eta_liquid, T_coexist, 'ro', markersize=3)
    plt.plot(spline_gas(T_fit), T_fit, 'b-', label='Phase de Gaz')
    plt.plot(spline_liquid(T_fit), T_fit, 'r-', label='Phase de Liquide')
    plt.plot([eta_critical], [T_critical], 'ko', label='Point Critique', markersize=10)

    plt.xlabel('Densité Réduite, $\\eta$')
    plt.ylabel('Température Réduite, $T^*$')
    plt.title('Diagramme de Phase Gaz-Liquide pour le Potentiel de Yukawa ($\\kappa = {}$)'.format(kappa))
    plt.legend()
    plt.grid(True)

    # Save plot directly as a PDF
    pdf_output_path = "Yukawa_Phase_Diagram.pdf"
    plt.savefig(pdf_output_path, format='pdf')
    print("PDF saved at:", pdf_output_path)

    plt.show()
else:
    print("No coexistence points were found. Try adjusting the parameters.")
