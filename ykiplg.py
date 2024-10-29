import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
num_particles = 10  # Number of particles
box_size = 10.0  # Size of the 2D box
kappa = 1.0  # Inverse screening length (Yukawa parameter)
epsilon = 1.0  # Interaction strength

# Define temperature and density ranges
temperatures = np.linspace(0.5, 3.0, 10)  # Range of temperatures
densities = np.linspace(0.1, 1.0, 10)  # Range of densities (particles/box area)
phase_data = np.zeros((len(temperatures), len(densities)))  # Store average energies

# Yukawa potential function
def yukawa_potential(r, kappa, epsilon):
    return epsilon * np.exp(-kappa * r) / r if r > 0 else 0

# Monte Carlo step
def monte_carlo_step(positions, box_size, temperature, kappa, epsilon):
    i = np.random.randint(len(positions))  # Random particle
    old_position = positions[i].copy()
    
    # Random move
    delta_move = (np.random.rand(2) - 0.5) * 0.1
    positions[i] += delta_move
    positions[i] %= box_size  # Periodic boundary conditions
    
    # Calculate energy change
    old_energy = total_potential_energy(positions, box_size, kappa, epsilon)
    new_energy = total_potential_energy(positions, box_size, kappa, epsilon)
    dE = new_energy - old_energy
    
    # Accept or reject move
    if dE > 0 and np.random.rand() > np.exp(-dE / temperature):
        positions[i] = old_position  # Reject move
    return new_energy

# Total potential energy calculation
def total_potential_energy(positions, box_size, kappa, epsilon):
    energy = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            # Periodic boundary conditions
            r_vec = positions[i] - positions[j]
            r_vec -= box_size * np.round(r_vec / box_size)
            r = np.linalg.norm(r_vec)
            energy += yukawa_potential(r, kappa, epsilon)
    return energy

# Simulation loop for phase diagram generation
for t_idx, temperature in enumerate(temperatures):
    for d_idx, density in enumerate(densities):
        # Set up initial particle positions based on density
        num_particles = int(density * box_size**2)
        positions = np.random.rand(num_particles, 2) * box_size
        
        # Run Monte Carlo to equilibrate
        energy_accum = 0.0
        num_steps = 1000
        for step in range(num_steps):
            energy = monte_carlo_step(positions, box_size, temperature, kappa, epsilon)
            if step > 500:  # Skip initial steps (equilibration)
                energy_accum += energy
        
        # Average energy as a proxy for phase information
        phase_data[t_idx, d_idx] = energy_accum / (num_steps - 500)

# Plotting phase diagram
plt.imshow(phase_data, extent=(densities[0], densities[-1], temperatures[0], temperatures[-1]),
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label="Average Potential Energy")
plt.xlabel("Density")
plt.ylabel("Temperature")
plt.title("Phase Diagram of Yukawa Particles")
plt.show()
