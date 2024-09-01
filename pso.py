import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import imageio
from potential import PotentialSphere

# Set up logging
logging.basicConfig(filename='logs/pso_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info("PSO Optimization started")

def Error(k):
    error = np.sqrt(np.sum(np.square(k - Vo), axis=0) * (1 / 300))
    return error

x = np.arange(-150, 150)
Xo = -79
h = 23
theta = 47
Vo = np.empty((300, 1))
for i in range(0, 300):
    Vo[i, 0] = PotentialSphere(x[i], Xo, h, theta)

noOfRuns = 0
found_solution = False
particle_positions = []  # To store positions of particles for the GIF
particle_velocities = []  # To store velocities for the arrows

# Create a grid for contour plotting
x_grid = np.linspace(-150, 150, 100)
h_grid = np.linspace(5, 50, 100)
X, H = np.meshgrid(x_grid, h_grid)
Z = np.array([[PotentialSphere(x_val, Xo, h_val, theta) for x_val in x_grid] for h_val in h_grid])

while not found_solution:
    noOfIterations = np.array([])
    errorG = np.array([])
    noOfRuns += 1
    logger.info(f"Starting run {noOfRuns}")

    position = np.empty((3, 20))
    velocity = np.zeros((3, 20))  # Initialize velocities
    for k in range(0, 20):
        position[0, k] = -150 + np.random.random() * 300
        position[1, k] = 5 + np.random.random() * 45
        position[2, k] = np.random.random() * 90
    l_best = position.copy()
    V = np.empty((300, 20))
    L = np.empty((300, 20))
    G = np.empty((300, 1))
    U = 2 * position - 1
    g = 500 * np.ones((3, 1))

    # PSO with progress bar
    for c in tqdm(range(1, 401), desc=f'Run {noOfRuns}'):
        # Record particle positions and velocities for GIF
        particle_positions.append(position.copy())
        particle_velocities.append(velocity.copy())

        # Finding and updating V
        for k in range(0, 20):
            for i in range(0, 300):
                V[i, k] = PotentialSphere(x[i], position[0, k], position[1, k], position[2, k])
                L[i, k] = PotentialSphere(x[i], l_best[0, k], l_best[1, k], l_best[2, k])
        for i in range(0, 300):
            G[i, 0] = PotentialSphere(x[i], g[0, 0], g[1, 0], g[2, 0])

        # Updating l_best
        for i in range(0, 20):
            if Error(V)[i] <= Error(L)[i]:
                l_best[:, i] = position[:, i]

        # Finding g_best
        if np.min(Error(V)) <= Error(G):
            g[:, 0] = position[:, np.argmin(Error(V))]

        # Updating parameters
        U = 0.1 * U - 2 * np.random.random((3, 20)) * (position - l_best) - 2 * np.random.random((3, 20)) * (position - g)
        position += U

        # Checking range
        position[0, :] = np.clip(position[0, :], -150, 150)
        position[1, :] = np.clip(position[1, :], 5, 50)
        position[2, :] = np.clip(position[2, :], 0, 90)

        # Update velocities
        velocity = U.copy()

        noOfIterations = np.append(noOfIterations, c)
        errorG = np.append(errorG, Error(G))
        
        if Error(G) < 0.001:
            logger.info(f"Solution found in run {noOfRuns} after {c} iterations")
            found_solution = True
            break

    if found_solution:
        break

# Log final results
logger.info(f'Total number of iterations: {(noOfRuns - 1) * 400 + c - 1}')
logger.info(f'Original parameters: x = {Xo}, h = {h}, theta = {theta}')
logger.info(f'Final parameters: x = {g[0, 0]}, h = {g[1, 0]}, theta = {g[2, 0]}')
logger.info(f'Final Error: {Error(G)}')
logger.info("PSO Optimization completed")

# Print results
print(f'Total number of iterations: {(noOfRuns - 1) * 400 + c - 1}')
print(f'Original parameters: x = {Xo}, h = {h}, theta = {theta}')
print(f'Final parameters: x = {g[0, 0]}, h = {g[1, 0]}, theta = {g[2, 0]}')
print(f'Final Error: {Error(G)}')

# Plot Error vs No. of iterations
plt.plot(noOfIterations, errorG)
plt.xlabel('No. of iterations')
plt.ylabel('Error')
plt.title('Error vs No. of iterations')
plt.savefig('images/Error PSO.png', dpi=300)

# Plot Self Potential Profile
Vans = np.empty((300, 1))
for i in range(0, 300):
    Vans[i, 0] = PotentialSphere(x[i], g[0, 0], g[1, 0], g[2, 0])
plt.plot(x, Vo * 1000, label='Actual', color='yellow')
plt.plot(x, Vans * 1000, label='PSO', color='black', linestyle='dashed')
plt.title('Self Potential Profile')
plt.xlabel('x')
plt.ylabel('V (mV)')
plt.legend()
plt.savefig('images/SP_Profile.png', dpi=300)

# Create GIF of particle movements with contour map and arrows
filenames = []
for i, (positions, velocities) in enumerate(zip(particle_positions, particle_velocities)):
    plt.figure(figsize=(8, 6))
    
    # Plot contour map
    plt.contourf(X, H, Z, levels=20, cmap='viridis', alpha=0.7)
    
    # Plot particle positions
    plt.scatter(positions[0, :], positions[1, :], c='blue', label='Particles')
    plt.scatter(g[0, 0], g[1, 0], c='red', label='Global Best')

    # Add arrows to represent velocities
    plt.quiver(positions[0, :], positions[1, :], velocities[0, :], velocities[1, :], 
               angles='xy', scale_units='xy', scale=1, color='yellow')

    plt.xlim(-150, 150)
    plt.ylim(5, 50)
    plt.xlabel('x')
    plt.ylabel('h')
    plt.title(f'Iteration {i + 1}')
    plt.legend()
    
    # Save each frame as a temporary file
    filename = f'images/iteration_{i}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Generate GIF from frames
with imageio.get_writer('images/particle_movement_PSO.gif', mode='I', duration=0.2) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Cleanup temporary files
import os
for filename in filenames:
    os.remove(filename)

logger.info("GIF creation completed")