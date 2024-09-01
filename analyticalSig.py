import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import matplotlib.animation as animation
from potential import AnalyticalSignal

# Set up logging
logging.basicConfig(filename='logs/pso_analytical_signal.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info("PSO Optimization for Analytical Signal started")

x = np.arange(-150, 150)
Xo = 89
h = 20
signalo = np.empty((300, 1))
for i in range(0, 300):
    signalo[i, 0] = AnalyticalSignal(x[i], Xo, h)

def Error(k):
    error = np.sqrt(np.sum(np.square(k-signalo), axis=0)*(1/300))
    return error

noOfRuns = 0
found_solution = False
positions_over_time = []  # To store positions at each iteration
velocities_over_time = []  # To store velocities at each iteration

# Create a grid for contour plotting
x_grid = np.linspace(-150, 150, 100)
h_grid = np.linspace(5, 50, 100)
X, H = np.meshgrid(x_grid, h_grid)
Z = np.array([[AnalyticalSignal(x_val, Xo, h_val) for x_val in x_grid] for h_val in h_grid])

while not found_solution:
    noOfIterations = np.array([])
    errorG = np.array([])
    noOfRuns += 1
    logger.info(f"Starting run {noOfRuns}")

    position = np.empty((2, 20))
    velocity = np.zeros((2, 20))  # Initialize velocities
    for k in range(0, 20):
        position[0, k] = -150 + np.random.random() * 300
        position[1, k] = 5 + np.random.random() * 45
    l_best = position.copy()
    signal = np.empty((300, 20))
    L = np.empty((300, 20))
    G = np.empty((300, 1))
    U = 2 * position - 1
    g = 500 * np.ones((2, 1))

    # PSO loop with logging and progress tracking
    for c in tqdm(range(1, 401), desc=f'Run {noOfRuns}'):
        # Store the current positions and velocities for the animation
        positions_over_time.append(position.copy())
        velocities_over_time.append(U.copy())  # Store velocities, not the updated ones

        # Log the current positions
        logger.info(f"Iteration {c} positions: {position}")

        # Updating signal values
        for k in range(0, 20):
            for i in range(0, 300):
                signal[i, k] = AnalyticalSignal(x[i], position[0, k], position[1, k])
                L[i, k] = AnalyticalSignal(x[i], l_best[0, k], l_best[1, k])
        for i in range(0, 300):
            G[i, 0] = AnalyticalSignal(x[i], g[0, 0], g[1, 0])

        # Updating local best (l_best)
        for i in range(0, 20):
            if Error(signal)[i] <= Error(L)[i]:
                l_best[:, i] = position[:, i]

        # Finding global best (g_best)
        if np.min(Error(signal)) <= Error(G):
            g[:, 0] = position[:, np.argmin(Error(signal))]

        # Updating parameters
        U = 0.1 * U - 2 * np.random.random((2, 20)) * (position - l_best) - 2 * np.random.random((2, 20)) * (position - g)
        position += U

        # Update velocities
        velocity = U.copy()

        # Checking range
        position[0, :] = np.clip(position[0, :], -150, 150)
        position[1, :] = np.clip(position[1, :], 5, 50)

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
logger.info(f'Original parameters: x = {Xo}, h = {h}')
logger.info(f'Final parameters: x = {g[0, 0]}, h = {g[1, 0]}')
logger.info(f'Final Error: {Error(G)}')
logger.info("PSO Optimization completed")

# Print results
print(f'Total number of iterations: {(noOfRuns - 1) * 400 + c - 1}')
print(f'Original parameters: x = {Xo}, h = {h}')
print(f'Final parameters: x = {g[0, 0]}, h = {g[1, 0]}')
print(f'Final Error: {Error(G)}')

# Plot Error vs No. of iterations
plt.plot(noOfIterations, errorG)
plt.xlabel('No. of iterations')
plt.ylabel('Error')
plt.title('Error vs No. of iterations')
plt.savefig('images/Error_Analytical.png', dpi=300)

# Plot Analytical Signal
signalans = np.empty((300, 1))
for i in range(0, 300):
    signalans[i, 0] = AnalyticalSignal(x[i], g[0, 0], g[1, 0])
plt.plot(x, signalo, label='Actual', color='yellow')
plt.plot(x, signalans, label='PSO', color='black', linestyle='dashed')
plt.title('Analytical Signal')
plt.xlabel('x')
plt.ylabel('Signal')
plt.legend()
plt.savefig('images/AnalyticalSignal.png', dpi=300)

# Create an animation showing particle movement with contour map and arrows
fig, ax = plt.subplots()
ax.set_xlim(-150, 150)
ax.set_ylim(5, 50)
contour = ax.contourf(X, H, Z, levels=20, cmap='viridis', alpha=0.7)
scat = ax.scatter([], [], s=100, color='blue')
quiver = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color='yellow')

def update(frame):
    positions = positions_over_time[frame]
    velocities = velocities_over_time[frame]
    
    # Update scatter plot
    scat.set_offsets(np.c_[positions[0], positions[1]])
    
    # Update quiver plot
    if len(positions) > 0 and len(velocities) > 0:
        quiver.set_offsets(np.c_[positions[0], positions[1]])
        quiver.set_UVC(velocities[0, :], velocities[1, :])
    
    return scat, quiver

ani = animation.FuncAnimation(fig, update, frames=len(positions_over_time), blit=True, repeat=False)
ani.save('images/particle_movement_Analytical.gif', writer='imagemagick')