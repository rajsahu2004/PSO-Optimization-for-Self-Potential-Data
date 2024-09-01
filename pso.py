import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
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
Xo = 30
h = 10
theta = 29
Vo = np.empty((300, 1))
for i in range(0, 300):
    Vo[i, 0] = PotentialSphere(x[i], Xo, h, theta)

noOfRuns = 0
found_solution = False

while not found_solution:
    noOfIterations = np.array([])
    errorG = np.array([])
    noOfRuns += 1
    logger.info(f"Starting run {noOfRuns}")

    position = np.empty((3, 20))
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
plt.savefig('Error.png', dpi=300)
plt.show()

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
plt.savefig('SP_Profile.png', dpi=300)
plt.show()