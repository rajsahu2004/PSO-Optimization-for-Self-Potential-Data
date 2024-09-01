# PSO Optimization for Self-Potential Data

[![Watching](https://img.shields.io/github/watchers/rajsahu2004/Anomaly-Detection?color=brightgreen)](https://github.com/rajsahu2004/Anomaly-Detection/watchers)
[![Stars](https://img.shields.io/github/stars/rajsahu2004/Anomaly-Detection?color=yellow)](https://github.com/rajsahu2004/Anomaly-Detection/stargazers)
[![Forks](https://img.shields.io/github/forks/rajsahu2004/Anomaly-Detection?color=blue)](https://github.com/rajsahu2004/Anomaly-Detection/network/members)

For an in-depth explanation of the methods and results, check out my Medium blog: [Optimizing Self-Potential Models with Particle Swarm Optimization: A Geophysical Approach](https://medium.com/@sahuraj457/optimizing-self-potential-models-with-particle-swarm-optimization-a-geophysical-approach-d83065f7fa62)


## Overview

This project involves optimizing the parameters of a self-potential model using Particle Swarm Optimization (PSO). The self-potential method is used in geophysics to detect anomalies in subsurface materials. The goal of this project is to find the optimal parameters that best fit a self-potential profile using PSO.

## Project Structure

1. **`analyticalSig.py`**: Contains the main script for running the PSO optimization, generating plots, and creating a GIF of particle movements.

2. **`potential.py`**: Defines the potential functions used in the optimization:
   - `PotentialSphere`: Calculates the potential due to a sphere.
   - `PotentialCylinder`: Calculates the potential due to a cylinder.
   - `AnalyticalSignal`: Calculates the analytical signal.

3. **`logs/pso_log.log`**: Log file that records the details of the PSO optimization process.

4. **`images/`**: Directory where results are saved, including:
   - `Error PSO.png`: Plot of error vs. number of iterations.
   - `SP_Profile.png`: Plot of self-potential profile.
   - `particle_movement_PSO.gif`: GIF showing particle movements during optimization.

## Installation

Ensure you have the following Python packages installed:

- `numpy`
- `matplotlib`
- `tqdm`
- `imageio`

You can install these packages using pip:

```bash
pip install numpy matplotlib tqdm imageio
```

## Usage

<ol>
    <li>Run the Optimization:
    Execute the analyticalSig.py script to start the PSO optimization process. This script will:
    </li>
    <ul>
        <li>Initialize the PSO parameters.
        <li>Optimize the parameters using PSO.
        <li>Save the results and generate plots and GIFs.
    </ul>

```bash
pip install numpy matplotlib tqdm imageio
```
<li>View Results:
After running the script, the following files will be generated in the images/ directory:
    <ul>
        <li>Error PSO.png: Shows the error vs. the number of iterations.
	    <li>SP_Profile.png: Compares the actual self-potential profile with the optimized profile.
	    <li>particle_movement_PSO.gif: Visualizes the particle movements during the optimization process.
    </ul>
</ol>

## Description

### PSO Optimization
<ul>
<li>Initialization: Particles are initialized randomly within a specified range.
<li>Velocity Update: Particles’ velocities are updated based on their personal best and the global best positions.
<li>Position Update: Particles’ positions are updated based on their velocities.
<li>Error Calculation: The error is calculated as the root mean square difference between the predicted and actual self-potential values.
<li>Stopping Criteria: The optimization stops when the error falls below a predefined threshold.
</ul>

### Potential Functions

<ul>
<li>PotentialSphere: Calculates the self-potential due to a sphere.
<li>PotentialCylinder: Calculates the self-potential due to a cylinder.
<li>AnalyticalSignal: Calculates the analytical signal for comparison.
</ul>

### Logging

The optimization process is logged in `logs/pso_log.log`. The log file contains information about the optimization runs, including:

<ul>
<li>The number of iterations.
<li>Parameters used in each run.
<li>Error values.
</ul>

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

The self-potential model and optimization techniques used in this project are based on established geophysical methods and optimization algorithms.

## Contact

For any questions or feedback, please reach out to:

**Raj Sahu**  
Email: [sahuraj457@gmail.com](mailto:sahuraj457@gmail.com)