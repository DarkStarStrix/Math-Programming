import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

# Set random seed for reproducibility
np.random.seed (42)

# Define problem parameters
grid_size = 100
num_stores = 20
num_residential_areas = 10
num_particles = 50
num_iterations = 100
convergence_threshold = 1e-6
convergence_patience = 50

# PSO parameters
w = 0.5
c1 = 1
c2 = 2


# Helper functions (unchanged)
def generate_random_locations(num_locations):
    return np.random.rand (num_locations, 2) * grid_size


def distance(p1, p2):
    return np.sqrt (np.sum ((p1 - p2) ** 2))


def calculate_fitness(position, stores, residential_areas):
    store_distances = [distance (position, store) for store in stores]
    residential_distances = [distance (position, area) for area in residential_areas]
    penalty = sum (max (0, 15 - d) * 2000 for d in residential_distances)
    return sum (store_distances) + penalty


# Generate random store and residential area locations
stores = generate_random_locations (num_stores)
residential_areas = generate_random_locations (num_residential_areas)

# Initialize particles
particles = generate_random_locations (num_particles)
velocities = np.zeros_like (particles)
personal_best_positions = particles.copy ()
personal_best_fitness = np.array ([calculate_fitness (p, stores, residential_areas) for p in particles])
global_best_position = personal_best_positions [np.argmin (personal_best_fitness)]
global_best_fitness = np.min (personal_best_fitness)

# Create heatmap (unchanged)
resolution = 100
x = np.linspace (0, grid_size, resolution)
y = np.linspace (0, grid_size, resolution)
X, Y = np.meshgrid (x, y)

Z = np.zeros ((resolution, resolution))
for i in range (resolution):
    for j in range (resolution):
        Z [i, j] = calculate_fitness ([X [i, j], Y [i, j]], stores, residential_areas)

# Set up the main plot (unchanged)
fig_main, ax_main = plt.subplots (figsize=(10, 8))
colors = ['darkblue', 'blue', 'lightblue', 'yellow', 'orange', 'red']
cmap = LinearSegmentedColormap.from_list ('custom', colors, N=1000)

im = ax_main.imshow (Z, extent=[0, grid_size, 0, grid_size], origin='lower', cmap=cmap, alpha=0.7)
plt.colorbar (im, ax=ax_main, label='Fitness (lower is better)')

store_plot = ax_main.scatter (stores [:, 0], stores [:, 1], c='limegreen', s=100, label='Stores', edgecolors='black',
                              linewidth=2)
residential_plot = ax_main.scatter (residential_areas [:, 0], residential_areas [:, 1], c='red', s=150,
                                    label='Residential Areas', marker='s', edgecolors='black', linewidth=2)
particles_plot = ax_main.scatter (particles [:, 0], particles [:, 1], c='cyan', s=30, alpha=0.5, label='Particles')
best_position_plot = ax_main.scatter ([], [], c='white', s=200, marker='*', label='Best Position', edgecolors='black',
                                      linewidth=2)

ax_main.set_title ('Warehouse Location Optimization', fontsize=14, fontweight='bold')
ax_main.set_xlabel ('X coordinate', fontsize=12)
ax_main.set_ylabel ('Y coordinate', fontsize=12)
ax_main.legend (fontsize=10)
ax_main.grid (True, linestyle='--', alpha=0.7)

# Set up the convergence plot (modified)
fig_convergence, ax_convergence = plt.subplots (figsize=(10, 6))
convergence_line, = ax_convergence.plot ([], [], 'b-', linewidth=2, label='Best Fitness')
average_fitness_line, = ax_convergence.plot ([], [], 'r-', linewidth=2, label='Average Fitness')
ax_convergence.set_xlim (0, num_iterations)
ax_convergence.set_title ('Convergence of Fitness', fontsize=14, fontweight='bold')
ax_convergence.set_xlabel ('Iteration', fontsize=12)
ax_convergence.set_ylabel ('Fitness', fontsize=12)
ax_convergence.legend (fontsize=10)
ax_convergence.grid (True, linestyle='--', alpha=0.7)

best_fitness_history = [global_best_fitness]
average_fitness_history = [np.mean (personal_best_fitness)]


# Animation update function
def update(frame):
    global particles, velocities, personal_best_positions, personal_best_fitness, global_best_position, global_best_fitness

    for i in range (num_particles):
        fitness = calculate_fitness (particles [i], stores, residential_areas)

        if fitness < personal_best_fitness [i]:
            personal_best_fitness [i] = fitness
            personal_best_positions [i] = particles [i]

        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particles [i]

        r1, r2 = np.random.rand (2)
        velocities [i] = (w * velocities [i] +
                          c1 * r1 * (personal_best_positions [i] - particles [i]) +
                          c2 * r2 * (global_best_position - particles [i]))
        particles [i] += velocities [i]
        particles [i] = np.clip (particles [i], 0, grid_size)

    particles_plot.set_offsets (particles)
    best_position_plot.set_offsets ([global_best_position])

    best_fitness_history.append (global_best_fitness)
    average_fitness_history.append (np.mean (personal_best_fitness))

    # Update convergence plot
    ax_convergence.clear ()
    ax_convergence.plot (best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
    ax_convergence.plot (average_fitness_history, 'r-', linewidth=2, label='Average Fitness')
    ax_convergence.set_xlim (0, num_iterations)
    ax_convergence.set_title (f'Convergence of Fitness (Iteration {frame + 1})', fontsize=14, fontweight='bold')
    ax_convergence.set_xlabel ('Iteration', fontsize=12)
    ax_convergence.set_ylabel ('Fitness', fontsize=12)
    ax_convergence.legend (fontsize=10)
    ax_convergence.grid (True, linestyle='--', alpha=0.7)

    ax_main.set_title (f'Warehouse Location Optimization (Iteration {frame + 1})', fontsize=14, fontweight='bold')

    print (
        f"Iteration {frame + 1}: Best Fitness = {global_best_fitness}, Average Fitness = {np.mean (personal_best_fitness)}")

    if len (best_fitness_history) > convergence_patience:
        recent_improvements = np.diff (best_fitness_history [-convergence_patience:])
        if np.all (np.abs (recent_improvements) < convergence_threshold):
            print ("Convergence reached. Stopping optimization.")
            anim_main.event_source.stop ()
            anim_convergence.event_source.stop ()

    return particles_plot, best_position_plot


# Create animations
anim_main = FuncAnimation (fig_main, update, frames=num_iterations, interval=200, blit=False, repeat=False)
anim_convergence = FuncAnimation (fig_convergence, update, frames=num_iterations, interval=200, blit=False,
                                  repeat=False)

# Save the animations as GIFs
writer_main = PillowWriter (fps=5)
writer_convergence = PillowWriter (fps=5)
anim_main.save ('warehouse_optimization_main.gif', writer=writer_main)
anim_convergence.save ('warehouse_optimization_convergence.gif', writer=writer_convergence)

plt.show ()

print (f"Final optimal warehouse location: {global_best_position}")
print (f"Final optimal fitness value: {global_best_fitness}")
