import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from scipy.integrate import ode

# Gravitational Constant

G = 6.67430e-11

def acceleration(m,r):
    a = np.zeros_like(r)
    for i in range(3):
        for j in range(i+1, 3):
            r_ij = r[j] - r[i]
            dist = np.linalg.norm(r_ij)
            f = G * m[i] * m[j] / (dist**2 + 1e-10)**1.5
            a[i] += f * r_ij
            a[j] -= f * r_ij
    return a / m[:, np.newaxis]        

# Function to compute the derivatives (velocity and acceleration)
def derivatives(t, y, m):
    r = y[:9].reshape(3,3)
    v = y[9:].reshape(3,3)
    a = acceleration(m,r)
    return np.concatenate((v.flatten(), a.flatten()))

#Global Variables to store entries
masses = []
positions = []
velocities = []

# Initialize the simulation
def run_simulation():
    global masses, positions, velocities
    
    # Get Masses
    m = np.array([float(mass_entry.get()) for mass_entry in masses])
    
    # Get positions and velocities
    r = np.array([[float(entry.get()) for entry in body_pos] for body_pos in positions])
    v = np.array([[float(entry.get()) for entry in body_vel] for body_vel in velocities])

    y0 = np.concatenate((r.flatten(), v.flatten()))

    solver = ode(derivatives).set_integrator('dopri5', nsteps = 1000)
    solver.set_initial_value(y0, 0).set_f_params(m)

    t_end = 365.25 * 24 * 3600
    dt = t_end / 500  # 500 points for the entire year

    t_points = []
    y_points = []

    while solver.successful() and solver.t < t_end:
        solver.integrate(solver.t + dt)
        t_points.append(solver.t)
        y_points.append(solver.y)

    y_points = np.array(y_points)
    r_sol = y_points[:, :9].reshape(-1, 3, 3)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    max_dist = np.max(np.abs(r_sol))
    ax.set_xlim([-max_dist, max_dist])
    ax.set_ylim([-max_dist, max_dist])
    
    bodies = [ax.plot([], [], 'o', markersize=s, label=l)[0] 
              for s, l in zip([10, 5, 3], ['Sun', 'Earth', 'Moon'])]

    
    def update(frame):
        for body, r in zip(bodies, r_sol[frame]):
            body.set_data([r[0]], [r[1]])
        return bodies

    ani = FuncAnimation(fig, update, frames=len(t_points), interval=50, blit=False)
    ax.legend()
    plt.title("Sun-Earth-Moon System Simulation (Simplified 2D View)")
    plt.xlabel("Position X (m)")
    plt.ylabel("Position Y (m)")
    plt.grid(True)
    plt.show()

# Tkinter GUI setup
root = tk.Tk()
root.title("3-Body Problem Simulation")

# Mass inputs
masses = []
for i in range(3):
    tk.Label(root, text=f"Mass {i+1} (kg): ").grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    masses.append(entry)

# Position & Velocity inputs
velocities = []
positions = []
for i in range(3):
    body_positions = []
    body_velocities = []
    for j,axis in enumerate(['X','Y','Z']):
        row = 3 + i * 6 + j
        tk.Label(root, text=f"Body {i+1} Pos {axis} (m): ").grid(row=row, column=0)
        pos_entry = tk.Entry(root)
        pos_entry.grid(row=row, column=1)
        body_positions.append(pos_entry)

        tk.Label(root, text=f"Body {i+1} Vel {axis} (m/s): ").grid(row=row, column=2)
        vel_entry = tk.Entry(root)
        vel_entry.grid(row=row, column=3)
        body_velocities.append(vel_entry)

    positions.append(body_positions)
    velocities.append(body_velocities)

# Run button to start the simulation
run_button = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_button.grid(row=21, column=0, columnspan=4)

root.mainloop()