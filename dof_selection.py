import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.optimize import differential_evolution

# Defining the given system: G(s) = 1 / (s^3 + 3s^2 + 5s + 1)
numerator = [1]
denominator = [1, 3, 5, 1]
G = ctrl.TransferFunction(numerator, denominator)

# Defining simulation parameters
time = np.linspace(0, 10, 1000)  # Simulates for 10 seconds

# PID cost function
def pid_cost(params):
    Kp, Ki, Kd = params
    PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    closed_loop = ctrl.feedback(PID * G)
    t, y = ctrl.step_response(closed_loop, time)
    overshoot = np.max(y) - 1
    settling_time = t[np.where(np.abs(y - 1) < 0.01)[0][-1]] if np.any(np.abs(y - 1) < 0.01) else 10
    steady_state_error = np.abs(y[-1] - 1)
    noise_sensitivity = np.var(np.diff(y))
    control_effort = np.sum(np.abs(np.diff(y)))
    cost = (100 * overshoot) + (5 * settling_time) + (100 * steady_state_error) + (0.1 * control_effort) + (50 * noise_sensitivity)
    return cost

# Optimization bounds
bounds = [(0.1, 100), (0.1, 100), (0.1, 3)]

# Genetic Algorithm optimization
result_ga = differential_evolution(pid_cost, bounds, strategy='best1bin', maxiter=500, popsize=20, tol=0.01)
Kp_best, Ki_best, Kd_best = result_ga.x

# Define Multi-Degree-of-Freedom PID Control
I_PD = ctrl.TransferFunction([Ki_best], [1, 0])
PI_D = ctrl.TransferFunction([Kp_best, Ki_best], [1, 0]) + ctrl.TransferFunction([Kd_best, 0], [1])
PID_best = ctrl.TransferFunction([Kd_best, Kp_best, Ki_best], [1, 0])

# Closed-loop systems
system_PID = ctrl.feedback(PID_best * G)
system_I_PD = ctrl.feedback(I_PD * G)
system_PI_D = ctrl.feedback(PI_D * G)

# Step responses
t_pid, y_PID = ctrl.step_response(system_PID, time)
t_ipd, y_I_PD = ctrl.step_response(system_I_PD, time)
t_pid, y_PI_D = ctrl.step_response(system_PI_D, time)

# Plot comparisons
plt.figure(figsize=(10, 6))
plt.plot(t_pid, y_PID, label="PID Response")
plt.plot(t_ipd, y_I_PD, label="I-PD Response")
plt.plot(t_pid, y_PI_D, label="PI-D Response")
plt.axhline(1, color="r", linestyle="--", label="Reference")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Comparison of PID, I-PD, and PI-D Responses")
plt.legend()
plt.grid()
plt.show()

