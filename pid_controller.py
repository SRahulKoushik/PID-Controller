import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.optimize import minimize, differential_evolution

# Defining the given system: G(s) = 1 / (s^3 + 3s^2 + 5s + 1)
numerator = [1]
denominator = [1, 3, 5, 1]
G = ctrl.TransferFunction(numerator, denominator)

# Defining simulation parameters
time = np.linspace(0, 10, 1000)

def pid_control(error, prev_error, integral, Kp, Ki, Kd, dt, dead_zone=0.01):
    if abs(error) < dead_zone:
        return 0, integral  # No control action within the dead zone
    integral += error * dt
    derivative = (error - prev_error) / dt if prev_error is not None else 0
    u = Kp * error + Ki * integral + Kd * derivative
    return u, integral

def low_pass_filter(y, alpha=0.1):
    if not hasattr(low_pass_filter, 'prev_value'):
        low_pass_filter.prev_value = y
    filtered = alpha * y + (1 - alpha) * low_pass_filter.prev_value
    low_pass_filter.prev_value = filtered
    return filtered

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
    return (100 * overshoot) + (5 * settling_time) + (100 * steady_state_error) + (0.1 * control_effort) + (50 * noise_sensitivity)

# Optimization bounds
bounds = [(0.1, 100), (0.1, 100), (0.1, 3)]  # Limits for Kp, Ki, and Kd

# Nelder-Mead optimization
result_nm = minimize(pid_cost, [0.1, 0.01, 0.01], bounds=bounds, method='Nelder-Mead')
Kp_opt_nm, Ki_opt_nm, Kd_opt_nm = result_nm.x

# Genetic Algorithm optimization
result_ga = differential_evolution(pid_cost, bounds, strategy='best1bin', maxiter=500, popsize=20, tol=0.01)
Kp_opt_ga, Ki_opt_ga, Kd_opt_ga = result_ga.x

# Selecting the best result based on the lowest cost
costs = {
    'Nelder-Mead': pid_cost([Kp_opt_nm, Ki_opt_nm, Kd_opt_nm]),
    'Genetic Algorithm': pid_cost([Kp_opt_ga, Ki_opt_ga, Kd_opt_ga])
}
best_method = min(costs, key=costs.get)
Kp_best, Ki_best, Kd_best = {
    'Nelder-Mead': (Kp_opt_nm, Ki_opt_nm, Kd_opt_nm),
    'Genetic Algorithm': (Kp_opt_ga, Ki_opt_ga, Kd_opt_ga)
}[best_method]

# Root Locus Analysis
plt.figure()
ctrl.root_locus(G)
plt.title("Root Locus of Open-Loop System")
plt.show()

# Bode Plot Analysis
plt.figure()
ctrl.bode(G, dB=True)
plt.title("Bode Plot of Open-Loop System")
plt.show()

PID_best = ctrl.TransferFunction([Kd_best, Kp_best, Ki_best], [1, 0])
system_best = ctrl.feedback(PID_best * G)
t, y_best = ctrl.step_response(system_best, time)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, y_best, label="Optimized PID Response")
plt.axhline(1, color="r", linestyle="--", label="Reference")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Optimized PID Step Response Using Genetic Algorithm / Nelson-Mead")
plt.legend()
plt.grid()
plt.show()

# State-space matrices (controllable canonical form)
A = np.array([[0, 1, 0], [0, 0, 1], [-1, -5, -3]])
B = np.array([0, 0, 1])
C = np.array([1, 0, 0])
D = 0

# Simulation parameters for second part
dt = 0.005
t = np.arange(0, 5, dt)
n = len(t)

# Initialize states and PID terms
x = np.array([0.0, 0.0, 0.0])
integral = 0.0
prev_error = None
umin, umax = -20, 20
noise_std = 0.05

# Storage for outputs
y_out = np.zeros(n)
u_out = np.zeros(n)
ref = np.ones(n)

for i in range(n):
    r = ref[i]
    y = x[0] + np.random.normal(0, noise_std)
    y_filtered = low_pass_filter(y)
    error = r - y_filtered
    u, integral = pid_control(error, prev_error, integral, Kp_best, Ki_best, Kd_best, dt, dead_zone=0.01)
    u_sat = np.clip(u, umin, umax)
    u_out[i] = u_sat
    dxdt = np.array([x[1], x[2], -x[0] - 5*x[1] - 3*x[2] + u_sat])
    x += dxdt * dt
    y_out[i] = x[0]
    prev_error = error

plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.plot(t, y_out, label='Output')
plt.plot(t, ref, 'r--', label='Reference')
plt.title('System Response')
plt.ylabel('Output')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, u_out, 'g', label='Control Signal')
plt.title('Control Signal')
plt.xlabel('Time (s)')
plt.ylabel('Control')
plt.legend()
plt.tight_layout()
plt.show()





