import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.optimize import differential_evolution

# System Model: Robotic Arm Dynamics
J, B, Kt, Ke = 0.01, 0.1, 0.01, 0.01  # Inertia, damping, torque constant, back EMF constant
num = [Kt]
den = [J, B, Kt*Ke]
G = ctrl.TransferFunction(num, den)

time = np.linspace(0, 5, 1000)  # 5-second simulation

# PID Control with Dead Zone and Anti-Windup
def pid_control(error, prev_error, integral, Kp, Ki, Kd, dt, umin=-10, umax=10, dead_zone=0.01):
    if abs(error) < dead_zone:
        return 0, integral
    integral += error * dt
    derivative = (error - prev_error) / dt if prev_error is not None else 0
    u = Kp * error + Ki * integral + Kd * derivative
    if u > umax:
        u = umax
        integral -= error * dt  # Anti-windup
    elif u < umin:
        u = umin
        integral -= error * dt  # Anti-windup
    return u, integral

# Low-pass filter for noisy measurements
def low_pass_filter(y, alpha=0.1):
    if not hasattr(low_pass_filter, 'prev_value'):
        low_pass_filter.prev_value = y
    filtered = alpha * y + (1 - alpha) * low_pass_filter.prev_value
    low_pass_filter.prev_value = filtered
    return filtered

# Cost function for PID optimization
def pid_cost(params):
    Kp, Ki, Kd = params
    PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    closed_loop = ctrl.feedback(PID * G)
    t, y = ctrl.step_response(closed_loop, time)
    
    overshoot = np.max(y) - 1
    settling_time = t[np.where(np.abs(y - 1) < 0.02)[0][-1]] if np.any(np.abs(y - 1) < 0.02) else 5
    steady_state_error = np.abs(y[-1] - 1)
    noise_sensitivity = np.var(np.diff(y))
    control_effort = np.sum(np.abs(np.diff(y)))
    
    return (100 * overshoot) + (5 * settling_time) + (100 * steady_state_error) + (0.1 * control_effort) + (50 * noise_sensitivity)

# Optimization
bounds = [(0.1, 100), (0.1, 100), (0.1, 3)]
result_ga = differential_evolution(pid_cost, bounds, strategy='best1bin', maxiter=500, popsize=20, tol=0.01)
Kp_opt, Ki_opt, Kd_opt = result_ga.x

print(f"Optimized PID Parameters: Kp={Kp_opt:.4f}, Ki={Ki_opt:.4f}, Kd={Kd_opt:.4f}")

# Simulation with optimized PID
dt = 0.005
t = np.arange(0, 5, dt)
n = len(t)
x = np.array([0.0])  # Initial state (position)
integral = 0.0
prev_error = None
umin, umax = -10, 10  # Actuator limits
noise_std = 0.05

y_out = np.zeros(n)
u_out = np.zeros(n)
ref = np.ones(n)  # Step reference

for i in range(n):
    r = ref[i]
    y = x[0] + np.random.normal(0, noise_std)
    y_filtered = low_pass_filter(y)
    error = r - y_filtered
    u, integral = pid_control(error, prev_error, integral, Kp_opt, Ki_opt, Kd_opt, dt, umin, umax, dead_zone=0.02)
    dxdt = (-B/J) * x[0] + (Kt/J) * u
    x[0] += dxdt * dt
    y_out[i] = x[0]
    u_out[i] = u
    prev_error = error

# Plot Results
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t, y_out, label='Robotic Arm Position')
plt.plot(t, ref, 'r--', label='Reference')
plt.title('Robotic Arm Position Control')
plt.ylabel('Position')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, u_out, 'g', label='Control Signal')
plt.title('Control Effort')
plt.xlabel('Time (s)')
plt.ylabel('Torque Input')
plt.legend()

plt.tight_layout()
plt.show()

