import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Set up Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11

# Function definitions
def mu_harv(t, T=365, u_h_avg=0.01, A=0.5):
    if 200 <= t % T <= 300:
        return u_h_avg * (1 + A * np.sin(2 * np.pi * t / T))
    else:
        return 0

def r_p_seasonal(t, r_p, T=365):
    if 100 <= t % T <= 250:
        return r_p * (1 + 0.5 * np.sin(2 * np.pi * t / T))
    else:
        return r_p * 0.5

def migration(t, T=365):
    if 0 <= t % T <= 180:
        return 0.02 * np.sin(2 * np.pi * t / T)
    else:
        return -0.01 * np.sin(2 * np.pi * t / T)

def forest_to_farm(y, t, params):
    P, H, C = y
    r_p, K_p, alpha = params[:3]
    r_H, beta_c, u_p = 0.5381, 0.2108, 0.3007
    r_C, delta = 0.2, 0.20
    
    dPdt = r_p_seasonal(t, r_p) * P * (1 - P / K_p) - alpha * H * P - mu_harv(t) * P
    dHdt = alpha * P * H - r_H * H - beta_c * C * H - u_p * H
    dCdt = r_C * H * C - delta * C + migration(t)
    
    return [dPdt, dHdt, dCdt]

# Parameters for different scenarios
scenarios = {
    '10%': [0.540, 110, 0.187],
    '30%': [0.621, 130, 0.162],
    '50%': [0.702, 150, 0.137],
    '70%': [0.783, 170, 0.111],
    '90%': [0.864, 190, 0.086]
}

# Time points
t = np.linspace(0, 365, 1000)
y0 = [10, 10, 10]

# Create figure
plt.figure(figsize=(12, 8))

# Colors for different components
colors = {
    'Crops': ['#1f77b4', '#a6cee3', '#4292c6', '#08519c', '#08306b'],
    'Pests': ['#d62728', '#fb9a99', '#ef3b2c', '#cb181d', '#99000d'],
    'Bats': ['#2ca02c', '#b2df8a', '#41ab5d', '#238b45', '#005a32']
}

# Line styles for different scenarios
line_styles = ['-', '--', ':', '-.', '-']

# Plot for each scenario
for idx, (label, params) in enumerate(scenarios.items()):
    solution = odeint(forest_to_farm, y0, t, args=(params,))
    P, H, C = solution.T
    
    plt.plot(t, P, color=colors['Crops'][idx], linestyle=line_styles[idx], 
             label=f'Crops ({label})', linewidth=1.5)
    plt.plot(t, H, color=colors['Pests'][idx], linestyle=line_styles[idx], 
             label=f'Pests ({label})', linewidth=1.5)
    plt.plot(t, C, color=colors['Bats'][idx], linestyle=line_styles[idx], 
             label=f'Bats ({label})', linewidth=1.5)

plt.xlabel('Time (days)', fontsize=12)
plt.ylabel('Population (k)', fontsize=12)
plt.title('AEDM Ecosystem Dynamics Across Different Reemergence Scenarios', 
          fontsize=14, pad=20)

# Customize legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
          fontsize=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig("../figures/unified_ecosystem_dynamics.png", dpi=300)
plt.show()
