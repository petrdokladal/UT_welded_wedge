import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
# np.random.seed(42)

# Define the grid over the x-axis
num_points = 500
x_grid = np.linspace(0, 100, num_points)

def generate_welding_wedge(x, center, width, depth, slope_width=2.0):
    """
    Generates a negative trapezoidal pulse with finite slopes on each side.
    """
    t1 = center - width / 2
    t2 = t1 + slope_width
    t4 = center + width / 2
    t3 = t4 - slope_width
    
    y = np.zeros_like(x)
    
    # Ramp down side
    mask_ramp_down = (x >= t1) & (x < t2)
    y[mask_ramp_down] = -depth * (x[mask_ramp_down] - t1) / slope_width
    
    # Flat bottom
    mask_flat = (x >= t2) & (x < t3)
    y[mask_flat] = -depth
    
    # Ramp up side
    mask_ramp_up = (x >= t3) & (x < t4)
    y[mask_ramp_up] = -depth * (t4 - x[mask_ramp_up]) / slope_width
    
    return y

# Generate the dataset
num_samples = 100
X_data = []
Y_data = []

for _ in range(num_samples):
    # randomize the welding wedge geometry for each realization
    width = np.random.uniform(40, 50)
    center = np.random.uniform(40, 60)
    depth = np.random.uniform(3, 5)
    noise_level = 0.2
    slope_width = np.random.uniform(5, 15)

    # generate the wedge pulse and add noise
    pulse = generate_welding_wedge(x_grid, center, width, depth, slope_width=slope_width)
    noise = np.random.normal(0, noise_level, size=num_points)
    signal = pulse - np.min(pulse) + noise
    
    X_data.append(signal)
    # Y_data.append(width - .2*slope_width) # operator places the cursor high (close red in the colormap)
    Y_data.append(width - 2.*slope_width) # operator places the cursor low (close to zero)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Save the generated dataset to a CSV file
columns = [f'x_{i}' for i in range(num_points)] + ['width']
df = pd.DataFrame(np.column_stack((X_data, Y_data)), columns=columns)
df.to_csv('pulse_dataset.csv', index=False)

# Plot the first 5 realizations
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(5):
    ax.plot(x_grid, X_data[i], label=f'Realization {i+1} (Width: {Y_data[i]:.2f})')

ax.set_title('First 5 Examples of Welded Wedge')
ax.set_xlabel('Y Position')
ax.set_ylabel('US Amplitude')
ax.legend()
ax.grid(True)
#plt.show(block=True)
plt.savefig('welded_wedge_examples.png')
plt.close()
