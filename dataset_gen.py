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
    critical_width = 20 # on the lower end (or below) of the width range
    center = np.random.uniform(40, 60)
    depth = np.random.uniform(3, 5)
    noise_level = 0.1
    slope_width = np.random.uniform(5, 10)

    # generate the wedge pulse and add noise
    pulse = generate_welding_wedge(x_grid, center, width, depth, slope_width=slope_width)
    noise = np.random.normal(0, noise_level, size=num_points)
    signal = pulse - np.min(pulse) + noise
    
    if True:
        # the cursor is rather low (close to blue in the colormap)
        GT_width = width - 2.*slope_width
    else:
        # the cursor is rather high (close red in the colormap)
        GT_width = width - .2*slope_width

    reader_1 = GT_width + np.random.normal(0, 2)  # Simulate reader 1 with some noise
    reader_2 = GT_width + np.random.normal(0, 2)  # Simulate reader 2 with some noise
    conform_1 = 1 if reader_1 > critical_width else 0  # Conformity for reader 1
    conform_2 = 1 if reader_2 > critical_width else 0  # Conformity for reader 2
    # if conform_1 != conform_2 then reader_3 provides the tiebreaker
    reader_3 = ''
    conform_3 = ''
    if conform_1 != conform_2:
        # reader 3 provides reading close GT_width
        reader_3 = GT_width + np.random.normal(0, 2)  # Simulate reader 3 with less noise
        conform_3 = 1 if reader_3 > critical_width else 0  

    Y_data_example = [GT_width, reader_1, conform_1, reader_2, conform_2, reader_3, conform_3]

    X_data.append(signal)
    Y_data.append(Y_data_example)


X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Save the generated dataset to a CSV file
columns = [f'x_{i}' for i in range(num_points)] + ['GT width'] + ['reader 1'] + ['CONFORMITY 1'] + ['reader 2'] + ['CONFORMITY 2'] + ['reader 3'] + ['CONFORMITY 3'] 
df = pd.DataFrame(np.column_stack((X_data, Y_data)), columns=columns)
df.to_csv('weld_width_dataset.csv', index=False)

# Plot the first 5 realizations
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(5):
    ax.plot(x_grid, X_data[i], label=f'Realization {i+1} (GT Width: {float(Y_data[i][0]):.2f})')

ax.set_title('First 5 Examples of Welded Wedge')
ax.set_xlabel('Y Position')
ax.set_ylabel('US Amplitude')
ax.legend()
ax.grid(True)
#plt.show(block=True)
plt.savefig('welded_wedge_examples.png')
plt.close()
