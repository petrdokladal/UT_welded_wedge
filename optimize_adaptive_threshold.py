import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def load_data(filepath):
    """
    1. Import the dataset from a CSV file.
    Splits the data into features (X) and ground truth labels (Y).
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Extract X (all columns with pattern 'X_') and Y (the 'width' column)
    X_columns = [col for col in df.columns if col.startswith('x_')]
    X = df[X_columns].values  # Convert to numpy array  
    Y = df['GT width'].values
    
    return X, Y

def measure_width_adaptive(signal, threshold_fraction, total_range=100.0):
    """
    2. Dynamically calculates a local threshold for this specific signal 
       based on its own floor and baseline, then measures the width.
    
    Parameters:
        signal (np.ndarray): The 1D array of signal amplitudes.
        threshold_fraction (float): Value between 0 and 1 representing the height fraction.
        total_range (float): The total span of the x-axis (e.g., 100).
        
    Returns:
        float: The measured width of the pulse.
    """
    num_points = len(signal)
    dx = total_range / (num_points - 1)
    
    # Dynamically find the min (pulse floor) and max (local noise-averaged baseline)
    # Using percentiles helps ignore single extreme noise spikes
    sig_min = np.min(signal)
    sig_max = np.percentile(signal, 95) 
    
    # Calculate a unique threshold for this specific sample
    local_threshold = sig_min + threshold_fraction * (sig_max - sig_min)
    
    # Count the points where the signal is below its custom local threshold
    is_below_threshold = signal < local_threshold
    measured_width = np.sum(is_below_threshold) * dx
    
    return measured_width

def optimize_threshold_height(X, Y):
    """
    3. Procedure to optimize the threshold fraction given the ground truth Y.
    Minimizes the Mean Absolute Error across all samples.
    """
    # Define the objective function to minimize over the fraction space
    def objective_function(fraction):
        absolute_errors = []
        for i in range(len(X)):
            w_pred = measure_width_adaptive(X[i], fraction)
            absolute_errors.append(abs(w_pred - Y[i]))
        return np.mean(absolute_errors)

    print("Optimizing adaptive threshold fraction...")
    
    # Search bounds are naturally constrained between 0.0 (bottom floor) and 1.0 (top baseline)
    search_bounds = (0.0, 1.0)
    
    result = minimize_scalar(objective_function, bounds=search_bounds, method='bounded')
    
    if result.success:
        return result.x, result.fun
    else:
        raise RuntimeError("Optimization failed to converge.")


def plot_detection_zones (Y, Y_pred, critical_width, figname='predicted_vs_GT_width.png', safety_margin=0, font_size=14):

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(Y, Y_pred, alpha=0.5, color='blue', edgecolors='k', label='Samples')
    
    # Add ideal line (y=x)
    min_val = min(np.min(Y), np.min(Y_pred), critical_width)
    max_val = max(np.max(Y), np.max(Y_pred), critical_width + safety_margin)
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit ($Y_{pred} = Y_{GT}$)')

    # 2. Get current axis limits to bound the colored rectangles
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 3. Define soft, transparent background colors (Alpha controls transparency)
    alpha_val = 0.15
    colors = {
        'TN': '#2ca02c',  # Soft Green
        'TP': '#1f77b4',  # Soft Blue
        'FN': '#d62728',  # Soft Red
        'FP': '#ff7f0e'   # Soft Orange
    }
        
    # 4. Draw the background quadrants
    # NON_CONFORM: Bottom-Left
    ax.fill_between([xlim[0], critical_width], xlim[0], critical_width+safety_margin, color=colors['TP'], alpha=alpha_val, label='TP zone')
    # FP: Top-Left
    ax.fill_between([xlim[0], critical_width], critical_width+safety_margin, ylim[1], color=colors['FN'], alpha=alpha_val, label='FN Zone')
    # FP: Bottom-Right
    ax.fill_between([critical_width, xlim[1]], ylim[0], critical_width+safety_margin, color=colors['FP'], alpha=alpha_val, label='FP Zone')
    # TP: Top-Right
    ax.fill_between([critical_width, xlim[1]], critical_width+safety_margin, ylim[1], color=colors['TN'], alpha=alpha_val, label='TN Zone')
        
    # 5. Draw the critical_width lines to clearly separate the zones
    ax.axvline(x=critical_width, color='black', linestyle='--', linewidth=1.5, label=f'Critical_Width ({critical_width})')
    if safety_margin > 0:
        ax.axhline(y=critical_width+safety_margin, color='black', linestyle='--', linewidth=1.5)
        ax.axhline(y=critical_width, color='#cccccc', linestyle='--', linewidth=1.5)
    else:
        ax.axhline(y=critical_width, color='black', linestyle='--', linewidth=1.5)
        
    # 6. Add text labels inside each quadrant for maximum clarity
    text_y_low = (ylim[0] + critical_width + safety_margin) / 2
    text_y_high = (critical_width + safety_margin + ylim[1]) / 2
    text_x_low = (xlim[0] + critical_width) / 2
    text_x_high = (critical_width + xlim[1]) / 2
    
    ax.text(text_x_low, text_y_low, 'TP \n (vrai non-conforme)', fontsize=font_size, fontweight='bold', ha='center', va='center', color='green', alpha=0.6)
    ax.text(text_x_low, text_y_high, 'FN \n (non-conforme non détecté)', fontsize=font_size, fontweight='bold', ha='center', va='center', color='orange', alpha=0.6)
    ax.text(text_x_high, text_y_low, 'FP \n (conforme mal classé)', fontsize=font_size, fontweight='bold', ha='center', va='center', color='red', alpha=0.6)
    ax.text(text_x_high, text_y_high, 'TN \n (vrai conforme)', fontsize=font_size, fontweight='bold', ha='center', va='center', color='blue', alpha=0.6)
    
    # Re-adjust limits so the fill doesn't expand your original plot scale
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_xlabel('Ground Truth Width ($Y_{GT}$)')
    ax.set_ylabel('Predicted Width ($Y_{pred}$)')
    ax.set_title('Scatter Plot: Predicted Width vs. Ground Truth Width')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(figname, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully as '{figname}'.")





if __name__ == "__main__":
    # Path to the dataset file
    dataset_path = 'weld_width_dataset.csv'
    
    try:
        # 1. Import dataset
        X, Y = load_data(dataset_path)
        print(f"Dataset successfully imported. Shape of X: {X.shape}, Shape of Y: {Y.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found. Please ensure you have generated it first.")
        
    # 3. Optimize the threshold height
    optimal_fraction, min_mae = optimize_threshold_height(X, Y)
        
    print("\n--- Optimization Results ---")
    print(f"Optimal Threshold Fraction: {optimal_fraction:.4f} (aka {optimal_fraction*100:.1f}% up the pulse height)")
    print(f"Minimum Mean Absolute Error: {min_mae:.4f} units of width")
        
    # Verify on a sample realization
    sample_idx = 0
    sample_pred = measure_width_adaptive(X[sample_idx], optimal_fraction)
    print(f"\nVerification on Sample {sample_idx + 1}:")
    print(f"  Ground Truth Width: {Y[sample_idx]:.4f}")
    print(f"  Measured Width:     {sample_pred:.4f}")
    print(f"  Absolute Error:     {abs(sample_pred - Y[sample_idx]):.4f}")
        
    # Predict Y using the optimized global fraction
    Y_pred = np.array([measure_width_adaptive(sig, optimal_fraction) for sig in X])
    
    # 1. Define your threshold (e.g., 25)
    critical_width = 25.0
    safety_margin = 2.0  # Define safety margin to shift the critical width line       
    
    # Plot the detection zones
    plot_detection_zones(Y, Y_pred, critical_width, figname='predicted_vs_GT_width.png', font_size=12)
    # Plot the detection zones with safety margin
    plot_detection_zones(Y, Y_pred, critical_width, figname='predicted_vs_GT_width_safety_margin.png', safety_margin=safety_margin, font_size=12)