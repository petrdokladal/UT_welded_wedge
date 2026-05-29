import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt

def load_data(filepath):
    """
    1. Import the dataset from a CSV file.
    Splits the data into features (X) and ground truth labels (Y).
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Extract X (all columns except the last one) and Y (the 'width' column)
    X = df.iloc[:, :-1].values
    Y = df['width'].values
    
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




if __name__ == "__main__":
    # Path to the dataset file
    dataset_path = 'pulse_dataset.csv'
    
    try:
        # 1. Import dataset
        X, Y = load_data(dataset_path)
        print(f"Dataset successfully imported. Shape of X: {X.shape}, Shape of Y: {Y.shape}")
        
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

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(Y, Y_pred, alpha=0.5, color='blue', edgecolors='k', label='Samples')

        # Add ideal line (y=x)
        min_val = min(np.min(Y), np.min(Y_pred))
        max_val = max(np.max(Y), np.max(Y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit ($Y_{pred} = Y_{GT}$)')

        ax.set_xlabel('Ground Truth Width ($Y_{GT}$)')
        ax.set_ylabel('Predicted Width ($Y_{pred}$)')
        ax.set_title('Scatter Plot: Adaptive Predictions vs. Ground Truth Width')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

        plt.savefig('predicted_vs_gt_y_adaptive.png', bbox_inches='tight')
        plt.close()
        print("Plot saved successfully as 'predicted_vs_gt_y_adaptive.png'.")
        
    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found. Please ensure you have generated it first.")
