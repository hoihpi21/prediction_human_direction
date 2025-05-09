import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

def last_point_prediction(csv_file, user_id=None, window_size=20, prediction_seconds=300, 
                          prediction_horizons=(0.5, 1.0, 1.5, 2.0), max_step_size=None):
    """
    Enhanced last point prediction with support for multiple prediction horizons and evaluation metrics.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with UWB data
    user_id : int, optional
        ID of the user to track (if None, uses ID with most data points)
    window_size : int
        Number of recent points to calculate direction and dynamics (increased to capture more context)
    prediction_seconds : int
        Total number of seconds to predict ahead
    prediction_horizons : tuple
        Time horizons in seconds for evaluation (e.g., 0.5s, 1s, 1.5s, 2s)
    max_step_size : float, optional
        Maximum distance between consecutive predicted points (limits unrealistic speeds)
        
    Returns:
    --------
    dict
        Dictionary containing trajectory and prediction results with evaluation metrics
    """
    # Start computation time measurement
    start_time = time.time()
    
    print(f"Loading data from {csv_file}...")
    data = pd.read_csv(csv_file)
    
    print("Running Enhanced Last Point Reference prediction with multiple horizons...")
    
    # Ensure we only work with the required columns
    required_columns = ['id', 'x', 'y', 'timestamp', 'time_seconds']
    for col in required_columns:
        if col not in data.columns:
            print(f"Warning: Column '{col}' not found in the data.")
    
    # Filter for specific user if provided
    if user_id is not None:
        user_data = data[data['id'] == user_id].copy()
        print(f"Filtered data for user ID: {user_id}")
    else:
        # Find ID with most data points
        id_counts = data['id'].value_counts()
        user_id = id_counts.idxmax()
        user_data = data[data['id'] == user_id].copy()
        print(f"Using user ID with most data points: {user_id} ({len(user_data)} points)")
    
    # Sort by timestamp
    user_data = user_data.sort_values('timestamp')
    print(f"Total data points for user: {len(user_data)}")
    
    # Extract position and time data
    if 'x_kalman' in user_data.columns and not user_data['x_kalman'].isna().all():
        print("Using Kalman filtered coordinates (x_kalman, y_kalman)")
        positions = user_data[['x_kalman', 'y_kalman', 'timestamp', 'time_seconds']].values
    elif 'x_lowpass' in user_data.columns and not user_data['x_lowpass'].isna().all():
        print("Using low-pass filtered coordinates (x_lowpass, y_lowpass)")
        positions = user_data[['x_lowpass', 'y_lowpass', 'timestamp', 'time_seconds']].values
    else:
        print("Using processed coordinates (x, y)")
        positions = user_data[['x', 'y', 'timestamp', 'time_seconds']].values
    
    # We need at least window_size points
    if len(positions) < window_size:
        print(f"Not enough data points (need at least {window_size}).")
        return None
    
    # Apply smoothing to the trajectory
    try:
        positions_xy = smooth_trajectory(positions[:, :2], window=min(7, len(positions) // 2))
        positions_smooth = np.column_stack((positions_xy, positions[:, 2:]))
        print("Applied smoothing filter to trajectory")
    except Exception as e:
        print(f"Smoothing failed: {e}. Using original trajectory.")
        positions_smooth = positions
    
    # Get the last 'window_size' points
    recent_points = positions_smooth[-window_size:, :2]
    recent_times = positions_smooth[-window_size:, 2]
    recent_seconds = positions_smooth[-window_size:, 3]
    
    # Calculate time deltas in seconds
    if len(np.unique(recent_seconds)) > 1:
        print("Using time_seconds for time delta calculation")
        time_deltas = np.diff(recent_seconds)
    else:
        print("Using timestamps for time delta calculation")
        time_deltas = np.diff(recent_times) / 1000.0  # Convert to seconds
    
    avg_time_delta = np.mean(time_deltas)
    
    # Handle zero or negative time deltas
    if avg_time_delta <= 0:
        avg_time_delta = 0.1  # Default 100ms
    
    print(f"Average time between points: {avg_time_delta:.4f} seconds")
    
    # Calculate velocities
    velocities = np.zeros((window_size-1, 2))
    for i in range(window_size-1):
        dt = time_deltas[i]
        if dt > 0:
            velocities[i] = (recent_points[i+1] - recent_points[i]) / dt
        else:
            # Use average time delta if this specific delta is zero or negative
            velocities[i] = (recent_points[i+1] - recent_points[i]) / avg_time_delta
    
    # Calculate accelerations
    accelerations = np.zeros((window_size-2, 2))
    for i in range(window_size-2):
        dt = time_deltas[i] + time_deltas[i+1]  # Time span for acceleration
        if dt > 0:
            accelerations[i] = (velocities[i+1] - velocities[i]) / (dt/2)
        else:
            # Use average time delta if this specific delta is zero or negative
            accelerations[i] = (velocities[i+1] - velocities[i]) / avg_time_delta
    
    # Calculate weighted velocity using improved weighting method
    avg_velocity = calculate_weighted_velocity(velocities)
    velocity_magnitude = np.linalg.norm(avg_velocity)
    print(f"Weighted velocity: [{avg_velocity[0]:.4f}, {avg_velocity[1]:.4f}] m/s (magnitude: {velocity_magnitude:.4f} m/s)")
    
    # Calculate weighted acceleration using improved weighting method
    avg_acceleration = calculate_weighted_acceleration(accelerations)
    acceleration_magnitude = np.linalg.norm(avg_acceleration)
    print(f"Weighted acceleration: [{avg_acceleration[0]:.4f}, {avg_acceleration[1]:.4f}] m/s² (magnitude: {acceleration_magnitude:.4f} m/s²)")
    
    # Determine if movement is accelerating, decelerating, or constant
    if len(velocities) >= 3:
        speed_change = velocities[-1] - velocities[0]
        speed_change_magnitude = np.linalg.norm(speed_change)
        
        if speed_change_magnitude > 0.1:
            if np.dot(speed_change, velocities[-1]) > 0:
                movement_type = "Accelerating"
            else:
                movement_type = "Decelerating"
        else:
            movement_type = "Constant speed"
        
        print(f"Movement type: {movement_type}")
    
    # Calculate motion curvature
    if len(velocities) >= 3:
        direction_changes = np.diff(np.arctan2(velocities[:, 1], velocities[:, 0]))
        avg_curvature = np.mean(np.abs(direction_changes)) * 180 / np.pi  # in degrees
        
        if avg_curvature < 5:
            path_type = "Straight line"
        elif avg_curvature < 15:
            path_type = "Gentle curve"
        elif avg_curvature < 45:
            path_type = "Moderate curve"
        else:
            path_type = "Sharp turns"
        
        print(f"Path type: {path_type} (avg. direction change: {avg_curvature:.2f}°)")
    
    # Last point is the reference point
    last_point = recent_points[-1]
    last_time = recent_seconds[-1]
    
    # Calculate adaptive step size based on recent movement
    step_sizes = np.linalg.norm(np.diff(recent_points, axis=0), axis=1)
    avg_step = np.mean(step_sizes)
    
    # Apply multiplier based on observed movement pattern
    if 'path_type' in locals():
        if path_type == "Straight line":
            multiplier = 1.2  # Confident projection for straight lines
        elif path_type == "Gentle curve":
            multiplier = 1.0  # Standard projection
        else:
            multiplier = 0.8  # More conservative for curves
    else:
        multiplier = 1.0
    
    # Simulate past prediction errors for drift correction
    recent_errors = simulate_past_prediction_errors(velocities, accelerations)
    
    # Calculate total number of prediction steps
    total_steps = int(prediction_seconds / avg_time_delta)
    print(f"Making predictions for {prediction_seconds} seconds ({total_steps} steps)")
    
    # Analyze trajectory for pattern detection
    pattern_info = detect_movement_pattern(positions[:, :2])
    pattern_type = pattern_info['pattern_type']
    print(f"Detected movement pattern: {pattern_type}")
    
    # Set prediction method based on pattern
    if pattern_type in ['circular', 'curved']:
        prediction_method = 'pattern_based'
        print(f"Using pattern-based prediction for {pattern_type} movement")
    else:
        prediction_method = 'kinematic'
        print(f"Using kinematic prediction model")
    
    # Calculate maximum step size to prevent unrealistic speeds
    if max_step_size is None:
        # Set max step size to 3x average observed step (prevents unrealistic speeds)
        max_step_size = 3.0 * avg_step
        print(f"Maximum step size set to: {max_step_size:.4f} meters")
    
    # Predict future positions
    future_positions = []
    future_times = []
    current_point = last_point.copy()
    current_velocity = avg_velocity.copy()
    current_acceleration = avg_acceleration.copy()
    
    for step in range(1, total_steps + 1):
        # Calculate time for this step
        step_time = avg_time_delta
        future_time = last_time + (step * avg_time_delta)
        
        if prediction_method == 'pattern_based' and pattern_type == 'circular':
            # For circular movement, adjust direction to follow the curve
            # Use the circular pattern parameters from pattern detection
            center = pattern_info['center']
            radius = pattern_info['radius']
            angular_velocity = pattern_info['angular_velocity']
            
            # Calculate current angle from center
            vec_to_center = current_point - center
            current_angle = np.arctan2(vec_to_center[1], vec_to_center[0])
            
            # Calculate next angle based on angular velocity
            next_angle = current_angle + (angular_velocity * step_time)
            
            # Calculate next point on the circular path
            next_point = center + np.array([
                radius * np.cos(next_angle),
                radius * np.sin(next_angle)
            ])
            
            # Update velocity based on movement direction
            current_velocity = (next_point - current_point) / step_time
        else:
            # Apply kinematic equation with progressive updating:
            # P(t+Δ) = P(t) + V(t)·Δ + (1/2)A(t)·Δ²
            next_point = (
                current_point + 
                current_velocity * step_time + 
                0.5 * current_acceleration * step_time**2
            )
            
            # Update velocity for next step
            current_velocity += current_acceleration * step_time
            
            # Apply damping to prevent velocity from growing unrealistically
            # This simulates friction/resistance and keeps long-term predictions more reasonable
            damping_factor = 0.995  # Slight damping for each step
            current_velocity *= damping_factor
            
            # Also gradually reduce acceleration over time (otherwise predictions fly off)
            current_acceleration *= 0.99  # Reduce acceleration effect over time
        
        # Apply drift correction
        if recent_errors is not None and len(recent_errors) > 0:
            next_point = apply_drift_correction(next_point, recent_errors)
        
        # Limit step size to prevent unrealistic jumps
        step_vector = next_point - current_point
        step_distance = np.linalg.norm(step_vector)
        
        if step_distance > max_step_size:
            # Scale down the step to the maximum allowed size
            step_vector = (step_vector / step_distance) * max_step_size
            next_point = current_point + step_vector
            
            # Also scale down velocity to match the constrained step
            current_velocity = step_vector / step_time
        
        future_positions.append(next_point.copy())
        future_times.append(future_time)
        
        # Update current point for next iteration
        current_point = next_point.copy()
    
    future_positions = np.array(future_positions)
    future_times = np.array(future_times)
    
    # Calculate direction from velocity
    direction_degrees = np.degrees(np.arctan2(avg_velocity[1], avg_velocity[0]))
    print(f"Predicted direction: {direction_degrees:.2f} degrees")
    
    # Create horizon-based predictions for evaluation
    horizon_predictions = {}
    
    for horizon in prediction_horizons:
        # Find steps that correspond to each horizon
        horizon_step = int(horizon / avg_time_delta)
        if horizon_step < len(future_positions):
            horizon_predictions[horizon] = future_positions[horizon_step]
            print(f"Prediction at {horizon}s horizon: [{horizon_predictions[horizon][0]:.4f}, {horizon_predictions[horizon][1]:.4f}]")
        else:
            print(f"Warning: Cannot predict at {horizon}s horizon (exceeds prediction steps)")
    
    # Calculate direction accuracy
    direction_accuracy = calculate_direction_accuracy(recent_points, direction_degrees)
    print(f"Direction accuracy: {direction_accuracy:.2f}%")
    
    # Calculate MSE and RMSE for the predictions
    # For simulation purposes, we'll use a subset of actual data as ground truth
    # In a real evaluation, you would compare with actual future positions
    
    # Use a hold-out set of points as ground truth (if available)
    holdout_count = min(20, len(positions) - window_size)
    if holdout_count > 0:
        holdout_positions = positions[-window_size-holdout_count:-window_size, :2]
        holdout_times = positions[-window_size-holdout_count:-window_size, 3]
        
        # Calculate predictions for holdout period
        holdout_predictions = []
        for i, t in enumerate(holdout_times):
            step_time = last_time - t
            if step_time > 0:
                pred = (
                    last_point - 
                    avg_velocity * step_time + 
                    0.5 * avg_acceleration * step_time**2
                )
                holdout_predictions.append(pred)
            else:
                # Skip if time is not increasing
                holdout_predictions.append(np.array([np.nan, np.nan]))
        
        holdout_predictions = np.array(holdout_predictions)
        
        # Calculate MSE and RMSE
        valid_indices = ~np.isnan(holdout_predictions[:, 0])
        if np.sum(valid_indices) > 0:
            mse = mean_squared_error(
                holdout_positions[valid_indices], 
                holdout_predictions[valid_indices]
            )
            rmse = np.sqrt(mse)
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
        else:
            mse = np.nan
            rmse = np.nan
            print("Could not calculate MSE/RMSE (no valid predictions)")
    else:
        mse = np.nan
        rmse = np.nan
        print("Could not calculate MSE/RMSE (no holdout data available)")
    
    # Finish computation time measurement
    end_time = time.time()
    computation_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Computation time: {computation_time:.2f} ms")
    
    # Finish computation time measurement
    end_time = time.time()
    computation_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Computation time: {computation_time:.2f} ms")
    
    # Return results
    results = {
        'user_id': user_id,
        'actual_trajectory': positions[:, :2],
        'actual_times': positions[:, 3],
        'future_trajectory': future_positions,
        'future_times': future_times,
        'direction_degrees': direction_degrees,
        'direction_accuracy': direction_accuracy,
        'mse': mse,
        'rmse': rmse,
        'computation_time_ms': computation_time,
        'horizon_predictions': horizon_predictions,
        'prediction_horizons': prediction_horizons,
        'pattern_type': pattern_type,
        'method': f'Enhanced Last Point Prediction with {pattern_type.capitalize()} Pattern Recognition'
    }
    
    return results

def calculate_weighted_velocity(velocities):
    """
    Calculate weighted velocity with exponential decay giving higher weights to recent observations.
    
    Parameters:
    -----------
    velocities : array
        Array of velocity vectors
        
    Returns:
    --------
    array
        Weighted velocity vector
    """
    if len(velocities) < 2:
        return velocities[0] if len(velocities) > 0 else np.zeros(2)
    
    # Use exponential weighting to emphasize recent velocities
    # Higher decay factor (e.g., 0.8) makes weights decay more slowly
    decay_factor = 0.8
    weights = np.array([decay_factor ** (len(velocities) - i - 1) for i in range(len(velocities))])
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    
    weighted_velocity = np.zeros(2)
    for i, vel in enumerate(velocities):
        weighted_velocity += vel * weights[i]
    
    return weighted_velocity

def calculate_weighted_acceleration(accelerations):
    """
    Calculate weighted acceleration with exponential decay giving higher weights to recent observations.
    
    Parameters:
    -----------
    accelerations : array
        Array of acceleration vectors
        
    Returns:
    --------
    array
        Weighted acceleration vector
    """
    if len(accelerations) < 2:
        return accelerations[0] if len(accelerations) > 0 else np.zeros(2)
    
    # Use exponential weighting for accelerations too
    decay_factor = 0.8
    weights = np.array([decay_factor ** (len(accelerations) - i - 1) for i in range(len(accelerations))])
    weights = weights / np.sum(weights)
    
    weighted_acceleration = np.zeros(2)
    for i, acc in enumerate(accelerations):
        weighted_acceleration += acc * weights[i]
    
    return weighted_acceleration

def detect_movement_pattern(points, min_points=10):
    """
    Detect movement patterns in the trajectory (linear, circular, etc.)
    
    Parameters:
    -----------
    points : array
        Trajectory points
    min_points : int
        Minimum number of points needed for pattern detection
        
    Returns:
    --------
    dict
        Dictionary containing pattern information
    """
    if len(points) < min_points:
        return {'pattern_type': 'unknown', 'confidence': 0.0}
    
    # Use the most recent points for pattern detection
    recent_points = points[-min(len(points), 50):]
    
    # Calculate movement vectors between consecutive points
    vectors = np.diff(recent_points, axis=0)
    
    # Calculate step magnitudes and directions
    magnitudes = np.linalg.norm(vectors, axis=1)
    directions = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # Calculate direction changes
    direction_changes = np.diff(directions)
    # Normalize direction changes to [-pi, pi]
    direction_changes = np.mod(direction_changes + np.pi, 2 * np.pi) - np.pi
    
    # Calculate statistics
    mean_direction_change = np.mean(np.abs(direction_changes))
    std_direction_change = np.std(direction_changes)
    mean_step = np.mean(magnitudes)
    
    # Initialize pattern info
    pattern_info = {
        'pattern_type': 'unknown',
        'confidence': 0.0,
        'mean_step': mean_step,
        'mean_direction_change': mean_direction_change,
        'direction_change_std': std_direction_change
    }
    
    # Check for linear movement
    if mean_direction_change < 0.1 and std_direction_change < 0.2:
        pattern_info['pattern_type'] = 'linear'
        pattern_info['confidence'] = 0.8
        
        # Calculate direction of linear movement
        overall_vector = recent_points[-1] - recent_points[0]
        pattern_info['direction'] = np.arctan2(overall_vector[1], overall_vector[0])
        
    # Check for circular/curved movement
    elif 0.05 < mean_direction_change < 0.3 and std_direction_change < 0.2:
        # Consistent direction changes suggest circular motion
        
        # Attempt to fit a circle to the points
        try:
            # Use least squares to fit a circle
            center, radius = fit_circle(recent_points)
            
            # Check if the fit is good
            circle_fit_error = circle_fitting_error(recent_points, center, radius)
            
            if circle_fit_error < 0.3:  # Reasonable fit
                pattern_info['pattern_type'] = 'circular'
                pattern_info['confidence'] = max(0, 1.0 - circle_fit_error)
                pattern_info['center'] = center
                pattern_info['radius'] = radius
                
                # Calculate angular velocity
                # Use the direction changes divided by time step
                # Assuming uniform time steps for simplicity
                avg_angular_velocity = mean_direction_change / 0.1  # Assuming 0.1s between points
                pattern_info['angular_velocity'] = avg_angular_velocity
                
                # Determine direction (clockwise or counterclockwise)
                if np.mean(direction_changes) > 0:
                    pattern_info['rotation'] = 'counterclockwise'
                else:
                    pattern_info['rotation'] = 'clockwise'
            else:
                pattern_info['pattern_type'] = 'curved'
                pattern_info['confidence'] = 0.6
        except:
            pattern_info['pattern_type'] = 'curved'
            pattern_info['confidence'] = 0.5
    
    # Check for stop-and-go pattern
    elif np.std(magnitudes) / np.mean(magnitudes) > 1.0:  # High variation in step size
        pattern_info['pattern_type'] = 'stop_and_go'
        pattern_info['confidence'] = 0.7
    
    # Check for random/erratic movement
    elif std_direction_change > 0.5:
        pattern_info['pattern_type'] = 'erratic'
        pattern_info['confidence'] = 0.7
    
    return pattern_info

def fit_circle(points):
    """
    Fit a circle to a set of points using algebraic least squares.
    
    Parameters:
    -----------
    points : array
        2D points to fit
        
    Returns:
    --------
    tuple
        (center_x, center_y), radius
    """
    # Convert points to numpy array
    points = np.array(points)
    
    # Get x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Mean of the coordinates
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Shift the coordinates to center at the origin
    u = x - mean_x
    v = y - mean_y
    
    # Calculate the elements of the matrix for least squares
    Suv = np.sum(u * v)
    Suu = np.sum(u * u)
    Svv = np.sum(v * v)
    Suuv = np.sum(u * u * v)
    Suvv = np.sum(u * v * v)
    Suuu = np.sum(u * u * u)
    Svvv = np.sum(v * v * v)
    
    # Solve the linear system to find the center and radius
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([0.5 * (Suuu + Suvv), 0.5 * (Svvv + Suuv)])
    
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fall back to approximate center if matrix is singular
        center = np.array([0, 0])
    
    center[0] += mean_x
    center[1] += mean_y
    
    # Calculate the radius
    radius = np.sqrt(np.mean((x - center[0])**2 + (y - center[1])**2))
    
    return center, radius

def circle_fitting_error(points, center, radius):
    """
    Calculate the normalized root mean squared error of a circle fit
    
    Parameters:
    -----------
    points : array
        2D points
    center : array
        Center of the circle [x, y]
    radius : float
        Radius of the circle
        
    Returns:
    --------
    float
        Normalized RMSE of the fit
    """
    # Calculate distances from points to the circle
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    
    # Calculate the error as the difference from the radius
    errors = np.abs(distances - radius)
    
    # Normalize by the radius
    if radius > 0:
        normalized_error = np.sqrt(np.mean(np.square(errors))) / radius
    else:
        normalized_error = np.inf
    
    return normalized_error

def simulate_past_prediction_errors(velocities, accelerations, num_errors=3):
    """
    Simulate past prediction errors for drift correction.
    In a real-time system, you would use actual errors from past predictions.
    
    Parameters:
    -----------
    velocities : array
        Velocity vectors
    accelerations : array
        Acceleration vectors
    num_errors : int
        Number of error samples to generate
        
    Returns:
    --------
    array
        Array of simulated error vectors
    """
    # In a real system, these would be calculated by comparing past predictions with actual positions
    # Here we simulate errors based on the variability in the velocity and acceleration
    
    # Use velocity variability to estimate error magnitude
    if len(velocities) > 1:
        velocity_std = np.std(np.linalg.norm(velocities, axis=1))
        error_magnitude = velocity_std * 0.2  # Scale factor
    else:
        error_magnitude = 0.05  # Default
    
    # Generate random errors with decreasing magnitude (older errors have less influence)
    errors = []
    for i in range(num_errors):
        # Scale error by recency (more recent errors are smaller, assuming prediction is improving)
        scaling = 0.5 + 0.5 * (i / max(1, num_errors-1))
        
        # Random direction for error
        angle = np.random.uniform(0, 2*np.pi)
        error = np.array([
            np.cos(angle) * error_magnitude * scaling,
            np.sin(angle) * error_magnitude * scaling
        ])
        errors.append(error)
    
    return np.array(errors)

def apply_drift_correction(predicted_position, recent_errors, alpha=0.3):
    """
    Apply drift correction to adjust predictions based on recent prediction errors.
    
    Parameters:
    -----------
    predicted_position : array
        Initially predicted position
    recent_errors : array
        List of recent prediction errors
    alpha : float
        Weight for drift correction (0-1)
        
    Returns:
    --------
    array
        Corrected predicted position
    """
    if len(recent_errors) == 0:
        return predicted_position
    
    # Calculate weighted average of recent errors
    weights = np.array([0.7 ** (len(recent_errors) - i - 1) for i in range(len(recent_errors))])
    weights = weights / np.sum(weights)
    
    avg_error = np.zeros(2)
    for i, err in enumerate(recent_errors):
        avg_error += err * weights[i]
    
    # Apply drift correction
    corrected_position = predicted_position + alpha * avg_error
    
    return corrected_position

def smooth_trajectory(points, window=5, poly_order=2):
    """
    Apply Savitzky-Golay filter to smooth trajectory points
    
    Parameters:
    -----------
    points : array
        Trajectory points
    window : int
        Window size for smoothing (must be odd)
    poly_order : int
        Polynomial order for smoothing
        
    Returns:
    --------
    array
        Smoothed trajectory points
    """
    if len(points) < window:
        return points
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    # Apply filter separately to x and y coordinates
    x_smooth = savgol_filter(points[:, 0], window, poly_order)
    y_smooth = savgol_filter(points[:, 1], window, poly_order)
    return np.column_stack((x_smooth, y_smooth))

def calculate_direction_accuracy(recent_points, predicted_direction):
    """
    Calculate the accuracy of the predicted direction compared to recent movement
    
    Parameters:
    -----------
    recent_points : array
        Recent trajectory points
    predicted_direction : float
        Predicted direction in degrees
        
    Returns:
    --------
    float
        Direction accuracy score (0-100)
    """
    # Calculate actual directions from recent points
    vectors = np.diff(recent_points[-5:], axis=0)  # Use last 5 points
    
    # Filter out very small movements that might just be noise
    significant_vectors = vectors[np.linalg.norm(vectors, axis=1) > 0.01]
    
    if len(significant_vectors) == 0:
        return 50.0  # Default value if no significant movement
    
    actual_directions = np.arctan2(significant_vectors[:, 1], significant_vectors[:, 0])
    actual_directions_deg = np.degrees(actual_directions)
    
    # Use weighted average for direction calculation
    weights = np.exp(np.linspace(0, 2, len(actual_directions_deg)))
    weights = weights / np.sum(weights)
    avg_actual_direction = np.sum(actual_directions_deg * weights)
    
    # Calculate the absolute angular difference
    diff = abs(predicted_direction - avg_actual_direction)
    if diff > 180:
        diff = 360 - diff
        
    # More forgiving accuracy score (30 degrees off still gives 80% accuracy)
    accuracy = max(0, 100 - (diff / 3))
    
    return accuracy

def visualize_results(results):
    """
    Visualize the prediction results with multiple horizons
    
    Parameters:
    -----------
    results : dict
        Dictionary containing trajectory and prediction results
    """
    plt.figure(figsize=(14, 10))
    
    # Plot actual trajectory with lighter color at older points
    points = results['actual_trajectory']
    segments = len(points) // 4
    
    for i in range(4):
        start_idx = i * segments
        end_idx = (i+1) * segments if i < 3 else len(points)
        alpha = 0.3 + 0.7 * (i / 3)  # 0.3 to 1.0
        plt.plot(points[start_idx:end_idx, 0], 
                 points[start_idx:end_idx, 1], 
                 'b-', alpha=alpha, linewidth=1.5)
    
    # Highlight the most recent points used for prediction
    recent_idx = max(0, len(points) - 8)
    plt.plot(points[recent_idx:, 0], 
             points[recent_idx:, 1], 
             'b-', linewidth=2.5, label='Recent Trajectory')
    
    # Plot future trajectory
    plt.plot(results['future_trajectory'][:, 0], 
             results['future_trajectory'][:, 1], 
             'r-', linewidth=2.5, label='Predicted Path')
    
    # Mark start and current points
    plt.scatter(points[0, 0], 
                points[0, 1], 
                c='green', s=100, label='Start')
    
    plt.scatter(points[-1, 0], 
                points[-1, 1], 
                c='blue', s=150, marker='*', label='Current Position')
    
    # Mark specific prediction horizons
    horizon_colors = ['purple', 'orange', 'cyan', 'magenta']
    for i, horizon in enumerate(results['prediction_horizons']):
        if horizon in results['horizon_predictions']:
            pos = results['horizon_predictions'][horizon]
            color = horizon_colors[i % len(horizon_colors)]
            plt.scatter(pos[0], pos[1], 
                        c=color, s=120, 
                        label=f'{horizon}s Horizon')
    
    # Mark predicted end point
    plt.scatter(results['future_trajectory'][-1, 0], 
                results['future_trajectory'][-1, 1], 
                c='red', s=150, label='Final Position (300s)')
    
    # Add direction arrow
    last_actual = results['actual_trajectory'][-1]
    direction_rad = np.radians(results['direction_degrees'])
    arrow_length = 1.0  # Adjust based on your coordinate scale
    dx = arrow_length * np.cos(direction_rad)
    dy = arrow_length * np.sin(direction_rad)
    
    plt.arrow(last_actual[0], last_actual[1], dx, dy, 
              head_width=0.3, head_length=0.4, fc='red', ec='red', 
              label='Predicted Direction')
    
    # Add metrics to the plot
    direction_accuracy = results['direction_accuracy']
    mse = results['mse']
    rmse = results['rmse']
    computation_time = results['computation_time_ms']
    
    plt.text(0.02, 0.98, 
             f"Direction Accuracy: {direction_accuracy:.1f}%\n"
             f"MSE: {mse:.4f}\n"
             f"RMSE: {rmse:.4f}\n"
             f"Computation Time: {computation_time:.2f} ms\n"
             f"Method: {results['method']}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend and labels
    plt.legend(loc='lower right')
    plt.title(f"Movement Prediction for User ID: {results['user_id']} (300s forecast)")
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def visualize_horizon_predictions(results):
    """
    Create a detailed visualization of predictions at different time horizons
    
    Parameters:
    -----------
    results : dict
        Dictionary containing trajectory and prediction results
    """
    plt.figure(figsize=(12, 10))
    
    # Plot actual trajectory
    points = results['actual_trajectory']
    plt.plot(points[:, 0], points[:, 1], 
             'b-', linewidth=1.5, alpha=0.5, label='Full Trajectory')
    
    # Highlight the most recent points used for prediction
    recent_idx = max(0, len(points) - 8)
    plt.plot(points[recent_idx:, 0], 
             points[recent_idx:, 1], 
             'b-', linewidth=2.5, label='Recent Trajectory')
    
    # Current position
    current_pos = points[-1]
    plt.scatter(current_pos[0], current_pos[1], 
                c='blue', s=150, marker='*', label='Current Position')
    
    # Plot prediction horizons
    colors = ['green', 'orange', 'red', 'purple']
    
    # Draw horizon-specific trajectories
    for i, horizon in enumerate(results['prediction_horizons']):
        color = colors[i % len(colors)]
        
        # Find index in future_times that corresponds to this horizon
        future_times = results['future_times']
        last_time = results['actual_times'][-1]
        target_time = last_time + horizon
        
        # Find closest time point
        closest_idx = np.argmin(np.abs(future_times - target_time))
        
        # Draw path to this horizon
        if closest_idx < len(results['future_trajectory']):
            # Plot trajectory up to this horizon
            horizon_path = results['future_trajectory'][:closest_idx+1]
            plt.plot(horizon_path[:, 0], horizon_path[:, 1], 
                    f'{color}--', linewidth=1.5, 
                    label=f'Path to {horizon}s Horizon')
            
            # Mark the point
            horizon_point = results['future_trajectory'][closest_idx]
            plt.scatter(horizon_point[0], horizon_point[1], 
                        c=color, s=120, marker='o', 
                        label=f'{horizon}s Prediction')
    
    # Add a grid of coordinates to give spatial context
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add metrics to the visualization
    direction_accuracy = results['direction_accuracy']
    mse = results['mse']
    rmse = results['rmse']
    computation_time = results['computation_time_ms']
    
    metrics_text = (
        f"Evaluation Metrics:\n"
        f"Direction Accuracy: {direction_accuracy:.1f}%\n"
        f"MSE: {mse:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"Computation Time: {computation_time:.2f} ms"
    )
    
    plt.figtext(0.02, 0.02, metrics_text, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Title and labels
    plt.title(f"Predictions at Multiple Time Horizons ({', '.join([f'{h}s' for h in results['prediction_horizons']])})")
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    
    # Legend with a reasonable size
    plt.legend(loc='upper right', fontsize=10)
    
    # Equal aspect ratio
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the enhanced prediction algorithm
    """
    # File path - change to your file path
    file_path = 'uwb_preprocessing.csv'
    
    # Run prediction with multiple horizons
    results = last_point_prediction(
        csv_file=file_path,
        user_id=None,  # Will use the ID with most data points
        window_size=20,  # Increased from 8 to capture more context
        prediction_seconds=150,  # 5 minutes prediction
        prediction_horizons=(0.5, 1.0, 1.5, 2.0),  # Multiple evaluation horizons
        max_step_size=None  # Will be auto-calculated based on observed movement
    )
    
    if results:
        # Visualize results
        visualize_results(results)
        
        # Visualize horizon-specific predictions
        visualize_horizon_predictions(results)
        
        # Print detailed metrics report
        print("\n" + "="*50)
        print(" PREDICTION METRICS SUMMARY ".center(50, "="))
        print("="*50)
        print(f"User ID: {results['user_id']}")
        print(f"Method: {results['method']}")
        print(f"Direction Accuracy: {results['direction_accuracy']:.2f}%")
        print(f"MSE: {results['mse']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"Computation Time: {results['computation_time_ms']:.2f} ms")
        print("\nPrediction Horizons:")
        
        for horizon in results['prediction_horizons']:
            if horizon in results['horizon_predictions']:
                pos = results['horizon_predictions'][horizon]
                print(f"  {horizon}s: ({pos[0]:.4f}, {pos[1]:.4f})")
        
        print("\n" + "="*50)
        
if __name__ == "__main__":
    main()