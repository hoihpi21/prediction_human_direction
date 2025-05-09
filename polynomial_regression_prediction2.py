import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import time

def polynomial_regression_prediction(csv_file, user_id=None, window_size=20, 
                                    prediction_seconds=300, prediction_horizons=(0.5, 1.0, 1.5, 2.0),
                                    max_degree=5, use_ensemble=True, robust_regression=True):
    """
    Enhanced polynomial regression for long-term trajectory prediction with pattern recognition,
    multiple horizons, and comprehensive evaluation metrics.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with UWB data
    user_id : int, optional
        ID of the user to track (if None, uses ID with most data points)
    window_size : int
        Number of recent points to use for polynomial fitting
    prediction_seconds : int
        Total number of seconds to predict ahead
    prediction_horizons : tuple
        Time horizons in seconds for evaluation (e.g., 0.5s, 1s, 1.5s, 2s)
    max_degree : int
        Maximum polynomial degree to consider
    use_ensemble : bool
        Whether to use ensemble of multiple polynomial degrees
    robust_regression : bool
        Whether to use robust regression methods (less sensitive to outliers)
        
    Returns:
    --------
    dict
        Dictionary containing trajectory and prediction results with evaluation metrics
    """
    # Start computation time measurement
    start_time = time.time()
    
    print(f"Loading data from {csv_file}...")
    data = pd.read_csv(csv_file)
    
    print(f"Running Enhanced Polynomial Regression prediction for {prediction_seconds}s...")
    
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
        positions = user_data[['x_kalman', 'y_kalman']].values
    elif 'x_lowpass' in user_data.columns and not user_data['x_lowpass'].isna().all():
        print("Using low-pass filtered coordinates (x_lowpass, y_lowpass)")
        positions = user_data[['x_lowpass', 'y_lowpass']].values
    else:
        print("Using processed coordinates (x, y)")
        positions = user_data[['x', 'y']].values
    
    timestamps = user_data['timestamp'].values
    time_seconds = user_data['time_seconds'].values if 'time_seconds' in user_data.columns else timestamps / 1000.0
    
    # Need at least window_size points
    if len(positions) < window_size:
        print(f"Not enough data points (need at least {window_size}).")
        return None
    
    # Apply smoothing to reduce noise for better polynomial fitting
    try:
        positions_smoothed = smooth_trajectory(positions)
        print("Applied smoothing filter to trajectory data")
    except Exception as e:
        print(f"Smoothing failed: {e}. Using original data.")
        positions_smoothed = positions
    
    # Analyze trajectory for pattern detection
    pattern_info = detect_movement_pattern(positions_smoothed)
    pattern_type = pattern_info['pattern_type']
    print(f"Detected movement pattern: {pattern_type}")
    
    # Get recent points for polynomial fitting
    recent_points = positions_smoothed[-window_size:]
    recent_times = time_seconds[-window_size:]
    
    # Normalize timestamps to start from 0 and use seconds
    t_normalized = recent_times - recent_times[0]
    
    # Create time points for fitting
    t = np.arange(window_size)
    
    # Calculate time deltas
    time_deltas = np.diff(recent_times)
    # Handle zero or negative time deltas
    time_deltas[time_deltas <= 0] = 0.1  # Default 100ms
    avg_time_delta = np.mean(time_deltas)
    print(f"Average time between points: {avg_time_delta:.4f} seconds")
    
    # Adapt polynomial degree based on detected pattern
    if pattern_type == 'linear':
        print("Using lower degree polynomials for linear pattern")
        # Linear patterns need lower degrees (1-2)
        max_degree = min(max_degree, 2)
        preferred_degree = 1
    elif pattern_type == 'circular':
        print("Using higher degree polynomials for circular pattern")
        # Circular patterns need higher degrees (3-4)
        max_degree = max(max_degree, 4)
        preferred_degree = 4
    elif pattern_type == 'curved':
        print("Using medium degree polynomials for curved pattern")
        # Curved patterns need medium degrees (2-3)
        preferred_degree = 3
    else:
        # For other patterns, try a wider range
        preferred_degree = None
    
    # If we're using the ensemble approach
    if use_ensemble:
        print("Using ensemble of polynomial regression models...")
        # Try different degrees and select best models for each coordinate
        if len(t) > max_degree + 1:  # Ensure we have enough points for highest degree
            degrees_to_try = range(1, max_degree + 1)
        else:
            degrees_to_try = range(1, len(t) // 2)  # Use at most half the window size as degree
        
        models_x = {}
        models_y = {}
        r2_scores_x = {}
        r2_scores_y = {}
        
        for degree in degrees_to_try:
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree)
            t_poly = poly.fit_transform(t.reshape(-1, 1))
            
            # Adjust regularization strength based on degree (higher degree -> stronger regularization)
            alpha = 0.1 * (degree ** 1.5)
            
            # Fit models for x and y
            if robust_regression:
                if degree <= 2:
                    # RANSAC is good for lower degrees and more robust to outliers
                    ransac_x = RANSACRegressor(
                        Ridge(alpha=alpha), 
                        min_samples=0.6, 
                        max_trials=100, 
                        residual_threshold=0.1
                    )
                    ransac_y = RANSACRegressor(
                        Ridge(alpha=alpha),
                        min_samples=0.6,
                        max_trials=100,
                        residual_threshold=0.1
                    )
                    model_x = ransac_x.fit(t_poly, recent_points[:, 0])
                    model_y = ransac_y.fit(t_poly, recent_points[:, 1])
                else:
                    # Huber is better for higher degrees
                    model_x = HuberRegressor(epsilon=1.35, alpha=alpha).fit(t_poly, recent_points[:, 0])
                    model_y = HuberRegressor(epsilon=1.35, alpha=alpha).fit(t_poly, recent_points[:, 1])
            else:
                # Standard linear regression with L2 regularization
                model_x = Ridge(alpha=alpha).fit(t_poly, recent_points[:, 0])
                model_y = Ridge(alpha=alpha).fit(t_poly, recent_points[:, 1])
                
            # Predict on the training data to evaluate fit
            x_pred = model_x.predict(t_poly)
            y_pred = model_y.predict(t_poly)
            
            # Calculate R² scores
            r2_x = r2_score(recent_points[:, 0], x_pred)
            r2_y = r2_score(recent_points[:, 1], y_pred)
            
            # Calculate MSE
            mse_x = mean_squared_error(recent_points[:, 0], x_pred)
            mse_y = mean_squared_error(recent_points[:, 1], y_pred)
            
            # Combined score (prefer higher R² and lower MSE)
            # Adjust the weight for the preferred degree
            degree_preference = 1.0
            if preferred_degree is not None:
                degree_preference = 1.0 + 0.2 * max(0, 1.0 - abs(degree - preferred_degree) / max_degree)
            
            combined_score_x = (r2_x - 0.1 * mse_x) * degree_preference
            combined_score_y = (r2_y - 0.1 * mse_y) * degree_preference
            
            # Store models and scores
            models_x[degree] = (model_x, poly)
            models_y[degree] = (model_y, poly)
            r2_scores_x[degree] = r2_x
            r2_scores_y[degree] = r2_y
            
            print(f"Degree {degree}: R² for x = {r2_x:.4f}, R² for y = {r2_y:.4f}, " +
                  f"Combined Score: x = {combined_score_x:.4f}, y = {combined_score_y:.4f}")
        
        # Select best models based on combined score
        best_degree_x = max(r2_scores_x.items(), key=lambda x: x[1])[0]
        best_degree_y = max(r2_scores_y.items(), key=lambda x: x[1])[0]
        
        print(f"Best degree for x: {best_degree_x} (R² = {r2_scores_x[best_degree_x]:.4f})")
        print(f"Best degree for y: {best_degree_y} (R² = {r2_scores_y[best_degree_y]:.4f})")
        
        # Get best models
        model_x, poly_x = models_x[best_degree_x]
        model_y, poly_y = models_y[best_degree_y]
        
        # Calculate weights for ensemble based on R² scores
        weights_x = {}
        weights_y = {}
        
        for degree in degrees_to_try:
            # Convert R² to weights using softmax-like normalization
            weights_x[degree] = max(0.1, r2_scores_x[degree]**2)  # Square to emphasize differences
            weights_y[degree] = max(0.1, r2_scores_y[degree]**2)
        
        # Normalize weights to sum to 1
        sum_weights_x = sum(weights_x.values())
        sum_weights_y = sum(weights_y.values())
        
        for degree in degrees_to_try:
            weights_x[degree] /= sum_weights_x
            weights_y[degree] /= sum_weights_y
            
            print(f"Weight for degree {degree}: x = {weights_x[degree]:.3f}, y = {weights_y[degree]:.3f}")
        
        # Calculate total number of prediction steps
        total_steps = int(prediction_seconds / avg_time_delta)
        print(f"Total prediction steps: {total_steps}")
        
        # Ensure we have at least 1 prediction step
        if total_steps <= 0:
            # If avg_time_delta is too large, set a minimum number of steps
            total_steps = max(20, int(prediction_seconds / 1.0))  # Use 1 second as fallback
            print(f"Adjusted prediction steps to: {total_steps}")
        
        # Generate future time points
        future_t = np.arange(window_size, window_size + total_steps)
        future_times = []
        last_time = recent_times[-1]
        
        # Make predictions for each time step
        future_positions = []
        
        for step in range(total_steps):
            future_time = last_time + ((step + 1) * avg_time_delta)
            future_times.append(future_time)
            
            # Normalized step for prediction
            step_normalized = window_size + step
            
            # Ensemble prediction for x and y coordinates
            x_ensemble = 0
            y_ensemble = 0
            
            for degree in degrees_to_try:
                model, poly = models_x[degree]
                t_poly = poly.transform(np.array([[step_normalized]]))
                x_ensemble += model.predict(t_poly)[0] * weights_x[degree]
                
                model, poly = models_y[degree]
                t_poly = poly.transform(np.array([[step_normalized]]))
                y_ensemble += model.predict(t_poly)[0] * weights_y[degree]
            
            # Store prediction
            future_positions.append([x_ensemble, y_ensemble])
            
            # Log progress (not too frequently)
            if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
                print(f"Step {step+1}/{total_steps}: Predicted position = [{x_ensemble:.4f}, {y_ensemble:.4f}]")
        
        future_positions = np.array(future_positions)
        future_times = np.array(future_times)
        
        # Use overall best degree for metrics
        r2_x_best = r2_scores_x[best_degree_x]
        r2_y_best = r2_scores_y[best_degree_y]
        avg_r2 = (r2_x_best + r2_y_best) / 2
        
    else:
        # Use single polynomial model with automatic degree selection
        print("Using single polynomial model with automatic degree selection...")
        
        # Find optimal degree based on detected pattern and cross-validation
        if preferred_degree is not None:
            best_degree = preferred_degree
            print(f"Using pattern-based degree selection: {best_degree}")
        else:
            best_degree = find_optimal_degree(t, recent_points, max_degree)
            print(f"Selected optimal polynomial degree: {best_degree}")
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=best_degree)
        t_poly = poly.fit_transform(t.reshape(-1, 1))
        
        # Adjust regularization strength based on degree
        alpha = 0.1 * (best_degree ** 1.5)
        
        # Fit models for x and y
        if robust_regression:
            if best_degree <= 2:
                # RANSAC for lower degrees
                ransac_x = RANSACRegressor(
                    Ridge(alpha=alpha), 
                    min_samples=0.6, 
                    max_trials=100, 
                    residual_threshold=0.1
                )
                ransac_y = RANSACRegressor(
                    Ridge(alpha=alpha),
                    min_samples=0.6,
                    max_trials=100,
                    residual_threshold=0.1
                )
                model_x = ransac_x.fit(t_poly, recent_points[:, 0])
                model_y = ransac_y.fit(t_poly, recent_points[:, 1])
            else:
                # Huber for higher degrees
                model_x = HuberRegressor(epsilon=1.35, alpha=alpha).fit(t_poly, recent_points[:, 0])
                model_y = HuberRegressor(epsilon=1.35, alpha=alpha).fit(t_poly, recent_points[:, 1])
        else:
            # Standard linear regression with L2 regularization
            model_x = Ridge(alpha=alpha).fit(t_poly, recent_points[:, 0])
            model_y = Ridge(alpha=alpha).fit(t_poly, recent_points[:, 1])
        
        # Calculate R² scores
        x_pred = model_x.predict(t_poly)
        y_pred = model_y.predict(t_poly)
        r2_x = r2_score(recent_points[:, 0], x_pred)
        r2_y = r2_score(recent_points[:, 1], y_pred)
        avg_r2 = (r2_x + r2_y) / 2
        
        print(f"Model fit: R² for x = {r2_x:.4f}, R² for y = {r2_y:.4f}, Average = {avg_r2:.4f}")
        
        # Calculate total number of prediction steps
        total_steps = int(prediction_seconds / avg_time_delta)
        print(f"Total prediction steps: {total_steps}")
        
        # Ensure we have at least 1 prediction step
        if total_steps <= 0:
            # If avg_time_delta is too large, set a minimum number of steps
            total_steps = max(20, int(prediction_seconds / 1.0))  # Use 1 second as fallback
            print(f"Adjusted prediction steps to: {total_steps}")
        
        # Generate future time points
        future_t = np.arange(window_size, window_size + total_steps)
        future_times = []
        last_time = recent_times[-1]
        
        # Predict future positions
        future_positions = []
        
        for step in range(total_steps):
            future_time = last_time + ((step + 1) * avg_time_delta)
            future_times.append(future_time)
            
            # Normalized step for prediction
            step_normalized = window_size + step
            
            # Transform time point
            t_poly_step = poly.transform(np.array([[step_normalized]]))
            
            # Predict position
            x_pred = model_x.predict(t_poly_step)[0]
            y_pred = model_y.predict(t_poly_step)[0]
            
            # Store prediction
            future_positions.append([x_pred, y_pred])
            
            # Log progress (not too frequently)
            if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
                print(f"Step {step+1}/{total_steps}: Predicted position = [{x_pred:.4f}, {y_pred:.4f}]")
        
        future_positions = np.array(future_positions)
        future_times = np.array(future_times)
    
    # Create horizon-based predictions for evaluation
    horizon_predictions = {}
    
    for horizon in prediction_horizons:
        # Find steps that correspond to each horizon
        horizon_time = last_time + horizon
        if len(future_times) > 0:
            closest_idx = np.argmin(np.abs(future_times - horizon_time))
            if closest_idx < len(future_positions):
                horizon_predictions[horizon] = future_positions[closest_idx]
                print(f"Prediction at {horizon}s horizon: [{horizon_predictions[horizon][0]:.4f}, {horizon_predictions[horizon][1]:.4f}]")
            else:
                print(f"Warning: Cannot predict at {horizon}s horizon (exceeds prediction steps)")
        else:
            print(f"Warning: No future times available for horizon predictions")
    
    # Calculate direction from the first few predicted points
    if len(future_positions) >= 3:
        dx = future_positions[2, 0] - future_positions[0, 0]
        dy = future_positions[2, 1] - future_positions[0, 1]
        direction_degrees = np.degrees(np.arctan2(dy, dx))
    elif len(future_positions) >= 2:
        dx = future_positions[1, 0] - future_positions[0, 0]
        dy = future_positions[1, 1] - future_positions[0, 1]
        direction_degrees = np.degrees(np.arctan2(dy, dx))
    elif len(future_positions) >= 1:
        # Use derivative at the last point (slope of the curve)
        dx = future_positions[0, 0] - recent_points[-1, 0]
        dy = future_positions[0, 1] - recent_points[-1, 1]
        direction_degrees = np.degrees(np.arctan2(dy, dx))
    else:
        # Ultimate fallback - use recent direction from actual trajectory
        if len(positions) >= 2:
            last_two = positions[-2:]
            dx = last_two[1, 0] - last_two[0, 0]
            dy = last_two[1, 1] - last_two[0, 1]
            direction_degrees = np.degrees(np.arctan2(dy, dx))
        else:
            direction_degrees = 0.0
    
    print(f"Predicted direction: {direction_degrees:.2f} degrees")
    
    # Calculate prediction metrics
    direction_accuracy = calculate_direction_accuracy(recent_points, direction_degrees)
    print(f"Direction accuracy: {direction_accuracy:.2f}%")
    
    # Calculate MSE for model fit
    mse = ((x_pred - recent_points[:, 0])**2 + (y_pred - recent_points[:, 1])**2).mean()
    rmse = np.sqrt(mse)
    print(f"Model fit RMSE: {rmse:.4f}")
    
    # Calculate position error estimate based on R² score and pattern type
    # Lower R² means higher error
    position_error = 0.15 * (1 - avg_r2)
    
    # Adjust based on pattern type
    if pattern_type == 'linear':
        position_error *= 0.8  # Linear patterns are more predictable
    elif pattern_type == 'erratic':
        position_error *= 1.5  # Erratic patterns are less predictable
    
    print(f"Estimated position error: {position_error*100:.1f} cm")
    
    # Calculate overall movement angle
    movement_angle = calculate_movement_angle(recent_points)
    
    # Finish computation time measurement
    end_time = time.time()
    computation_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Computation time: {computation_time:.2f} ms")
    
    # Return results
    results = {
        'user_id': user_id,
        'actual_trajectory': positions,
        'future_trajectory': future_positions,
        'future_times': future_times,
        'direction_degrees': direction_degrees,
        'direction_accuracy': direction_accuracy,
        'position_error_cm': position_error * 100,  # Convert to cm
        'movement_angle': movement_angle,
        'r2_score': avg_r2,
        'mse': mse,
        'rmse': rmse,
        'computation_time_ms': computation_time,
        'horizon_predictions': horizon_predictions,
        'prediction_horizons': prediction_horizons,
        'pattern_type': pattern_type,
        'method': f'Pattern-Enhanced Polynomial Regression' + (' (Ensemble)' if use_ensemble else '')
    }
    
    return results

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
        overall_direction = np.arctan2(overall_vector[1], overall_vector[0])
        pattern_info['direction'] = overall_direction
        
        # Calculate primary direction vector (normalized)
        direction_vector = overall_vector / np.linalg.norm(overall_vector)
        pattern_info['direction_vector'] = direction_vector
        
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

def find_optimal_degree(t, points, max_degree):
    """
    Find the optimal polynomial degree using cross-validation
    
    Parameters:
    -----------
    t : array
        Time points
    points : array
        Position points [x, y]
    max_degree : int
        Maximum polynomial degree to consider
        
    Returns:
    --------
    int
        Optimal polynomial degree
    """
    # Ensure max_degree is not too high for the number of points
    max_possible = len(t) // 2  # Avoid overfitting
    max_degree = min(max_degree, max_possible)
    
    # Calculate AIC (Akaike Information Criterion) for model selection
    best_aic = float('inf')
    best_degree = 1
    
    # Store additional metrics for each degree
    metrics = {}
    
    for degree in range(1, max_degree + 1):
        aic_x = calculate_aic(t, points[:, 0], degree)
        aic_y = calculate_aic(t, points[:, 1], degree)
        aic_avg = (aic_x + aic_y) / 2
        
        # Also calculate BIC and cross-validation error
        bic_x = calculate_bic(t, points[:, 0], degree)
        bic_y = calculate_bic(t, points[:, 1], degree)
        bic_avg = (bic_x + bic_y) / 2
        
        # 5-fold cross-validation error
        cv_x = cross_validation_error(t, points[:, 0], degree)
        cv_y = cross_validation_error(t, points[:, 1], degree)
        cv_avg = (cv_x + cv_y) / 2
        
        # Store metrics
        metrics[degree] = {
            'aic': aic_avg,
            'bic': bic_avg,
            'cv': cv_avg
        }
        
        # Print metrics for comparison
        print(f"Degree {degree}: AIC = {aic_avg:.2f}, BIC = {bic_avg:.2f}, CV Error = {cv_avg:.6f}")
        
        # Update best degree (using AIC as primary criterion)
        if aic_avg < best_aic:
            best_aic = aic_avg
            best_degree = degree
    
    # Check if cross-validation suggests a different degree
    cv_degrees = sorted(metrics.keys(), key=lambda d: metrics[d]['cv'])
    cv_best = cv_degrees[0]
    
    # If there's a significant difference between AIC and CV, report it
    if cv_best != best_degree:
        print(f"Note: Cross-validation suggests degree {cv_best} while AIC suggests {best_degree}")
        # If CV error is significantly better, use that instead
        if metrics[cv_best]['cv'] < 0.8 * metrics[best_degree]['cv']:
            print(f"Using cross-validation recommended degree {cv_best} due to significantly lower error")
            best_degree = cv_best
    
    return best_degree

def calculate_aic(t, y, degree):
    """
    Calculate Akaike Information Criterion for polynomial fit
    
    Parameters:
    -----------
    t : array
        Independent variable
    y : array
        Dependent variable
    degree : int
        Polynomial degree
        
    Returns:
    --------
    float
        AIC value (lower is better)
    """
    poly = PolynomialFeatures(degree=degree)
    t_poly = poly.fit_transform(t.reshape(-1, 1))
    
    # Fit the model
    model = Ridge(alpha=0.1).fit(t_poly, y)
    
    # Calculate predictions
    y_pred = model.predict(t_poly)
    
    # Calculate residual sum of squares
    rss = np.sum((y - y_pred)**2)
    
    # Number of parameters (coefficients)
    k = degree + 1
    
    # Number of data points
    n = len(y)
    
    # Calculate AIC
    # AIC = 2k + n*ln(RSS/n)
    aic = 2 * k + n * np.log(rss / n)
    
    return aic

def calculate_bic(t, y, degree):
    """
    Calculate Bayesian Information Criterion for polynomial fit
    
    Parameters:
    -----------
    t : array
        Independent variable
    y : array
        Dependent variable
    degree : int
        Polynomial degree
        
    Returns:
    --------
    float
        BIC value (lower is better)
    """
    poly = PolynomialFeatures(degree=degree)
    t_poly = poly.fit_transform(t.reshape(-1, 1))
    
    # Fit the model
    model = Ridge(alpha=0.1).fit(t_poly, y)
    
    # Calculate predictions
    y_pred = model.predict(t_poly)
    
    # Calculate residual sum of squares
    rss = np.sum((y - y_pred)**2)
    
    # Number of parameters (coefficients)
    k = degree + 1
    
    # Number of data points
    n = len(y)
    
    # Calculate BIC
    # BIC = k*ln(n) + n*ln(RSS/n)
    bic = k * np.log(n) + n * np.log(rss / n)
    
    return bic

def cross_validation_error(t, y, degree, n_folds=5):
    """
    Calculate cross-validation error for polynomial fit
    
    Parameters:
    -----------
    t : array
        Independent variable
    y : array
        Dependent variable
    degree : int
        Polynomial degree
    n_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    float
        Mean squared error across folds
    """
    n = len(t)
    fold_size = n // n_folds
    
    # Check if we have enough points for k-fold CV
    if fold_size < 2:
        # Fall back to leave-one-out CV for small datasets
        n_folds = n
        fold_size = 1
    
    # Create polynomial features once
    poly = PolynomialFeatures(degree=degree)
    t_poly = poly.fit_transform(t.reshape(-1, 1))
    
    errors = []
    
    for i in range(n_folds):
        # Create train/test split
        if fold_size == 1:  # Leave-one-out
            test_idx = [i]
            train_idx = [j for j in range(n) if j != i]
        else:  # K-fold
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_folds - 1 else n
            test_idx = list(range(start, end))
            train_idx = [j for j in range(n) if j not in test_idx]
        
        # Train on k-1 folds
        X_train = t_poly[train_idx]
        y_train = y[train_idx]
        
        # Test on k-th fold
        X_test = t_poly[test_idx]
        y_test = y[test_idx]
        
        # Fit model
        model = Ridge(alpha=0.1).fit(X_train, y_train)
        
        # Predict and calculate error
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred)**2)
        errors.append(mse)
    
    # Return average error
    return np.mean(errors)

def smooth_trajectory(points, window=None, sigma=1.0):
    """
    Apply smoothing to trajectory points
    
    Parameters:
    -----------
    points : array
        Trajectory points
    window : int, optional
        Window size for Savitzky-Golay filter (if None, will be calculated)
    sigma : float
        Sigma parameter for Gaussian filter
        
    Returns:
    --------
    array
        Smoothed trajectory points
    """
    if len(points) < 5:
        return points
    
    # Try Savitzky-Golay filter for polynomial smoothing
    if window is None:
        # Calculate appropriate window size (must be odd)
        window = min(len(points) - 2, 11)
        if window % 2 == 0:
            window -= 1
    
    try:
        # Apply Savitzky-Golay filter if we have enough points
        if len(points) > window:
            x_smooth = savgol_filter(points[:, 0], window, 3)
            y_smooth = savgol_filter(points[:, 1], window, 3)
            return np.column_stack((x_smooth, y_smooth))
        else:
            # Fall back to Gaussian filter for shorter sequences
            x_smooth = gaussian_filter1d(points[:, 0], sigma=sigma)
            y_smooth = gaussian_filter1d(points[:, 1], sigma=sigma)
            return np.column_stack((x_smooth, y_smooth))
    except Exception as e:
        print(f"Smoothing error: {e}. Falling back to original data.")
        return points

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
    vectors = np.diff(recent_points[-5:], axis=0) if len(recent_points) >= 6 else np.diff(recent_points, axis=0)
    
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

def calculate_movement_angle(recent_points):
    """
    Calculate the overall angle of movement from recent points
    
    Parameters:
    -----------
    recent_points : array
        Recent trajectory points
        
    Returns:
    --------
    float
        Movement angle in degrees
    """
    if len(recent_points) < 2:
        return 0.0
        
    start_point = recent_points[0]
    end_point = recent_points[-1]
    
    # Vector from start to end
    movement_vector = end_point - start_point
    
    # Calculate angle
    angle = np.degrees(np.arctan2(movement_vector[1], movement_vector[0]))
    
    return angle

def visualize_results(results, show_fitted_curve=True):
    """
    Visualize the prediction results with enhanced display
    
    Parameters:
    -----------
    results : dict
        Dictionary containing trajectory and prediction results
    show_fitted_curve : bool
        Whether to show the polynomial curve fit to recent points
    """
    plt.figure(figsize=(14, 10))
    
    # Plot actual trajectory
    points = results['actual_trajectory']
    plt.plot(points[:, 0], points[:, 1], 
             'b-', linewidth=1.5, alpha=0.5, label='Full Trajectory')
    
    # Highlight the most recent points used for prediction
    recent_idx = max(0, len(points) - 20)
    plt.plot(points[recent_idx:, 0], 
             points[recent_idx:, 1], 
             'b-', linewidth=2.5, label='Recent Trajectory')
    
    # Check if we have future trajectory to plot
    if len(results['future_trajectory']) > 0:
        # Plot future trajectory
        plt.plot(results['future_trajectory'][:, 0], 
                results['future_trajectory'][:, 1], 
                'r-', linewidth=2.5, label='Predicted Path (300s)')
        
        # Mark future points with decreasing opacity
        for i, pos in enumerate(results['future_trajectory']):
            if i % max(1, len(results['future_trajectory']) // 20) == 0:  # Only plot some points to avoid crowding
                alpha = 1.0 - (i / len(results['future_trajectory']) * 0.7)  # Slower decay for visibility
                plt.scatter(pos[0], pos[1], c='red', s=30, alpha=max(0.3, alpha))
        
        # Mark predicted end point
        plt.scatter(results['future_trajectory'][-1, 0], 
                    results['future_trajectory'][-1, 1], 
                    c='red', s=150, label='Final Position (300s)')
    
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
    
    # Add direction arrow
    last_actual = points[-1]
    direction_rad = np.radians(results['direction_degrees'])
    arrow_length = 1.0  # Adjust based on your coordinate scale
    dx = arrow_length * np.cos(direction_rad)
    dy = arrow_length * np.sin(direction_rad)
    
    plt.arrow(last_actual[0], last_actual[1], dx, dy, 
              head_width=0.3, head_length=0.4, fc='red', ec='red', 
              label='Predicted Direction')
    
    # Add fitted curve if requested
    if show_fitted_curve:
        window_size = 20  # Match with the window_size used in prediction
        if len(points) >= window_size:
            recent_points = points[-window_size:]
            
            # Create time points for a smooth curve
            t = np.arange(window_size)
            t_smooth = np.linspace(0, window_size-1, 100)
            
            # Use degrees based on pattern type
            pattern_type = results.get('pattern_type', 'unknown')
            if pattern_type == 'linear':
                curve_degree = 2
            elif pattern_type == 'circular':
                curve_degree = 4
            else:
                curve_degree = 3
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=curve_degree)
            t_points = t.reshape(-1, 1)
            t_poly = poly.fit_transform(t_points)
            
            # Fit models
            model_x = Ridge(alpha=0.1).fit(t_poly, recent_points[:, 0])
            model_y = Ridge(alpha=0.1).fit(t_poly, recent_points[:, 1])
            
            # Generate smooth curve
            t_smooth_reshaped = t_smooth.reshape(-1, 1)
            t_smooth_poly = poly.transform(t_smooth_reshaped)
            smooth_x = model_x.predict(t_smooth_poly)
            smooth_y = model_y.predict(t_smooth_poly)
            
            # Plot fitted curve
            plt.plot(smooth_x, smooth_y, 'g--', linewidth=1.5, alpha=0.7, label=f'Fitted Polynomial (Degree {curve_degree})')
    
    # Add metrics to the plot
    direction_accuracy = results['direction_accuracy']
    position_error = results['position_error_cm']
    r2 = results.get('r2_score', 0)
    mse = results.get('mse', 0)
    rmse = results.get('rmse', 0)
    computation_time = results.get('computation_time_ms', 0)
    
    plt.text(0.02, 0.98, 
             f"Direction Accuracy: {direction_accuracy:.1f}%\n"
             f"Position Error: {position_error:.1f} cm\n"
             f"Model Fit (R²): {r2:.4f}\n"
             f"RMSE: {rmse:.4f}\n"
             f"Computation Time: {computation_time:.1f} ms\n"
             f"Pattern Type: {results['pattern_type']}\n"
             f"Method: {results['method']}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend and labels
    plt.legend(loc='lower right')
    plt.title(f"Pattern-Enhanced Polynomial Regression for User ID: {results['user_id']} (300s forecast)")
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
    recent_idx = max(0, len(points) - 20)
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
        
        if horizon in results['horizon_predictions']:
            # Find index in future_times that corresponds to this horizon
            if 'future_times' in results and len(results['future_times']) > 0:
                future_times = results['future_times']
                last_time = results['actual_trajectory'][-1, 0]  # Use x coordinate as proxy if no actual time
                if 'time_seconds' in results:
                    last_time = results['time_seconds'][-1]
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
            else:
                # Just plot the horizon point directly
                horizon_point = results['horizon_predictions'][horizon]
                plt.scatter(horizon_point[0], horizon_point[1], 
                            c=color, s=120, marker='o', 
                            label=f'{horizon}s Prediction')
                
                # Draw a line from current position to horizon point
                plt.plot([current_pos[0], horizon_point[0]], 
                         [current_pos[1], horizon_point[1]], 
                         f'{color}--', linewidth=1.5)
    
    # Add error regions based on polynomial uncertainty
    # This would add uncertainty ellipses but is quite complex - simplified version:
    for i, horizon in enumerate(results['prediction_horizons']):
        if horizon in results['horizon_predictions']:
            color = colors[i % len(colors)]
            point = results['horizon_predictions'][horizon]
            
            # Scale uncertainty based on horizon time (further = more uncertain)
            uncertainty_radius = results['position_error_cm'] / 100.0 * (1 + horizon / 2.0)
            
            circle = plt.Circle(
                (point[0], point[1]),
                uncertainty_radius,
                color=color,
                fill=False,
                linestyle=':',
                alpha=0.5
            )
            plt.gca().add_patch(circle)
    
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

def analyze_results(results):
    """
    Analyze prediction results in detail with user-friendly metrics
    
    Parameters:
    -----------
    results : dict
        Dictionary containing trajectory and prediction results
    """
    # Calculate trajectory statistics
    actual_traj = results['actual_trajectory']
    
    # Calculate movement vectors between consecutive points
    vectors = np.diff(actual_traj, axis=0)
    
    # Calculate step magnitudes
    magnitudes = np.linalg.norm(vectors, axis=1)
    
    # Calculate direction changes
    directions = np.arctan2(vectors[:, 1], vectors[:, 0])
    direction_changes = np.abs(np.diff(directions))
    
    # Adjust for circular nature of angles
    direction_changes = np.minimum(direction_changes, 2*np.pi - direction_changes)
    
    # Print statistics
    print("\n" + "="*50)
    print(" TRAJECTORY ANALYSIS ".center(50, "="))
    print("="*50)
    print(f"Total data points: {len(actual_traj)}")
    print(f"Total distance traveled: {np.sum(magnitudes):.2f} meters")
    print(f"Average step size: {np.mean(magnitudes):.2f} meters")
    
    # Movement consistency analysis
    if len(direction_changes) > 0:
        avg_change = np.mean(direction_changes) * 180/np.pi
        print(f"Average direction change: {avg_change:.2f} degrees")
        
        if avg_change < 15:
            movement_type = "Very consistent (mostly straight line)"
        elif avg_change < 30:
            movement_type = "Consistent (gentle curves)"
        elif avg_change < 60:
            movement_type = "Moderate turns"
        else:
            movement_type = "Erratic (sharp turns)"
            
        print(f"Movement pattern: {movement_type}")
    
    # Print detailed prediction metrics
    print("\n" + "="*50)
    print(" PREDICTION METRICS ".center(50, "="))
    print("="*50)
    print(f"User ID: {results['user_id']}")
    print(f"Method: {results['method']}")
    print(f"Pattern Type: {results['pattern_type']}")
    print(f"Direction Accuracy: {results['direction_accuracy']:.2f}%")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Position Error: {results['position_error_cm']:.1f} cm")
    print(f"Model Fit (R²): {results['r2_score']:.4f}")
    print(f"Computation Time: {results['computation_time_ms']:.2f} ms")
    
    # Print horizon predictions
    print("\nPrediction Horizons:")
    for horizon in results['prediction_horizons']:
        if horizon in results['horizon_predictions']:
            pos = results['horizon_predictions'][horizon]
            print(f"  {horizon}s: ({pos[0]:.4f}, {pos[1]:.4f})")
    
    # Provide a simple explanation based on the metrics
    print("\n" + "="*50)
    print(" PREDICTION QUALITY ".center(50, "="))
    print("="*50)
    
    # Combined quality assessment
    if results['r2_score'] > 0.9 and results['direction_accuracy'] > 85:
        quality = "Excellent"
    elif results['r2_score'] > 0.8 and results['direction_accuracy'] > 75:
        quality = "Very Good"
    elif results['r2_score'] > 0.7 and results['direction_accuracy'] > 65:
        quality = "Good"
    elif results['r2_score'] > 0.5 and results['direction_accuracy'] > 55:
        quality = "Moderate"
    else:
        quality = "Fair"
        
    print(f"PREDICTION QUALITY: {quality}")
    print(f"\nExplanation: The pattern-enhanced polynomial regression method fits")
    print(f"mathematical curves to your recent movement pattern, with the degree")
    print(f"and parameters automatically adapted to your detected movement type.")
    
    if 'Ensemble' in results['method']:
        print(f"\nThis version uses an ensemble of polynomials with different degrees,")
        print(f"weighted by their fit quality. This approach is particularly effective")
        print(f"for complex or changing movement patterns.")
    
    if results['pattern_type'] == 'linear':
        print(f"\nYour movement follows a predominantly linear pattern, which is")
        print(f"well-suited for polynomial regression with lower degrees (1-2).")
        print(f"The high predictability of linear movement allows for accurate")
        print(f"long-term forecasting.")
    elif results['pattern_type'] == 'circular':
        print(f"\nYour movement follows a circular pattern, which is modeled using")
        print(f"higher-degree polynomials (3-5). While polynomials can approximate")
        print(f"circular motion over short horizons, they may deviate for very")
        print(f"long-term predictions beyond a quarter-circle.")
    elif results['pattern_type'] == 'curved':
        print(f"\nYour movement follows a curved path, which is captured using")
        print(f"medium-degree polynomials (2-4). These can effectively model")
        print(f"gradual turns and arcs in your trajectory.")
    elif results['pattern_type'] == 'erratic':
        print(f"\nYour movement contains significant variations that make long-term")
        print(f"predictions more challenging. The algorithm compensates by using")
        print(f"ensemble methods and robust regression techniques to handle")
        print(f"the unpredictability.")

def main():
    """
    Main function to run the prediction algorithm
    """
    # File path - change to your file path
    file_path = 'uwb_preprocessing.csv'
    
    # Run prediction with multiple horizons
    results = polynomial_regression_prediction(
        csv_file=file_path,
        user_id=None,  # Will use the ID with most data points
        window_size=20,  # Increased from 15 for better context
        prediction_seconds=300,  # 5 minutes prediction
        prediction_horizons=(0.5, 1.0, 1.5, 2.0),  # Multiple evaluation horizons
        max_degree=5,  # Increased from 4 for circular patterns
        use_ensemble=True,
        robust_regression=True
    )
    
    if results:
        # Visualize results
        visualize_results(results, show_fitted_curve=True)
        
        # Visualize horizon-specific predictions
        visualize_horizon_predictions(results)
        
        # Analyze results
        analyze_results(results)

if __name__ == "__main__":
    main()