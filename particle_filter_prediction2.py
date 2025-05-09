import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
from sklearn.metrics import mean_squared_error

class PatternEnhancedParticleFilter:
    """
    Particle Filter implementation with pattern recognition for enhanced prediction
    """
    def __init__(self, n_particles=200, process_noise=0.05, measurement_noise=0.1, pattern_type=None):
        """
        Initialize Particle Filter with pattern recognition
        
        Parameters:
        -----------
        n_particles : int
            Number of particles to use
        process_noise : float
            Process noise parameter (higher = more random movement)
        measurement_noise : float
            Measurement noise parameter (higher = less trust in measurements)
        pattern_type : str
            Detected movement pattern type (linear, circular, etc.)
        """
        self.n_particles = n_particles
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.particles = None
        self.weights = None
        self.velocities = None
        self.accelerations = None
        self.pattern_type = pattern_type
        self.pattern_params = {}
        
    def initialize(self, position):
        """
        Initialize particles around the initial position
        
        Parameters:
        -----------
        position : array
            Initial position [x, y]
        """
        self.particles = np.tile(position, (self.n_particles, 1)) + \
                         np.random.normal(0, self.process_noise, size=(self.n_particles, 2))
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.velocities = np.zeros((self.n_particles, 2))
        self.accelerations = np.zeros((self.n_particles, 2))
        
    def predict(self, dt=1.0, max_speed=2.0):
        """
        Move particles according to their velocities with pattern-based adjustments
        
        Parameters:
        -----------
        dt : float
            Time step
        max_speed : float
            Maximum speed constraint
        """
        # Apply pattern-specific dynamics based on detected pattern
        if self.pattern_type == 'circular' and 'center' in self.pattern_params and 'radius' in self.pattern_params:
            # For circular pattern, incorporate circular motion model
            center = self.pattern_params['center']
            radius = self.pattern_params['radius']
            angular_velocity = self.pattern_params.get('angular_velocity', 0.5)  # rad/s
            rotation = self.pattern_params.get('rotation', 'counterclockwise')
            rotation_factor = 1.0 if rotation == 'counterclockwise' else -1.0
            
            # For each particle, calculate position relative to the circle center
            for i in range(self.n_particles):
                # Current vector from center
                vec_to_center = self.particles[i] - center
                # Current angle
                angle = np.arctan2(vec_to_center[1], vec_to_center[0])
                # Update angle based on angular velocity
                angle += rotation_factor * angular_velocity * dt
                # Add noise to radius
                current_radius = np.linalg.norm(vec_to_center)
                noise_radius = current_radius + np.random.normal(0, self.process_noise * radius)
                # Set new position
                self.particles[i] = center + noise_radius * np.array([np.cos(angle), np.sin(angle)])
                # Update velocity based on this motion
                new_vec = self.particles[i] - center
                new_angle = np.arctan2(new_vec[1], new_vec[0])
                tangent_direction = np.array([-np.sin(new_angle), np.cos(new_angle)]) * rotation_factor
                self.velocities[i] = tangent_direction * angular_velocity * noise_radius
        
        elif self.pattern_type == 'linear' and 'direction_vector' in self.pattern_params:
            # For linear pattern, add bias toward the main direction
            direction = self.pattern_params['direction_vector']
            direction_strength = 0.7  # How strongly to bias toward the direction
            
            # Add random noise to velocities (process noise)
            self.velocities += np.random.normal(0, self.process_noise, size=(self.n_particles, 2))
            
            # Add direction bias (align particles more with the main direction)
            for i in range(self.n_particles):
                random_factor = np.random.uniform(0, direction_strength)
                speed = np.linalg.norm(self.velocities[i])
                if speed > 0.01:  # Only if moving
                    # Blend current velocity with direction
                    self.velocities[i] = (1 - random_factor) * self.velocities[i] + \
                                         random_factor * direction * speed
        
        else:
            # Standard motion model with velocity and acceleration
            # Update velocity based on acceleration
            self.velocities += self.accelerations * dt
            
            # Add random noise to velocities (process noise)
            self.velocities += np.random.normal(0, self.process_noise, size=(self.n_particles, 2))
        
        # Apply speed limit to velocities
        speeds = np.linalg.norm(self.velocities, axis=1)
        # Find particles exceeding max_speed
        idx = speeds > max_speed
        if np.any(idx):
            # Scale down velocities to max_speed while preserving direction
            self.velocities[idx] = self.velocities[idx] * (max_speed / speeds[idx, np.newaxis])
        
        # Move particles according to their velocities
        self.particles += self.velocities * dt
        
        # Apply damping factor to velocities and accelerations
        damping = 0.98
        self.velocities *= damping
        self.accelerations *= damping * 0.95  # Stronger damping for acceleration
    
    def update(self, measurement):
        """
        Update particle weights based on measurement
        
        Parameters:
        -----------
        measurement : array
            Measured position [x, y]
        """
        # Calculate distances between particles and measurement
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        
        # Calculate weights using Gaussian likelihood
        self.weights *= np.exp(-0.5 * (distances**2) / (self.measurement_noise**2))
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
            
        # Calculate effective sample size
        n_eff = 1.0 / np.sum(self.weights**2)
        
        # Resample if effective sample size is too low
        if n_eff < self.n_particles / 2:
            self.resample()
            return True  # Return True if resampling occurred
        
        return False
    
    def resample(self):
        """
        Resample particles based on weights
        """
        # Sample indices with probability proportional to weights
        indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
        
        # Resample particles, velocities, and accelerations
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        self.accelerations = self.accelerations[indices]
        
        # Add small jitter to avoid sample impoverishment
        self.particles += np.random.normal(0, self.process_noise * 0.5, size=(self.n_particles, 2))
        
        # Reset weights
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def update_velocity(self, old_pos, new_pos, dt):
        """
        Update particle velocities based on position changes
        
        Parameters:
        -----------
        old_pos : array
            Previous estimated position
        new_pos : array
            Current estimated position
        dt : float
            Time step
        """
        if dt > 0:
            # Calculate current velocity from position change
            current_velocity = (new_pos - old_pos) / dt
            
            # Add current velocity information to particles
            for i in range(self.n_particles):
                # Calculate particle-specific velocity
                particle_velocity = self.velocities[i]
                
                # Blend with current velocity estimate (velocity feedback)
                alpha = 0.3  # Blend factor
                self.velocities[i] = (1 - alpha) * particle_velocity + alpha * current_velocity
    
    def update_acceleration(self, old_vel, new_vel, dt):
        """
        Update particle accelerations based on velocity changes
        
        Parameters:
        -----------
        old_vel : array
            Previous velocity estimate
        new_vel : array
            Current velocity estimate
        dt : float
            Time step
        """
        if dt > 0:
            # Calculate current acceleration from velocity change
            current_acceleration = (new_vel - old_vel) / dt
            
            # Add to particles
            for i in range(self.n_particles):
                # Calculate particle-specific acceleration
                particle_acceleration = self.accelerations[i]
                
                # Blend with current acceleration estimate
                alpha = 0.2  # Lower blend factor for acceleration
                self.accelerations[i] = (1 - alpha) * particle_acceleration + alpha * current_acceleration
    
    def estimate_state(self):
        """
        Estimate current state (position) as weighted average of particles
        
        Returns:
        --------
        array
            Estimated position [x, y]
        """
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def estimate_velocity(self):
        """
        Estimate current velocity as weighted average of particle velocities
        
        Returns:
        --------
        array
            Estimated velocity [vx, vy]
        """
        return np.average(self.velocities, weights=self.weights, axis=0)
    
    def estimate_acceleration(self):
        """
        Estimate current acceleration as weighted average of particle accelerations
        
        Returns:
        --------
        array
            Estimated acceleration [ax, ay]
        """
        return np.average(self.accelerations, weights=self.weights, axis=0)
    
    def get_uncertainty(self):
        """
        Calculate position uncertainty as weighted standard deviation
        
        Returns:
        --------
        float
            Position uncertainty in meters
        """
        mean_pos = self.estimate_state()
        variance = np.average(np.sum((self.particles - mean_pos)**2, axis=1), weights=self.weights)
        return np.sqrt(variance)
    
    def get_directional_uncertainty(self):
        """
        Calculate directional uncertainty by analyzing particle spread
        
        Returns:
        --------
        float
            Directional uncertainty in degrees
        """
        velocities = self.velocities
        # Consider only particles with significant velocity
        valid_idx = np.linalg.norm(velocities, axis=1) > 0.01
        if np.sum(valid_idx) < 2:
            return 45.0  # Default if not enough valid velocities
        
        valid_velocities = velocities[valid_idx]
        valid_weights = self.weights[valid_idx] / np.sum(self.weights[valid_idx])
        
        # Calculate weighted mean direction
        directions = np.arctan2(valid_velocities[:, 1], valid_velocities[:, 0])
        mean_direction = np.average(directions, weights=valid_weights)
        
        # Calculate circular variance
        circular_var = 1.0 - np.abs(np.average(np.exp(1j * directions), weights=valid_weights))
        
        # Convert to degrees (0 = perfect certainty, 90 = maximum uncertainty)
        direction_uncertainty = np.degrees(np.sqrt(circular_var) * np.pi/2)
        
        return direction_uncertainty
    
    def set_pattern_params(self, pattern_info):
        """
        Set pattern-specific parameters for motion prediction
        
        Parameters:
        -----------
        pattern_info : dict
            Dictionary containing pattern parameters
        """
        self.pattern_type = pattern_info['pattern_type']
        self.pattern_params = {k: v for k, v in pattern_info.items() if k != 'pattern_type'}


def particle_filter_prediction(csv_file, user_id=None, n_particles=300, prediction_seconds=300, 
                              prediction_horizons=(0.5, 1.0, 1.5, 2.0), movement_std=0.1):
    """
    Predict trajectory using pattern-enhanced particle filter
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with UWB data
    user_id : int, optional
        ID of the user to track (if None, uses ID with most data points)
    n_particles : int
        Number of particles to use in the filter
    prediction_seconds : int
        Total seconds to predict into the future
    prediction_horizons : tuple
        Time horizons in seconds for evaluation
    movement_std : float
        Standard deviation of movement model
        
    Returns:
    --------
    dict
        Dictionary containing trajectory and prediction results with evaluation metrics
    """
    # Start computation time measurement
    start_time = time.time()
    
    print(f"Loading data from {csv_file}...")
    data = pd.read_csv(csv_file)
    
    print(f"Running Enhanced Particle Filter prediction with {n_particles} particles...")
    
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
    
    # Apply smoothing to reduce noise
    try:
        positions_xy = smooth_trajectory(positions)
        positions = positions_xy  # Use smoothed positions
        print("Applied smoothing to trajectory for more stable tracking")
    except Exception as e:
        print(f"Smoothing failed: {e}. Using original trajectory.")
    
    # Need at least 2 points
    if len(positions) < 2:
        print("Not enough data points for Particle Filter (need at least 2).")
        return None
    
    # Analyze trajectory for pattern detection
    pattern_info = detect_movement_pattern(positions)
    pattern_type = pattern_info['pattern_type']
    print(f"Detected movement pattern: {pattern_type}")
    
    # Calculate time deltas
    if len(np.unique(time_seconds)) > 1:
        print("Using time_seconds for time delta calculation")
        time_deltas = np.diff(time_seconds)
    else:
        print("Using timestamps for time delta calculation")
        time_deltas = np.diff(timestamps) / 1000.0  # Convert to seconds
    
    # Handle zero or negative time deltas
    time_deltas[time_deltas <= 0] = 0.1  # Default 100ms
    avg_time_delta = np.mean(time_deltas)
    print(f"Average time between points: {avg_time_delta:.4f} seconds")
    
    # Initialize particle filter with pattern information
    pf = PatternEnhancedParticleFilter(
        n_particles=n_particles, 
        process_noise=movement_std,
        measurement_noise=movement_std*2,
        pattern_type=pattern_type
    )
    
    # Set pattern-specific parameters
    pf.set_pattern_params(pattern_info)
    
    # Initialize with first position
    pf.initialize(positions[0])
    
    # Track all filter states
    filtered_positions = [pf.estimate_state()]
    filtered_velocities = [pf.estimate_velocity()]
    uncertainties = [pf.get_uncertainty()]
    
    # Process all measurements
    resampling_count = 0
    
    for i in range(1, len(positions)):
        # Calculate time delta
        dt = time_deltas[i-1]
        
        # Predict step
        pf.predict(dt=dt)
        
        # Update step
        resampling_occurred = pf.update(positions[i])
        if resampling_occurred:
            resampling_count += 1
        
        # Store filter state
        new_pos = pf.estimate_state()
        new_vel = pf.estimate_velocity()
        
        # Update velocity if we have previous estimates
        if i > 1:
            pf.update_velocity(filtered_positions[-1], new_pos, dt)
            
            # Update acceleration if we have previous velocities
            if i > 2:
                pf.update_acceleration(filtered_velocities[-1], new_vel, dt)
        
        filtered_positions.append(new_pos)
        filtered_velocities.append(new_vel)
        uncertainties.append(pf.get_uncertainty())
    
    # Convert to numpy arrays
    filtered_positions = np.array(filtered_positions)
    filtered_velocities = np.array(filtered_velocities)
    uncertainties = np.array(uncertainties)
    
    print(f"Processed {len(positions)} positions with {resampling_count} resampling events")
    print(f"Average uncertainty: {np.mean(uncertainties):.4f} meters")
    
    # Predict future trajectories
    print(f"Making predictions for {prediction_seconds} seconds...")
    
    # Calculate total number of prediction steps
    total_steps = int(prediction_seconds / avg_time_delta)
    print(f"Total prediction steps: {total_steps}")
    
    # Ensure we have at least 1 prediction step
    if total_steps <= 0:
        # If avg_time_delta is too large, set a minimum number of steps
        total_steps = max(20, int(prediction_seconds / 1.0))  # Use 1 second as fallback
        print(f"Adjusted prediction steps to: {total_steps}")
    
    # Use a copy of the particle filter for prediction
    pf_predict = PatternEnhancedParticleFilter(
        n_particles=n_particles,
        process_noise=movement_std,
        measurement_noise=movement_std*2,
        pattern_type=pattern_type
    )
    
    # Set pattern-specific parameters
    pf_predict.set_pattern_params(pattern_info)
    
    # Initialize with final state
    pf_predict.particles = pf.particles.copy()
    pf_predict.velocities = pf.velocities.copy()
    pf_predict.accelerations = pf.accelerations.copy()
    pf_predict.weights = pf.weights.copy()
    
    # Prepare for prediction
    future_positions = []
    future_uncertainties = []
    future_times = []
    last_time = time_seconds[-1]
    
    # Apply adaptive process noise for prediction
    future_noise = movement_std
    
    for step in range(total_steps):
        # Increase noise slightly for each future step (accounting for growing uncertainty)
        future_noise *= 1.05
        pf_predict.process_noise = future_noise
        
        # Predict next position
        step_time = avg_time_delta
        future_time = last_time + ((step + 1) * avg_time_delta)
        
        pf_predict.predict(dt=step_time)
        
        # Get predicted position and uncertainty
        predicted_position = pf_predict.estimate_state()
        predicted_uncertainty = pf_predict.get_uncertainty()
        
        # Store prediction
        future_positions.append(predicted_position)
        future_uncertainties.append(predicted_uncertainty)
        future_times.append(future_time)
        
        # Log prediction (but not too frequently)
        if step % max(1, total_steps // 10) == 0 or step == total_steps - 1:
            print(f"Step {step+1}/{total_steps}: Predicted position = [{predicted_position[0]:.4f}, {predicted_position[1]:.4f}], "
                  f"Uncertainty = {predicted_uncertainty:.4f}m")
    
    future_positions = np.array(future_positions)
    future_uncertainties = np.array(future_uncertainties)
    future_times = np.array(future_times)
    
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
    
    # Calculate final metrics
    # Direction calculated from estimated velocity
    final_velocity = pf.estimate_velocity()
    direction_degrees = np.degrees(np.arctan2(final_velocity[1], final_velocity[0]))
    direction_uncertainty = pf.get_directional_uncertainty()
    
    # Calculate direction accuracy
    direction_accuracy = calculate_direction_accuracy(
        positions[-min(10, len(positions)):], 
        direction_degrees
    )
    
    # Calculate position uncertainty
    position_uncertainty = pf.get_uncertainty()
    
    # Calculate mean squared error between filtered and actual positions
    mse = mean_squared_error(positions, filtered_positions[:len(positions)])
    rmse = np.sqrt(mse)
    
    # Calculate overall movement angle
    movement_angle = calculate_movement_angle(positions)
    
    # Finish computation time measurement
    end_time = time.time()
    computation_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Computation time: {computation_time:.2f} ms")
    
    # Return results
    results = {
        'user_id': user_id,
        'actual_trajectory': positions,
        'filtered_trajectory': filtered_positions,
        'future_trajectory': future_positions,
        'future_uncertainties': future_uncertainties,
        'future_times': future_times,
        'direction_degrees': direction_degrees,
        'direction_uncertainty': direction_uncertainty,
        'direction_accuracy': direction_accuracy,
        'position_error_cm': position_uncertainty * 100,  # Convert to cm
        'movement_angle': movement_angle,
        'uncertainty': np.mean(uncertainties),
        'mse': mse,
        'rmse': rmse,
        'computation_time_ms': computation_time,
        'horizon_predictions': horizon_predictions,
        'prediction_horizons': prediction_horizons,
        'pattern_type': pattern_type,
        'particles': pf.particles,
        'weights': pf.weights,
        'method': f'Pattern-Enhanced Particle Filter ({n_particles} particles, {pattern_type} pattern)'
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

def smooth_trajectory(points, window=None):
    """
    Apply smoothing to trajectory points
    
    Parameters:
    -----------
    points : array
        Trajectory points
    window : int, optional
        Window size for Savitzky-Golay filter (if None, will be calculated)
        
    Returns:
    --------
    array
        Smoothed trajectory points
    """
    if len(points) < 5:
        return points
    
    # Calculate appropriate window size (must be odd)
    if window is None:
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
            return points
    except Exception as e:
        print(f"Smoothing error: {e}. Returning original points.")
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

def calculate_movement_angle(points):
    """
    Calculate the overall angle of movement from all points
    
    Parameters:
    -----------
    points : array
        Trajectory points
        
    Returns:
    --------
    float
        Movement angle in degrees
    """
    if len(points) < 2:
        return 0.0
        
    start_point = points[0]
    end_point = points[-1]
    
    # Vector from start to end
    movement_vector = end_point - start_point
    
    # Calculate angle
    angle = np.degrees(np.arctan2(movement_vector[1], movement_vector[0]))
    
    return angle

def visualize_results(results, show_particles=True):
    """
    Visualize the prediction results
    
    Parameters:
    -----------
    results : dict
        Dictionary containing trajectory and prediction results
    show_particles : bool
        Whether to show final particle distribution
    """
    plt.figure(figsize=(14, 10))
    
    # Plot actual trajectory
    plt.plot(results['actual_trajectory'][:, 0], 
             results['actual_trajectory'][:, 1], 
             'b.', alpha=0.4, markersize=4, label='Raw Measurements')
    
    # Plot filtered trajectory
    plt.plot(results['filtered_trajectory'][:, 0], 
             results['filtered_trajectory'][:, 1], 
             'g-', linewidth=2, label='Particle Filter Path')
    
    # Plot final particles if requested
    if show_particles and 'particles' in results and 'weights' in results:
        particles = results['particles']
        weights = results['weights']
        
        # Scale point sizes based on weights
        max_weight = np.max(weights)
        if max_weight > 0:
            sizes = 50 * weights / max_weight
        else:
            sizes = np.ones(len(weights)) * 10
            
        plt.scatter(particles[:, 0], particles[:, 1], 
                   c='orange', alpha=0.5, s=sizes, label='Particles')
    
    # Check if we have future trajectory to plot
    if len(results['future_trajectory']) > 0:
        # Plot future trajectory
        plt.plot(results['future_trajectory'][:, 0], 
                 results['future_trajectory'][:, 1], 
                 'r-', linewidth=2.5, label='Predicted Path')
        
        # Mark predicted future points with decreasing opacity
        for i, pos in enumerate(results['future_trajectory']):
            if i % 10 == 0:  # Only plot every 10th point to avoid crowding
                alpha = 1.0 - (i * 0.01)  # Slower decay for visibility
                plt.scatter(pos[0], pos[1], 
                            c='red', s=50, alpha=max(0.3, alpha))
        
        # Mark predicted end point
        plt.scatter(results['future_trajectory'][-1, 0], 
                    results['future_trajectory'][-1, 1], 
                    c='red', s=150, label='Predicted End (300s)')
    else:
        print("No future trajectory available to plot")
    
    # Mark start and current points
    plt.scatter(results['actual_trajectory'][0, 0], 
                results['actual_trajectory'][0, 1], 
                c='green', s=100, label='Start')
    
    plt.scatter(results['filtered_trajectory'][-1, 0], 
                results['filtered_trajectory'][-1, 1], 
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
    last_filtered = results['filtered_trajectory'][-1]
    direction_rad = np.radians(results['direction_degrees'])
    arrow_length = 1.0  # Adjust based on your coordinate scale
    dx = arrow_length * np.cos(direction_rad)
    dy = arrow_length * np.sin(direction_rad)
    
    plt.arrow(last_filtered[0], last_filtered[1], dx, dy, 
              head_width=0.3, head_length=0.4, fc='red', ec='red', 
              label='Predicted Direction')
    
    # Add uncertainty circles at different horizons
    if 'future_uncertainties' in results and len(results['future_uncertainties']) > 0:
        # Plot first horizon uncertainty
        if 0 < len(results['future_trajectory']):
            first_uncertain = results['future_uncertainties'][0]
            first_pos = results['future_trajectory'][0]
            
            circle1 = plt.Circle(
                (first_pos[0], first_pos[1]),
                first_uncertain,
                color='red',
                fill=False,
                linestyle='--',
                alpha=0.7,
                label=f'Initial Uncertainty ({first_uncertain:.2f}m)'
            )
            plt.gca().add_patch(circle1)
        
        # Plot last horizon uncertainty
        if 1 < len(results['future_trajectory']):
            last_uncertain = results['future_uncertainties'][-1]
            last_pos = results['future_trajectory'][-1]
            
            circle2 = plt.Circle(
                (last_pos[0], last_pos[1]),
                last_uncertain,
                color='purple',
                fill=False,
                linestyle=':',
                alpha=0.7,
                label=f'Final Uncertainty ({last_uncertain:.2f}m)'
            )
            plt.gca().add_patch(circle2)
    
    # Add metrics to the plot
    direction_accuracy = results['direction_accuracy']
    position_error = results['position_error_cm']
    uncertainty = results['uncertainty'] * 100  # Convert to cm
    mse = results['mse']
    rmse = results['rmse']
    computation_time = results['computation_time_ms']
    
    plt.text(0.02, 0.98, 
             f"Direction Accuracy: {direction_accuracy:.1f}%\n"
             f"Position Error: {position_error:.1f} cm\n"
             f"Tracking Uncertainty: {uncertainty:.1f} cm\n"
             f"MSE: {mse:.4f}\n"
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
    plt.title(f"Pattern-Enhanced Particle Filter Prediction for User ID: {results['user_id']} (300s forecast)")
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
    plt.plot(results['actual_trajectory'][:, 0], 
             results['actual_trajectory'][:, 1], 
             'b-', linewidth=1.5, alpha=0.5, label='Full Trajectory')
    
    # Highlight the most recent points used for prediction
    recent_idx = max(0, len(results['actual_trajectory']) - 20)
    plt.plot(results['actual_trajectory'][recent_idx:, 0], 
             results['actual_trajectory'][recent_idx:, 1], 
             'b-', linewidth=2.5, label='Recent Trajectory')
    
    # Current position
    current_pos = results['filtered_trajectory'][-1]
    plt.scatter(current_pos[0], current_pos[1], 
                c='blue', s=150, marker='*', label='Current Position')
    
    # Plot prediction horizons
    colors = ['green', 'orange', 'red', 'purple']
    
    # Draw horizon-specific trajectories
    for i, horizon in enumerate(results['prediction_horizons']):
        color = colors[i % len(colors)]
        
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
                
                # Add uncertainty circle if available
                if 'future_uncertainties' in results and closest_idx < len(results['future_uncertainties']):
                    uncertainty = results['future_uncertainties'][closest_idx]
                    circle = plt.Circle(
                        (horizon_point[0], horizon_point[1]),
                        uncertainty,
                        color=color,
                        fill=False,
                        linestyle='--',
                        alpha=0.5,
                        label=f'{horizon}s Uncertainty'
                    )
                    plt.gca().add_patch(circle)
        
        # If horizon is directly available in horizon_predictions
        elif horizon in results['horizon_predictions']:
            horizon_point = results['horizon_predictions'][horizon]
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
    filtered_traj = results['filtered_trajectory']
    
    # Calculate movement vectors between consecutive filtered points
    vectors = np.diff(filtered_traj, axis=0)
    
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
    
    if len(magnitudes) > 0:
        print(f"Total filtered distance: {np.sum(magnitudes):.2f} meters")
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
    
    # Particle filter performance analysis
    uncertainty = results.get('uncertainty', 0)
    print(f"\nAverage particle filter uncertainty: {uncertainty*100:.2f} cm")
    
    # Calculate filtered path error (comparing filtered to actual)
    errors = []
    for i in range(min(len(filtered_traj), len(actual_traj))):
        error = np.linalg.norm(filtered_traj[i] - actual_traj[i])
        errors.append(error)
    
    if errors:
        avg_error = np.mean(errors)
        print(f"Average tracking error: {avg_error*100:.2f} cm")
    
    # Print detailed prediction metrics
    print("\n" + "="*50)
    print(" PREDICTION METRICS ".center(50, "="))
    print("="*50)
    print(f"User ID: {results['user_id']}")
    print(f"Method: {results['method']}")
    print(f"Pattern Type: {results['pattern_type']}")
    print(f"Direction Accuracy: {results['direction_accuracy']:.2f}%")
    print(f"Direction Uncertainty: {results.get('direction_uncertainty', 'N/A')}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Position Error: {results['position_error_cm']:.1f} cm")
    print(f"Computation Time: {results['computation_time_ms']:.2f} ms")
    
    # Print horizon predictions
    print("\nPrediction Horizons:")
    for horizon in results['prediction_horizons']:
        if horizon in results['horizon_predictions']:
            pos = results['horizon_predictions'][horizon]
            print(f"  {horizon}s: ({pos[0]:.4f}, {pos[1]:.4f})")
    
    # Uncertainty growth analysis
    if 'future_uncertainties' in results and len(results['future_uncertainties']) > 0:
        initial_uncertainty = results['future_uncertainties'][0]
        final_uncertainty = results['future_uncertainties'][-1]
        uncertainty_growth = (final_uncertainty / initial_uncertainty - 1) * 100
        
        print(f"\nUncertainty growth over prediction period: {uncertainty_growth:.1f}%")
        print(f"Initial uncertainty: {initial_uncertainty*100:.1f} cm")
        print(f"Final uncertainty: {final_uncertainty*100:.1f} cm")
    
    # Provide a simple explanation based on the metrics
    print("\n" + "="*50)
    print(" PREDICTION QUALITY ".center(50, "="))
    print("="*50)
    if results['direction_accuracy'] > 85 and results['position_error_cm'] < 15:
        quality = "Excellent"
    elif results['direction_accuracy'] > 75 and results['position_error_cm'] < 30:
        quality = "Good"
    elif results['direction_accuracy'] > 60 and results['position_error_cm'] < 50:
        quality = "Moderate"
    else:
        quality = "Fair"
        
    print(f"PREDICTION QUALITY: {quality}")
    print(f"\nExplanation: The pattern-enhanced particle filter combines statistical")
    print(f"filtering with movement pattern recognition to improve prediction accuracy.")
    
    if quality in ["Excellent", "Good"]:
        print(f"\nThe filter has successfully recognized your {results['pattern_type']} movement")
        print(f"pattern and is accurately predicting your trajectory. The prediction")
        print(f"shows high confidence with low uncertainty.")
    elif quality == "Moderate":
        print(f"\nThe filter has identified a {results['pattern_type']} pattern in your movement")
        print(f"with moderate confidence. There are some elements of unpredictability")
        print(f"causing increased uncertainty in the long-term predictions.")
    else:
        print(f"\nYour movement contains significant variability, making long-term")
        print(f"predictions challenging. The particle filter is representing multiple")
        print(f"possible futures, resulting in higher uncertainty.")
    
    # Comparison with other methods
    print("\n" + "="*50)
    print(" METHODOLOGY COMPARISON ".center(50, "="))
    print("="*50)
    
    pattern_type = results['pattern_type']
    
    if pattern_type == 'linear':
        print("Your movement follows a predominantly linear pattern, which is:")
        print("- WELL-SUITED for this particle filter approach")
        print("- Also well-suited for simpler methods like linear regression")
        print("- Better than purely kinematic prediction for long horizons")
        
    elif pattern_type == 'circular':
        print("Your movement follows a circular/curved pattern, which is:")
        print("- VERY WELL-SUITED for this pattern-enhanced particle filter")
        print("- Less suited for purely kinematic predictions")
        print("- Could also be modeled with specialized circle-fitting approaches")
        
    elif pattern_type in ['curved', 'stop_and_go']:
        print(f"Your movement follows a {pattern_type} pattern, which is:")
        print("- WELL-SUITED for particle filtering approaches")
        print("- Challenging for simple kinematic or linear models")
        print("- Requires the stochastic nature of particles to model uncertainty")
        
    else:
        print(f"Your movement follows an {pattern_type} pattern, which is:")
        print("- Best approached with particle filtering or ensemble methods")
        print("- Too unpredictable for simple deterministic models")
        print("- Requires representing multiple possible future trajectories")

def main():
    """
    Main function to run the enhanced particle filter prediction
    """
    # File path - change to your file path
    file_path = 'uwb_preprocessing.csv'
    
    # Run prediction with multiple horizons
    results = particle_filter_prediction(
        csv_file=file_path,
        user_id=None,  # Will use the ID with most data points
        n_particles=150,  # Increased from 200 for better pattern modeling
        prediction_seconds=150,  # 5 minutes prediction
        prediction_horizons=(0.5, 1.0, 1.5, 2.0),  # Multiple evaluation horizons
        movement_std=0.1  # Process noise (adjust based on movement variability)
    )
    
    if results:
        # Visualize results
        visualize_results(results, show_particles=True)
        
        # Visualize horizon-specific predictions
        visualize_horizon_predictions(results)
        
        # Analyze results
        analyze_results(results)
        
if __name__ == "__main__":
    main()