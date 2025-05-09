import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import time
import os

# Set TensorFlow to only use necessary GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Using GPU: {physical_devices}")
    except:
        print("GPU memory configuration failed")

def lstm_pattern_prediction(csv_file, user_id=None, sequence_length=20, prediction_seconds=300, 
                          prediction_horizons=(0.5, 1.0, 1.5, 2.0), epochs=100, 
                          batch_size=16, use_bidirectional=True):
    """
    Enhanced LSTM prediction with pattern recognition and multiple prediction horizons
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with UWB data
    user_id : int, optional
        ID of the user to track (if None, uses ID with most data points)
    sequence_length : int
        Number of timesteps to use as input sequence
    prediction_seconds : int
        Total number of seconds to predict ahead
    prediction_horizons : tuple
        Time horizons in seconds for evaluation (e.g., 0.5s, 1s, 1.5s, 2s)
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training
    use_bidirectional : bool
        Whether to use bidirectional LSTM layers
        
    Returns:
    --------
    dict
        Dictionary containing trajectory and prediction results with evaluation metrics
    """
    # Start computation time measurement
    start_time = time.time()
    
    print(f"Loading data from {csv_file}...")
    data = pd.read_csv(csv_file)
    
    print("Running Enhanced LSTM with Pattern Recognition...")
    
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
    
    # Need enough data points for LSTM training
    if len(positions) < sequence_length * 3:
        print(f"Not enough data points for LSTM (need at least {sequence_length*3}).")
        return None
    
    # Apply Savitzky-Golay filter to smooth trajectories
    if len(positions) > 10:  # Need enough points for the filter
        window_length = min(11, len(positions) - (1 if len(positions) % 2 == 0 else 0))
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 5:  # Minimum window size for polynomial order 3
            try:
                positions_smooth = np.zeros_like(positions)
                positions_smooth[:, 0] = savgol_filter(positions[:, 0], window_length, 3)
                positions_smooth[:, 1] = savgol_filter(positions[:, 1], window_length, 3)
                positions = positions_smooth
                print(f"Applied smoothing filter with window size {window_length}")
            except Exception as e:
                print(f"Smoothing failed: {e}. Using original data.")
    
    # Calculate time deltas in seconds
    if len(np.unique(time_seconds)) > 1:
        print("Using time_seconds for time delta calculation")
        time_deltas = np.diff(time_seconds)
    else:
        print("Using timestamps for time delta calculation")
        time_deltas = np.diff(timestamps) / 1000.0  # Convert to seconds
    
    avg_time_delta = np.mean(time_deltas)
    print(f"Average time between points: {avg_time_delta:.4f} seconds")
    
    # Handle zero or negative time deltas
    time_deltas[time_deltas <= 0] = avg_time_delta
    
    # Analyze trajectory for pattern detection
    pattern_info = detect_movement_pattern(positions)
    pattern_type = pattern_info['pattern_type']
    print(f"Detected movement pattern: {pattern_type}")
    
    # Prepare features for LSTM
    # We'll include position, velocity, and pattern-specific features
    base_features = positions.copy()
    
    # Calculate velocities
    velocities = np.zeros_like(positions)
    velocities[1:] = np.diff(positions, axis=0) / time_deltas[:, np.newaxis]
    
    # Calculate accelerations
    accelerations = np.zeros_like(positions)
    if len(time_deltas) > 1:
        acc_time_deltas = np.array([time_deltas[:-1] + time_deltas[1:]]).T / 2  # Average of consecutive deltas
        accelerations[2:] = np.diff(velocities[1:], axis=0) / acc_time_deltas
    
    # Prepare base features (position, velocity, acceleration)
    features = np.hstack([positions, velocities, accelerations])
    feature_names = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
    
    # Add pattern-specific features based on pattern type
    if pattern_type == 'circular':
        # For circular motion, add radius and angle from center
        center = pattern_info['center']
        # Calculate distance from center (radius)
        radii = np.sqrt(np.sum((positions - center)**2, axis=1)).reshape(-1, 1)
        # Calculate angle from center
        angles = np.arctan2(positions[:, 1] - center[1], 
                           positions[:, 0] - center[0]).reshape(-1, 1)
        # Add sin and cos of angle for better learning
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)
        
        # Add these features
        pattern_features = np.hstack([radii, sin_angles, cos_angles])
        features = np.hstack([features, pattern_features])
        feature_names.extend(['radius', 'sin_angle', 'cos_angle'])
        
        print("Added circular pattern features (radius, sin/cos of angle)")
    
    elif pattern_type == 'linear':
        # For linear motion, add distance along primary direction
        direction_vector = pattern_info['direction_vector']
        # Project each point onto the direction vector
        projections = np.dot(positions - positions[0], direction_vector).reshape(-1, 1)
        # Add orthogonal distance from the line as well
        ortho_vector = np.array([-direction_vector[1], direction_vector[0]])
        ortho_dist = np.abs(np.dot(positions - positions[0], ortho_vector)).reshape(-1, 1)
        
        pattern_features = np.hstack([projections, ortho_dist])
        features = np.hstack([features, pattern_features])
        feature_names.extend(['proj_dist', 'ortho_dist'])
        
        print("Added linear pattern features (projection, orthogonal distance)")
    
    elif pattern_type in ['curved', 'stop_and_go']:
        # Add curvature features
        if len(velocities) > 2:
            # Calculate tangential and normal components of acceleration
            vel_magnitudes = np.linalg.norm(velocities, axis=1).reshape(-1, 1)
            vel_magnitudes[vel_magnitudes == 0] = 1e-10  # Avoid division by zero
            vel_normalized = velocities / vel_magnitudes
            
            # Tangential acceleration (component along velocity)
            acc_tangential = np.zeros((len(positions), 1))
            acc_normal = np.zeros((len(positions), 1))
            
            for i in range(2, len(positions)):
                # Project acceleration onto velocity direction
                acc_tangential[i] = np.dot(accelerations[i], vel_normalized[i])
                # Calculate normal component (perpendicular to velocity)
                acc_normal[i] = np.linalg.norm(accelerations[i] - 
                                              acc_tangential[i] * vel_normalized[i])
            
            # Add curvature estimation (normal acceleration / velocity^2)
            curvature = np.zeros((len(positions), 1))
            valid_idx = vel_magnitudes[:, 0] > 0.01
            curvature[valid_idx] = acc_normal[valid_idx] / (vel_magnitudes[valid_idx]**2)
            
            pattern_features = np.hstack([acc_tangential, acc_normal, curvature])
            features = np.hstack([features, pattern_features])
            feature_names.extend(['acc_tang', 'acc_norm', 'curvature'])
            
            print("Added curved motion features (tangential/normal acceleration, curvature)")
    
    print(f"Total features: {features.shape[1]} ({', '.join(feature_names)})")
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences for LSTM training
    X, y = [], []
    
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i+sequence_length])
        y.append(scaled_features[i+sequence_length, :2])  # Predict only position (x,y)
    
    X, y = np.array(X), np.array(y)
    print(f"Created {len(X)} training sequences")
    
    # Split into training and validation sets (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"Training set: {len(X_train)} sequences, Validation set: {len(X_val)} sequences")
    
    # Apply data augmentation for small datasets
    if len(X_train) < 100:
        print("Applying data augmentation...")
        X_train, y_train = augment_data(X_train, y_train)
        print(f"Augmented training set: {len(X_train)} sequences")
    
    # Build LSTM model with pattern-specific architecture
    print("Building LSTM model...")
    feature_dim = scaled_features.shape[1]
    model = build_lstm_model(sequence_length, feature_dim, pattern_type, use_bidirectional)
    
    # Define callbacks for training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print(f"Training model for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model on validation set
    val_predictions = model.predict(X_val, verbose=0)
    val_mse = np.mean(np.square(val_predictions - y_val))
    print(f"Validation MSE (scaled): {val_mse:.5f}")
    
    # Create validation predictions in original scale
    val_pred_full = np.zeros((len(val_predictions), feature_dim))
    val_pred_full[:, :2] = val_predictions
    
    val_actual_full = np.zeros((len(y_val), feature_dim))
    val_actual_full[:, :2] = y_val
    
    val_pred = scaler.inverse_transform(val_pred_full)[:, :2]
    val_actual = scaler.inverse_transform(val_actual_full)[:, :2]
    
    # Calculate RMSE in meters
    val_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
    print(f"Validation RMSE: {val_rmse:.4f} meters")
    
    # Make long-term predictions
    print(f"Making predictions for {prediction_seconds} seconds...")
    
    # Calculate total number of prediction steps
    total_steps = int(prediction_seconds / avg_time_delta)
    print(f"Total prediction steps: {total_steps}")
    
    # Ensure we have at least 1 prediction step
    if total_steps <= 0:
        # If avg_time_delta is too large, set a minimum number of steps
        total_steps = max(20, int(prediction_seconds / 1.0))  # Use 1 second as fallback time delta
        print(f"Adjusted prediction steps to: {total_steps}")
    
    # Get the last known sequence
    last_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, feature_dim)
    
    # Prepare for recursive prediction
    future_features = last_sequence[0].copy()  # Last sequence_length feature vectors
    future_positions = []
    future_times = []
    last_time = time_seconds[-1]
    
    # Make recursive predictions
    current_position = positions[-1].copy()
    
    for step in range(total_steps):
        # Predict next position (scaled)
        next_scaled = model.predict(
            future_features.reshape(1, sequence_length, feature_dim),
            verbose=0
        )[0]
        
        # Create full feature vector for next prediction
        next_feature_vector = np.zeros(feature_dim)
        
        # First fill in the position (the predicted values)
        next_feature_vector[:2] = next_scaled
        
        # Calculate time for this step
        step_time = avg_time_delta
        next_time = last_time + ((step + 1) * avg_time_delta)
        
        # Convert predicted position to original scale
        temp_vector = np.zeros(feature_dim)
        temp_vector[:2] = next_scaled
        unscaled_point = scaler.inverse_transform([temp_vector])[0, :2]
        
        # Calculate derivatives for the feature vector
        if step > 0:
            # Velocity (based on position change)
            prev_point = scaler.inverse_transform([future_features[-1]])[0, :2]
            velocity = (unscaled_point - prev_point) / step_time
            
            # Convert velocity to scaled space
            temp_vector = np.zeros(feature_dim)
            temp_vector[2:4] = velocity
            next_feature_vector[2:4] = scaler.transform([temp_vector])[0, 2:4]
            
            # Acceleration (based on velocity change)
            if step > 1:
                prev_velocity = (prev_point - 
                               scaler.inverse_transform([future_features[-2]])[0, :2]) / step_time
                acceleration = (velocity - prev_velocity) / step_time
                
                # Convert acceleration to scaled space
                temp_vector = np.zeros(feature_dim)
                temp_vector[4:6] = acceleration
                next_feature_vector[4:6] = scaler.transform([temp_vector])[0, 4:6]
        else:
            # For first step, use the last known derivatives
            next_feature_vector[2:6] = future_features[-1, 2:6]
        
        # Update pattern-specific features
        if pattern_type == 'circular' and 'radius' in feature_names:
            radius_idx = feature_names.index('radius')
            angle_sin_idx = feature_names.index('sin_angle')
            angle_cos_idx = feature_names.index('cos_angle')
            
            # Calculate new radius and angle based on predicted position
            center = pattern_info['center']
            radius = np.sqrt(np.sum((unscaled_point - center)**2))
            angle = np.arctan2(unscaled_point[1] - center[1], 
                              unscaled_point[0] - center[0])
            
            # Update these features in scaled space
            temp_vector = np.zeros(feature_dim)
            temp_vector[radius_idx] = radius
            temp_vector[angle_sin_idx] = np.sin(angle)
            temp_vector[angle_cos_idx] = np.cos(angle)
            
            scaled_pattern = scaler.transform([temp_vector])[0]
            next_feature_vector[radius_idx] = scaled_pattern[radius_idx]
            next_feature_vector[angle_sin_idx] = scaled_pattern[angle_sin_idx]
            next_feature_vector[angle_cos_idx] = scaled_pattern[angle_cos_idx]
        
        elif pattern_type == 'linear' and 'proj_dist' in feature_names:
            proj_idx = feature_names.index('proj_dist')
            ortho_idx = feature_names.index('ortho_dist')
            
            # Calculate projection distance
            direction_vector = pattern_info['direction_vector']
            projection = np.dot(unscaled_point - positions[0], direction_vector)
            
            # Calculate orthogonal distance
            ortho_vector = np.array([-direction_vector[1], direction_vector[0]])
            ortho_dist = np.abs(np.dot(unscaled_point - positions[0], ortho_vector))
            
            # Update features in scaled space
            temp_vector = np.zeros(feature_dim)
            temp_vector[proj_idx] = projection
            temp_vector[ortho_idx] = ortho_dist
            
            scaled_pattern = scaler.transform([temp_vector])[0]
            next_feature_vector[proj_idx] = scaled_pattern[proj_idx]
            next_feature_vector[ortho_idx] = scaled_pattern[ortho_idx]
        
        # Store prediction
        future_positions.append(unscaled_point)
        future_times.append(next_time)
        
        # Update sequence for next prediction by removing oldest and adding newest
        future_features = np.vstack([future_features[1:], next_feature_vector])
    
    future_positions = np.array(future_positions)
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
        # Fallback if only one prediction is available
        last_actual = positions[-1]
        first_pred = future_positions[0]
        dx = first_pred[0] - last_actual[0]
        dy = first_pred[1] - last_actual[1]
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
    
    # Calculate direction accuracy
    direction_accuracy = calculate_direction_accuracy(positions[-sequence_length:], direction_degrees)
    print(f"Direction accuracy: {direction_accuracy:.2f}%")
    
    # Calculate MSE and RMSE for the predictions against validation data
    if val_rmse is not None:
        print(f"MSE: {val_rmse**2:.4f}")
        print(f"RMSE: {val_rmse:.4f}")
    
    # Finish computation time measurement
    end_time = time.time()
    computation_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Computation time: {computation_time:.2f} ms")
    
    # Return results
    results = {
        'user_id': user_id,
        'actual_trajectory': positions,
        'actual_times': time_seconds,
        'future_trajectory': future_positions,
        'future_times': future_times,
        'direction_degrees': direction_degrees,
        'direction_accuracy': direction_accuracy,
        'mse': val_rmse**2 if val_rmse is not None else None,
        'rmse': val_rmse,
        'computation_time_ms': computation_time,
        'horizon_predictions': horizon_predictions,
        'prediction_horizons': prediction_horizons,
        'pattern_type': pattern_type,
        'history': history.history,
        'method': f'Pattern-Enhanced LSTM Neural Network ({pattern_type.capitalize()})'
    }
    
    return results

def build_lstm_model(sequence_length, feature_dim, pattern_type, use_bidirectional=True):
    """
    Build LSTM model architecture based on detected movement pattern
    
    Parameters:
    -----------
    sequence_length : int
        Length of input sequences
    feature_dim : int
        Number of features
    pattern_type : str
        Type of detected movement pattern
    use_bidirectional : bool
        Whether to use bidirectional LSTM layers
        
    Returns:
    --------
    model : Model
        Compiled Keras model
    """
    model = Sequential()
    
    # Adjust architecture based on pattern type
    if pattern_type == 'linear':
        # Simpler architecture for linear patterns
        if use_bidirectional:
            model.add(Bidirectional(
                LSTM(32, return_sequences=True, 
                     kernel_regularizer=l2(0.001),
                     recurrent_regularizer=l2(0.001)),
                input_shape=(sequence_length, feature_dim)
            ))
        else:
            model.add(LSTM(32, return_sequences=True, 
                          kernel_regularizer=l2(0.001),
                          recurrent_regularizer=l2(0.001),
                          input_shape=(sequence_length, feature_dim)))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        if use_bidirectional:
            model.add(Bidirectional(LSTM(16)))
        else:
            model.add(LSTM(16))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
    elif pattern_type == 'circular':
        # Medium complexity for circular patterns
        if use_bidirectional:
            model.add(Bidirectional(
                LSTM(64, return_sequences=True, 
                     kernel_regularizer=l2(0.001),
                     recurrent_regularizer=l2(0.001)),
                input_shape=(sequence_length, feature_dim)
            ))
        else:
            model.add(LSTM(64, return_sequences=True, 
                          kernel_regularizer=l2(0.001),
                          recurrent_regularizer=l2(0.001),
                          input_shape=(sequence_length, feature_dim)))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        if use_bidirectional:
            model.add(Bidirectional(LSTM(32)))
        else:
            model.add(LSTM(32))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
    else:
        # More complex architecture for erratic or complex patterns
        if use_bidirectional:
            model.add(Bidirectional(
                LSTM(128, return_sequences=True, 
                     kernel_regularizer=l2(0.001),
                     recurrent_regularizer=l2(0.001)),
                input_shape=(sequence_length, feature_dim)
            ))
        else:
            model.add(LSTM(128, return_sequences=True, 
                          kernel_regularizer=l2(0.001),
                          recurrent_regularizer=l2(0.001),
                          input_shape=(sequence_length, feature_dim)))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        if use_bidirectional:
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
        else:
            model.add(LSTM(64, return_sequences=True))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        
        if use_bidirectional:
            model.add(Bidirectional(LSTM(32)))
        else:
            model.add(LSTM(32))
        
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
    
    # Common output layers for all patterns
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(2))  # Output: x, y position
    
    # Compile model with learning rate scheduling
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def augment_data(X, y, noise_level=0.05, shift_ratio=0.1, num_shifts=3):
    """
    Augment training data with noise and shifts
    
    Parameters:
    -----------
    X : array
        Input sequences
    y : array
        Target values
    noise_level : float
        Standard deviation of noise to add
    shift_ratio : float
        Ratio of data to augment with shifts
    num_shifts : int
        Number of shifted samples to create
        
    Returns:
    --------
    X_aug, y_aug : arrays
        Augmented datasets
    """
    X_aug = [X]
    y_aug = [y]
    
    # Add noise
    noise = np.random.normal(0, noise_level, X.shape)
    X_noise = X + noise
    X_aug.append(X_noise)
    y_aug.append(y)
    
    # Create shifted versions
    num_to_shift = int(X.shape[0] * shift_ratio)
    if num_to_shift > 0:
        indices = np.random.choice(X.shape[0], num_to_shift, replace=False)
        
        for i in range(num_shifts):
            # Create a copy of the selected sequences
            X_shift = X[indices].copy()
            y_shift = y[indices].copy()
            
            # Apply shift (moving average)
            for j in range(len(X_shift)):
                shift_noise = np.random.normal(0, noise_level/2, X_shift[j].shape)
                X_shift[j] = X_shift[j] + shift_noise
                
                # Apply small rotation to some sequences
                if np.random.random() < 0.3:
                    angle = np.random.uniform(-0.2, 0.2)  # Small rotation angle
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    
                    # Apply rotation to position coordinates
                    for k in range(len(X_shift[j])):
                        x, y = X_shift[j, k, 0], X_shift[j, k, 1]
                        X_shift[j, k, 0] = x * cos_a - y * sin_a
                        X_shift[j, k, 1] = x * sin_a + y * cos_a
            
            X_aug.append(X_shift)
            y_aug.append(y_shift)
    
    # Concatenate all augmented data
    X_aug = np.vstack(X_aug)
    y_aug = np.vstack(y_aug)
    
    # Shuffle the augmented dataset
    indices = np.arange(len(X_aug))
    np.random.shuffle(indices)
    X_aug = X_aug[indices]
    y_aug = y_aug[indices]
    
    return X_aug, y_aug

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
    recent_idx = max(0, len(points) - 20)
    plt.plot(points[recent_idx:, 0], 
             points[recent_idx:, 1], 
             'b-', linewidth=2.5, label='Recent Trajectory')
    
    # Check if we have future trajectory to plot
    if len(results['future_trajectory']) > 0:
        # Plot future trajectory
        plt.plot(results['future_trajectory'][:, 0], 
                 results['future_trajectory'][:, 1], 
                 'r-', linewidth=2.5, label='Predicted Path')
        
        # Mark predicted future points with decreasing opacity
        for i, pos in enumerate(results['future_trajectory']):
            alpha = 1.0 - (i * 0.01)  # Slower decay for visibility
            if i % 10 == 0:  # Plot every 10th point to avoid crowding
                plt.scatter(pos[0], pos[1], 
                            c='red', s=50, alpha=max(0.3, alpha))
        
        # Mark predicted end point
        plt.scatter(results['future_trajectory'][-1, 0], 
                    results['future_trajectory'][-1, 1], 
                    c='red', s=150, label='Final Position (300s)')
    else:
        print("No future trajectory available to plot")
    
    # Mark start and current points
    plt.scatter(points[0, 0], 
                points[0, 1], 
                c='orange', s=100, label='Start')
    
    plt.scatter(points[-1, 0], 
                points[-1, 1], 
                c='brown', s=150, marker='*', label='Current Position')
    
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
    mse = results.get('mse', 'N/A')
    rmse = results.get('rmse', 'N/A')
    computation_time = results['computation_time_ms']
    
    metrics_text = f"Direction Accuracy: {direction_accuracy:.1f}%\n"
    
    if mse != 'N/A':
        metrics_text += f"MSE: {mse:.4f}\n"
    else:
        metrics_text += "MSE: N/A\n"
        
    if rmse != 'N/A':
        metrics_text += f"RMSE: {rmse:.4f}\n"
    else:
        metrics_text += "RMSE: N/A\n"
        
    metrics_text += f"Computation Time: {computation_time:.2f} ms\n"
    metrics_text += f"Method: {results['method']}"
    
    plt.text(0.02, 0.98, 
             metrics_text,
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
    recent_idx = max(0, len(points) - 20)
    plt.plot(points[recent_idx:, 0], 
             points[recent_idx:, 1], 
             'b-', linewidth=2.5, label='Recent Trajectory')
    
    # Current position
    current_pos = points[-1]
    plt.scatter(current_pos[0], current_pos[1], 
                c='blue', s=150, marker='*', label='Current Position')
    
    # Plot prediction horizons
    colors = ['blue', 'orange', 'red', 'purple']
    
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

def visualize_learning_curves(results):
    """
    Visualize the LSTM training history
    
    Parameters:
    -----------
    results : dict
        Dictionary containing training history
    """
    if 'history' not in results:
        print("No training history available for visualization")
        return
    
    history = results['history']
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate if available
    if 'lr' in history:
        plt.subplot(1, 2, 2)
        plt.semilogy(history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the enhanced LSTM prediction algorithm
    """
    # File path - change to your file path
    file_path = 'uwb_preprocessing.csv'
    
    # Run prediction with multiple horizons
    results = lstm_pattern_prediction(
        csv_file=file_path,
        user_id=None,  # Will use the ID with most data points
        sequence_length=20,  # Use more context for better pattern recognition
        prediction_seconds=300,  # 5 minutes prediction
        prediction_horizons=(0.5, 1.0, 1.5, 2.0),  # Multiple evaluation horizons
        epochs=100,
        batch_size=16,
        use_bidirectional=True
    )
    
    if results:
        # Visualize results
        visualize_results(results)
        
        # Visualize horizon-specific predictions
        visualize_horizon_predictions(results)
        
        # Visualize learning curves
        visualize_learning_curves(results)
        
        # Print detailed metrics report
        print("\n" + "="*50)
        print(" PREDICTION METRICS SUMMARY ".center(50, "="))
        print("="*50)
        print(f"User ID: {results['user_id']}")
        print(f"Method: {results['method']}")
        print(f"Pattern Type: {results['pattern_type']}")
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