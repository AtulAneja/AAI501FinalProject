import tensorflow as tf
import numpy as np

def build_model(input_shape):
    """Build a deep learning model."""
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    # Use gradient clipping to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the model with early stopping and learning rate reduction."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data."""
    test_loss = model.evaluate(X_test, y_test)
    return test_loss 