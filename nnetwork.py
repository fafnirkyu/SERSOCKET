from trainer import X_train, X_test, y_train, y_test
from organizer import emotion_map
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, 
    Dropout, BatchNormalization, Activation,
    Bidirectional, GRU, GlobalAvgPool1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np

# Reshape X_train and X_test to 3D (samples, n_mfcc, 1)
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (5953, 13, 1)
X_test = np.expand_dims(X_test, axis=-1)    # Shape: (1489, 13, 1)

# Define input_shape
input_shape = (X_train.shape[1], X_train.shape[2])  # (13, 1)

def create_high_accuracy_model(input_shape, num_classes):
    model = Sequential([
        # Block 1: Conv + Pooling
        Conv1D(128, 3, padding='same', kernel_regularizer=l2(0.01), input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        
        # Block 2: Deeper Conv
        Conv1D(256, 3, padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(2),
        Dropout(0.4),
        
        # Block 3: Bidirectional GRU (for temporal patterns)
        Bidirectional(GRU(64, return_sequences=True)),
        Dropout(0.3),
        
        # Block 4: Global Pooling instead of Flatten
        GlobalAvgPool1D(),
        
        # Dense layers
        Dense(128, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize model
model = create_high_accuracy_model(input_shape, len(emotion_map))

# Enhanced callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

# Train with more epochs
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,  # Increased for better convergence
    batch_size=32,
    callbacks=callbacks
)

# Save the model for later use
model.save('emotion_recognition_high_acc.keras')