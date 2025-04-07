#!/usr/bin/env python3
"""
Emotion Recognition Pipeline - Complete Controller
"""

import logging
from pathlib import Path
import time
import argparse
import numpy as np
import sys
from typing import Optional

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmotionRecognitionPipeline:
    def __init__(self, mode: str = 'full'):
        """
        Initialize the complete emotion recognition pipeline
        
        Args:
            mode: Pipeline mode ('full', 'data', 'train', 'predict', or 'implement')
        """
        self.mode = mode
        self.model = None
        self.ready = False
        self.steps = {
            'full': [self.data_preparation, self.model_training, self.implement_prediction],
            'data': [self.data_preparation],
            'train': [self.model_training],
            'predict': [self.predict_websocket],
            'implement': [self.implement_prediction]
        }

    def data_preparation(self):
        """Data download and audio conversion"""
        try:
            logger.info("Starting data preparation phase...")
            
            # Import and run cleanup process
            from cleanup import download_dataset, convert_audio_files
            
            # Download dataset from Kaggle
            download_dataset()
            
            # Convert audio files to standardized format
            convert_audio_files()
            
            logger.info("Data preparation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

    def model_training(self):
        """Feature extraction and model training"""
        try:
            from training.trainer import prepare_training_data
            from nnetwork import create_high_accuracy_model
            from training.organizer import emotion_map
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            logger.info("Starting model training phase...")
            
            # Prepare train/test splits
            global X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = prepare_training_data()
            
            # Reshape data for CNN
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
            
            # Create and train model
            self.model = create_high_accuracy_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_classes=len(emotion_map)
            )
            
            # Train the model
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
            ]
            
            self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks
            )
            
            self.ready = True
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def predict_websocket(self):
        """WebSocket-based real-time prediction"""
        try:
            from predictwebsocket import continuous_listener
            from training.trainer import scaler
            from nnetwork import model

            if not self.ready and self.mode != 'predict':
                raise RuntimeError("Pipeline not trained - run training first")
            
            logger.info("Starting WebSocket prediction...")
            continuous_listener(model, scaler)
            return True
            
        except Exception as e:
            logger.error(f"WebSocket prediction failed: {str(e)}")
            raise

    def implement_prediction(self):
        """Direct microphone implementation prediction"""
        try:
            from implement import continuous_listener
            from training.trainer import scaler
            from nnetwork import model

            if not self.ready and self.mode != 'implement':
                raise RuntimeError("Pipeline not trained - run training first")
            
            logger.info("Starting direct microphone prediction...")
            continuous_listener(model, scaler)
            return True
            
        except Exception as e:
            logger.error(f"Direct prediction failed: {str(e)}")
            raise

    def run(self):
        """Execute the complete pipeline"""
        try:
            logger.info(f"Starting emotion recognition pipeline in {self.mode} mode")
            
            for step in self.steps[self.mode]:
                if not step():
                    logger.error(f"Pipeline failed at step {step.__name__}")
                    return False
                    
            logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.critical(f"Pipeline execution failed: {str(e)}")
            return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Emotion Recognition Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'data', 'train', 'predict', 'implement'],
        default='full',
        help='Pipeline execution mode'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    pipeline = EmotionRecognitionPipeline(mode=args.mode)
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        exit(1)