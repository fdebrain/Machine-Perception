import os
import time
import json
import tensorflow as tf
from restore_and_evaluate import main as evaluate
from config import config

if __name__ == '__main__':
    # Evaluate model after training and create submission file.
    tf.reset_default_graph()
    config['checkpoint_id'] = './runs/model9_lstm1_512_cnn5_drop5_4e5_avg_loss_1524818174/model-5355'
    evaluate(config)
