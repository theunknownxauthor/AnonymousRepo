# Prediction Head
PREDICTION_HEAD_BLOCKS = [2, 4]
PREDICTION_HEAD_DROPOUT = [0.1, 0.2, 0.3]
PREDICTION_HEAD_BOTTLENECK = [2, 4, 8]

# Backbone
FREEZE_BACKBONE = [True]

# Optimization
BATCH_SIZE = [32]
LEARNING_RATE = [5e-4]
WEIGHT_DECAY = [1e-5]

# Training
TRAINING_BUDGET_FACTOR = [10]
EPOCHS = [20]

# Loss
LOSS = [ "mse"]