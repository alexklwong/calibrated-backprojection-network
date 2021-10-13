# Batch settings
N_BATCH                                     = 8
N_HEIGHT                                    = 320
N_WIDTH                                     = 768

# Input settings
INPUT_CHANNELS_IMAGE                        = 3
INPUT_CHANNELS_DEPTH                        = 2
NORMALIZED_IMAGE_RANGE                      = [0, 1]
OUTLIER_REMOVAL_KERNEL_SIZE                 = 7
OUTLIER_REMOVAL_THRESHOLD                   = 1.5

# Sparse to dense pool settings
MIN_POOL_SIZES_SPARSE_TO_DENSE_POOL         = [5, 7, 9, 11, 13]
MAX_POOL_SIZES_SPARSE_TO_DENSE_POOL         = [15, 17]
N_CONVOLUTION_SPARSE_TO_DENSE_POOL          = 3
N_FILTER_SPARSE_TO_DENSE_POOL               = 8

# Depth network settings
N_FILTERS_ENCODER_IMAGE                     = [48, 96, 192, 384, 384]
N_FILTERS_ENCODER_DEPTH                     = [16, 32, 64, 128, 128]
RESOLUTIONS_BACKPROJECTION                  = [0, 1, 2, 3]
N_FILTERS_DECODER                           = [256, 128, 128, 64, 12]
DECONV_TYPE                                 = 'up'
MIN_PREDICT_DEPTH                           = 1.5
MAX_PREDICT_DEPTH                           = 100.0

# Weight settings
WEIGHT_INITIALIZER                          = 'xavier_normal'
ACTIVATION_FUNC                             = 'leaky_relu'

# Training settings
LEARNING_RATES                              = [5e-5, 1e-4, 15e-5, 1e-4, 5e-5, 2e-5]
LEARNING_SCHEDULE                           = [2, 8, 20, 30, 45, 60]
AUGMENTATION_PROBABILITIES                  = [1.00, 0.50, 0.25]
AUGMENTATION_SCHEDULE                       = [50, 55, 60]
AUGMENTATION_RANDOM_CROP_TYPE               = ['horizontal', 'vertical', 'anchored', 'bottom']
AUGMENTATION_RANDOM_FLIP_TYPE               = ['none']
AUGMENTATION_RANDOM_REMOVE_POINTS           = [0.60, 0.70]
AUGMENTATION_RANDOM_NOISE_TYPE              = 'none'
AUGMENTATION_RANDOM_NOISE_SPREAD            = -1

# Loss function settings
W_COLOR                                     = 0.15
W_STRUCTURE                                 = 0.95
W_SPARSE_DEPTH                              = 0.60
W_SMOOTHNESS                                = 0.04
W_WEIGHT_DECAY_DEPTH                        = 0.00
W_WEIGHT_DECAY_POSE                         = 0.00

# Evaluation settings
MIN_EVALUATE_DEPTH                          = 0.00
MAX_EVALUATE_DEPTH                          = 100.0

# Checkpoint settings
CHECKPOINT_PATH                             = 'trained_kbnet'
N_CHECKPOINT                                = 5000
N_SUMMARY                                   = 5000
N_SUMMARY_DISPLAY                           = 4
VALIDATION_START_STEP                       = 200000
RESTORE_PATH                                = None

# Hardware settings
CUDA                                        = 'cuda'
CPU                                         = 'cpu'
GPU                                         = 'gpu'
DEVICE                                      = 'cuda'
DEVICE_AVAILABLE                            = [CPU, CUDA, GPU]
N_THREAD                                    = 8
