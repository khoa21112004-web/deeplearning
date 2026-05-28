import os

config = {
    'max_epoch': 50,
    'log_train': 100,
    'lr': 2e-5,
    'starting_epoch': 0,

    # Kaggle-friendly
    'batch_size': 4,
    'image_size': 224,
    'target_slices': 24,
    'num_workers': 2,

    # Task
    'task': 'acl',

    # Optimizer
    'weight_decay': 1e-4,
    'patience': 7,
    'save_model': 1,
    'exp_name': 'baseline',

    # Gradient accumulation
    'use_gradient_accumulation': 1,
    'gradient_accumulation_steps': 8,

    # Loss / imbalance options
    'loss_name': 'bce',
    'use_weighted_sampler': 0,
    'use_pos_weight': 1,
    'focal_alpha': 0.75,
    'focal_gamma': 2.0,

    # Warm-start
    'abnormal_warmstart_path': os.environ.get(
        'ABNORMAL_WARMSTART_PTH',
        '/kaggle/working/weights/abnormal/efficientnetb0_best_model.pth',
    ),
    'warmstart_tasks': ['acl', 'meniscus'],
    'warmstart_from_abnormal': 1,
}