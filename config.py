from easydict import EasyDict as edict

cfg = edict()

# Train
cfg.train = edict()
cfg.train.dataset_dir = "data/cropped"
cfg.train.dataset_resize = 24
cfg.train.batch_size = 62
cfg.train.epochs = 500
cfg.train.lr = 0.05
cfg.train.lr_gamma = 0.7
cfg.train.lr_gamma_steps = 1
cfg.train.momentum = 0.7
cfg.train.model_path = 'pths/cropped_5.pth'
cfg.train.save_every_nth_epoch = 25
cfg.train.print_loss_every_n_batches = 50

# Test
cfg.test = edict()
cfg.test.dataset_dir = "data/cropped_test"
cfg.test.dataset_resize = 24
cfg.test.batch_size = 48
cfg.test.model_path = 'pths/cropped_5_125e.pth'

# Inference
cfg.inference = edict()
cfg.inference.image_path = 'test.jpg'
cfg.inference.model_path = 'pths/cropped_5_200e.pth'
cfg.inference.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
cfg.inference.dataset_resize = 24
cfg.inference.show_letters = False
