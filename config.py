from easydict import EasyDict as edict

cfg = edict()

# Train
cfg.train = edict()
cfg.train.dataset_dir = "data/l"
cfg.train.dataset_resize = 24
cfg.train.batch_size = 24
cfg.train.epochs = 500
cfg.train.lr = 0.1
cfg.train.lr_reduce_cooldown = 2
cfg.train.lr_reduce_factor = 0.5
cfg.train.lr_reduce_patience = 2
cfg.train.momentum = 0.25
cfg.train.model_path = 'pths/l_8.pth'
cfg.train.save_every_nth_epoch = 20

# Test
cfg.test = edict()
cfg.test.dataset_dir = "data/l_test"
cfg.test.dataset_resize = 24
cfg.test.batch_size = 62
cfg.test.model_path = 'pths/l_7_80e.pth'

# Inference
cfg.inference = edict()
cfg.inference.image_path = 'images/salon.png'
cfg.inference.model_path = 'pths/l_7_80e.pth'
cfg.inference.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
cfg.inference.dataset_resize = 24
cfg.inference.show_letters = False
