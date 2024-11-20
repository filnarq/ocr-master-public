from easydict import EasyDict as edict

cfg = edict()

# Train
cfg.train = edict()
cfg.train.dataset_dir = "data/skewed"
cfg.train.dataset_resize = 24
cfg.train.batch_size = 48
cfg.train.epochs = 20
cfg.train.lr = 0.05
cfg.train.momentum = 0.75
cfg.train.model_path = 'pths/skewed_r_24_b_62_e_20_lr_005_m_09_5e.pth'
cfg.save_every_nth_epoch = 5
cfg.print_loss_every_n_batches = 450

# Test
cfg.test = edict()
cfg.test.dataset_dir = "data/skewed"
cfg.test.dataset_resize = 24
cfg.test.batch_size = 48
cfg.test.model_path = 'pths/skewed_r_24_b_62_e_20_lr_005_m_09_5e.pth'

# Inference
cfg.inference = edict()
cfg.inference.image_path = 'test.jpg'
cfg.inference.model_path = 'pths/skewed_r_24_b_62_e_20_lr_005_m_09_5e.pth'
cfg.inference.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
cfg.inference.dataset_resize = 24
cfg.inference.show_letters = True

# modelPath = 'pths/skewed_r_%d_b_%d_e_%d_lr_%s_m_%s%s.pth'%(
#     resize,
#     batchSize,
#     epochs,
#     str(lr).replace('.',''),
#     str(momentum).replace('.',''),
#     ''
# )

# imagePath = 'data/random/0QA7VoDbm3_590.jpg'
# imagePath = 'images/kaggle.png'