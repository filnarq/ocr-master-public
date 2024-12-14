from easydict import EasyDict as edict

cfg = edict()

cfg.train = edict()
cfg.train.mean = [0.5, 0.5, 0.5]
cfg.train.std = [0.25, 0.25, 0.25]

cfg.test = edict()
cfg.test.model_pth = './CRAFT/pths/pretrain/model_iter_50000.pth'
cfg.test.out_img = './output/boxes.png'
cfg.test.region_thresh = 0.29
cfg.test.affinity_thresh = 0.37
cfg.test.remove_thresh = 6 * 1e-4
cfg.test.long_side = 960
