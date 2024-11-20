import cv2
import random
# from aim import Run
from PIL import Image

import CRAFT.detect
from load_dataset import *
from inference import *

from train import *


### Inference
resize = 24
batchSize = 48
epochs = 20
lr = 0.05
momentum = 0.75
modelPath = 'pths/skewed_r_%d_b_%d_e_%d_lr_%s_m_%s%s.pth'%(
    resize,
    batchSize,
    epochs,
    str(lr).replace('.',''),
    str(momentum).replace('.',''),
    ''
)
# modelPath = 'pths/skewed_r_24_b_62_e_50_lr_005_m_06_14e.pth'


datasetDir = "data/skewed"
classes = get_classes(datasetDir)
trainloader = load_dataset_torch(dataset_dir=datasetDir, resize=resize, batch_size=batchSize, show=False)
train_torch(modelPath, trainloader=trainloader, epochs=epochs, lr=lr, momentum=momentum, epochsPerSave=5, elsPerStat=450)
# testloader = load_dataset_torch(dataset_dir=datasetDir+'_test', resize=resize, batch_size=batchSize, show=False)
# test_torch(modelPath, testloader=testloader, classes=classes, num=batchSize)

# # imagePath = 'data/random/0QA7VoDbm3_590.jpg'
# # imagePath = 'images/kaggle.png'
# imagePath = 'test.jpg'
# img = cv2.imread(imagePath)
# imgOut = img.copy()
# net = Net()
# net.load_state_dict(torch.load(modelPath, weights_only=True))
# _, boxes = CRAFT.detect.main(img_path=imagePath, model_path='CRAFT/pths/ft/model_iter_31600.pth', out='res.bmp')
# print(len(boxes))
# for box in boxes:
#     x,y,w,h = int(box[0]), int(box[1]), int(box[4])-int(box[0]), int(box[5])-int(box[1])
#     padding = resize/5
#     perspective = np.reshape(np.array(box, dtype=np.float32), (4,2))
#     perspectiveDest = np.array([[padding,padding],[resize-padding,padding],[resize-padding,resize-padding],[padding,resize-padding]], dtype=np.float32)
#     # print(perspective, perspectiveDest)
#     # cv2.imshow('crop', crop(img, x, y, w, h, padding=padding))

#     prediction, probability = inference_torch(cv2.warpPerspective(img, cv2.getPerspectiveTransform(perspective, perspectiveDest), (resize,resize)), net=net)
#     cv2.imshow('%s %f%%'%(classes[prediction],probability), cv2.warpPerspective(img, cv2.getPerspectiveTransform(perspective, perspectiveDest), (resize,resize)))
#     cv2.waitKey(1000000)
#     cv2.destroyWindow('%s %f%%'%(classes[prediction],probability))

#     cv2.putText(imgOut, text=classes[prediction], org=(x, y+h), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,255))

#     # cv2.waitKey(1000000)
#     # cv2.destroyWindow('crop')

# cv2.imshow('ocr',imgOut)
# cv2.waitKey(1000000)
