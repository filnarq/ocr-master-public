import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from train import Net
from load_dataset import imshow

def crop(img, x, y, w, h, padding):
    return img[max(0, y-padding):min(img.shape[0], y+h+padding), max(0, x-padding):min(img.shape[1], x+w+padding)]

def processImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return thresh
    # morph = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,5)), iterations=2)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,5)), iterations=2)
    # morph = cv2.erode(thresh, kernel2, iterations=1)
    # morph = cv2.dilate(morph, kernel1, iterations=1)
    # return morph

def getSortedContours(img):
    ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # if byArea: 
    #     sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[2]*cv2.boundingRect(ctr)[3], reverse=True)
    # else:
    #     sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # avgContourHeight = 0
    res = []
    for i in range(len(ctrs)):
        # print(i, 'ctr', ctrs[i], 'hier', hier)
        # avgContourHeight += (cv2.boundingRect(ctr))[3]
        if True:
        # if hier[0][i][3] != -1:
            res.append(ctrs[i])
    # avgContourHeight = int(avgContourHeight/len(sorted_ctrs))
    return res

def ocr(img, net, resize, padding, classes=None):
    data = []
    processedImage = processImage(img)
    sortedContours = getSortedContours(processedImage)
    for ctr in sortedContours:
        x, y, w, h = cv2.boundingRect(ctr)
        prediction, probability = inference_torch(net=net, image=np.array(cv2.resize(crop(img,x,y,w,h,padding=padding), (resize, resize))))
        if classes:
            cv2.imshow('%s %f%%'%(classes[prediction],probability), cv2.resize(crop(img,x,y,w,h,padding=padding), (resize, resize)))
            cv2.waitKey(1000000)
            cv2.destroyWindow('%s %f%%'%(classes[prediction],probability))
        data.append((prediction, probability, x, y, w, h))
    return data

def inference_torch(image, net):
    image = torch.Tensor(np.expand_dims(np.transpose(image, (2,0,1)),axis=0))   
    outputs = net(image)
    probability, predicted = torch.max(outputs, 1)
    return predicted.item(), probability.item()
