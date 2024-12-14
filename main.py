import argparse
import numpy as np
import cv2

import CRAFT.detect
from load_dataset import *
from train import *
from config import cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(cfg=cfg.train):
    metadata = open(cfg.model_path+'.meta', 'w')
    metadata.write('%s\n%s'%(str(Net()),str(cfg)))
    metadata.close()
    print(str(cfg))
    trainloader = load_dataset_torch(dataset_dir=cfg.dataset_dir, resize=cfg.dataset_resize, batch_size=cfg.batch_size, show=False)
    testloader = load_dataset_torch(dataset_dir=cfg.dataset_dir, resize=cfg.dataset_resize, batch_size=cfg.batch_size, show=False)
    train_torch(device=device, modelPath=cfg.model_path, trainloader=trainloader, valloader=testloader,
                epochs=cfg.epochs, lr=cfg.lr, lr_cd=cfg.lr_reduce_cooldown, lr_f=cfg.lr_reduce_factor, lr_p=cfg.lr_reduce_patience,
                momentum=cfg.momentum, epochsPerSave=cfg.save_every_nth_epoch)

def test(cfg=cfg.test):
    classes = get_classes(cfg.dataset_dir)
    testloader = load_dataset_torch(dataset_dir=cfg.dataset_dir, resize=cfg.dataset_resize, batch_size=cfg.batch_size, show=False)
    test_torch(device, cfg.model_path, testloader=testloader, classes=classes, batchSize=cfg.batch_size)

def inference(cfg=cfg.inference):
    # Set vars
    size = cfg.dataset_resize
    padding = size/5

    # Load image and net
    img = cv2.imread(cfg.image_path)
    imgOut = img.copy()
    net = Net()
    net.load_state_dict(torch.load(cfg.model_path, weights_only=True, map_location=device))
    net.eval()

    # Detect textboxes
    _, boxes, wordsNums = CRAFT.detect.main(img_path=cfg.image_path)
    words = [[] for i in range(int(wordsNums[0][0]))]
    for i, box in enumerate(boxes):
        # Warp textboxes
        x,y,w,h = int(box[0]), int(box[1]), int(box[4])-int(box[0]), int(box[5])-int(box[1])
        perspective = np.reshape(np.array(box, dtype=np.float32), (4,2))
        perspectiveDest = np.array([[padding,padding],[size-padding,padding],[size-padding,size-padding],[padding,size-padding]], dtype=np.float32)

        # Get prediction
        prediction, probability = inference_torch(image=cv2.warpPerspective(img, cv2.getPerspectiveTransform(perspective, perspectiveDest), (size,size)), net=net)
        cv2.putText(imgOut, text=cfg.classes[prediction], org=(x, y+h), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(200,200,55),  thickness=3)
        words[int(wordsNums[i][1])].append(cfg.classes[prediction])

        # Show prediction
        if cfg.show_letters:
            cv2.imshow('%s %f%%'%(cfg.classes[prediction],probability), cv2.warpPerspective(img, cv2.getPerspectiveTransform(perspective, perspectiveDest), (size,size)))
            cv2.waitKey(1000000)
            cv2.destroyWindow('%s %f%%'%(cfg.classes[prediction],probability))

    # Return results
    print('\n'.join([''.join(word) for word in words]))
    cv2.imwrite('output/res.png', imgOut)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ocr')
    subparsers = parser.add_subparsers(dest='subcommand')
    train_subcommand = subparsers.add_parser(name='train')
    test_subcommand = subparsers.add_parser(name='test')
    inference_subcommand = subparsers.add_parser(name='inference')
    args = parser.parse_args()
    match args.subcommand:
        case 'train':
            train()
        case 'test':
            test()
        case 'inference':
            inference()
        case _:
            print('Available subcommands: train, test, inference\nFill config.py with values')
