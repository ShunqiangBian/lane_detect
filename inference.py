import os
import cv2
import torch
import numpy as np
from model.deeplab import DeepLab
from model.unet import ResNetUNet
from utils.image_process import LaneDataset, ToTensor
from utils.process_labels import decode_color_labels
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

predict_net = 'deeplabv3p'
# 设置一个字典，通过 key 值来选取使用的 model
nets = {'deeplabv3p': DeepLab, 'unet': ResNetUNet}


def load_model(model_path, args):
    # net 做了一个通用化的处理，取出来
    net = nets[predict_net]()
    # eval 不会进行反向传播
    net.eval()
    model_param = torch.load(model_path)['state_dict']
    model_param = {k.replace('module.', ''): v for k, v in model_param.items()}
    net.load_state_dict(model_param)
    return net


def get_color_mask(pred):
    pred = torch.softmax(pred, dim=1)
    # 将 channel 取得的最大的响应作为标签
    pred = torch.argmax(pred, dim=1)
    # squeeze 将某些维度上的1 去掉
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu().numpy()
    # 转换成 color 的 label
    pred = decode_color_labels(pred)
    # 将通道数返回来
    pred = np.transpose(pred, (1, 2, 0))
    return pred


def inference(args):
    kwargs = {'num_workers': args.num_works, 'pin_memory': True} if torch.cuda.is_available() else {}
    test_dataset = LaneDataset("test.csv", transform=transforms.Compose([ToTensor()]))
    test_data_batch = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, **kwargs)
    model_dir = 'result'
    model_path = os.path.join(model_dir, 'finalNet_unet.pth.tar')
    net = load_model(model_path)
    i = 0
    dataprocess = tqdm(test_data_batch)
    for batch_item in range(dataprocess):
        image, gray_mask = batch_item['image'], batch_item['mask']
        predict = net(image)
        i = i + 1
    # 对预测的结果进行处理，进行了颜色的转换
    color_mask = get_color_mask(predict)
    cv2.imwrite(os.path.join("image", 'color_mask_unet' + str(i) + '.jpg'), color_mask)
    cv2.imwrite(os.path.join("image", 'gray_mask' + str(i) + '.jpg'), gray_mask)
