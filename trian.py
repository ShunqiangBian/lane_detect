# -*- coding: utf-8 -*-
# Author : Shun Qiang
# Date   : 01/03/2020

import argparse
from tqdm import tqdm
import torch
import os
import shutil
from utils.metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug
from utils.image_process import ScaleAug, CutOut, ToTensor
from utils.loss import MySoftmaxCrossEntropyLoss
from utils.metric import Evaluator
from model.deeplab import DeepLab
from utils.logconfig import getLogger
from model.unet import ResNetUNet
from inference import inference

logger = getLogger()


def start_seg():
    """start semantic segment
    """
    logger.info("======start semantic segment=======")
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--save-path', action='store_true', default='result', help='skip validation during training')
    parser.add_argument('--cuda', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--base-lr', action='store_true', default=0.01, help='skip validation during training')
    parser.add_argument('--weight-decay', action='store_true', default=0.0001, help='skip validation during training')
    parser.add_argument('--num-works', action='store_true', default=0, help='skip validation during training')
    parser.add_argument('--epochs', action='store_true', default=4, help='skip validation during training')
    parser.add_argument('--number-class', action='store_true', default=8, help='skip validation during training')
    parser.add_argument('--net', action='store_true', default='unet', help='skip validation during training')
    parser.add_argument('--backbone', action='store_true', default='drn', help='skip validation during training')
    parser.add_argument('--sync-bn', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--freeze-bn', action='store_true', default=False, help='skip validation during training')
    # parser.add_argument('--cuda', action='store_truestore_true', default=False,help='skip validation during training')
    args = parser.parse_args()
    logger.info("Training config : {}".format(args))
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info("GPU Device:s%", args.gpu_ids)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    train(args)
    inference(args)


def train(args):
    predict_net = args.net
    nets = {'deeplabv3p': DeepLab, 'unet': ResNetUNet}
    trainF = open(os.path.join(args.save_path, "train.csv"), 'w')
    valF = open(os.path.join(args.save_path, "test.csv"), 'w')
    kwargs = {'num_workers': args.num_works, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_dataset = LaneDataset("train.csv", transform=transforms.Compose(
        [ImageAug(), DeformAug(), ScaleAug(), CutOut(32, 0.5), ToTensor()]))
    train_data_batch = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True, **kwargs)
    val_dataset = LaneDataset("val.csv", transform=transforms.Compose([ToTensor()]))
    val_data_batch = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True, **kwargs)
    net = nets[predict_net](args)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    # Training and test
    for epoch in range(args.epochs):
        # 在train_epoch中
        train_epoch(net, epoch, train_data_batch, optimizer, trainF, args)
        val_epoch(net, epoch, val_data_batch, valF, args)
        if epoch % 2 == 0:
            torch.save({'state_dict': net.state_dict()},
                       os.path.join(os.getcwd(), args.save_path, "laneNet{}.pth.tar".format(epoch)))
    trainF.close()
    valF.close()
    torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), "result", "finalNet_unet.pth.tar"))


def train_epoch(net, epoch, dataLoader, optimizer, trainF, args):
    logger.info("======start training epoch step=======")
    net.train()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        optimizer.zero_grad()
        out = net(image)
        logger.info("train predict shape: {}".format(out.shape))
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=args.number_class)(out, mask)
        total_mask_loss += mask_loss.item()
        mask_loss.backward()
        # optimizer 进行更新
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    # 记录数据迭代了多少次
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def val_epoch(net, epoch, dataLoader, valF, args):
    logger.info("======start val epoch step=======")
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)
    result = {"TP": {i: 0 for i in range(8)}, "TA": {i: 0 for i in range(8)}}
    evaluator = Evaluator(args.number_class)
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        out = net(image)
        mask_loss = MySoftmaxCrossEntropyLoss(nbclasses=args.number_class)(out, mask)
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    valF.write("Epoch:{}, val/mIoU is {:.4f} \n".format(epoch, mIoU))
    valF.write("Epoch:{}, val/Acc is {:.4f} \n".format(epoch, Acc))
    valF.write("Epoch:{}, val/Acc_class is {:.4f} \n".format(epoch, Acc_class))
    valF.write("Epoch:{}, val/FWIoU is {:.4f} \n".format(epoch, FWIoU))
    for i in range(8):
        result_string = "{}: {:.4f} \n".format(i, result["TP"][i] / result["TA"][i])
        logger.info("val class result {}".format(result_string))
        valF.write(result_string)
    valF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    logger.info("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    valF.flush()


if __name__ == "__main__":
    logger.info("Lane Segmentation Start...")
    start_seg()
