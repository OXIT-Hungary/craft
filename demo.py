"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

import shutil

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

""" parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='submodules/craft/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.2, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=5.0, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='datasets/craft_input', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='path/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args() """

def get_character_position(img, brightness_thresh, output_img_=None):

    bb_size_threshold = 10  ### Minimum size of the bounding box ###
    color_threshold = 0.3   ### Brightness value, that determines the lower threshold of the color palette; lower value - red , upper value - green ###

    ret, thresh = cv2.threshold(img, brightness_thresh, 1, 0)
    thresh = cv2.convertScaleAbs(thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:

        #find the biggest area of the contour
        c = max(contours, key = cv2.contourArea)

        if output_img_ is not None:

            # the contours are drawn here
            #cv.drawContours(output_img_, [c], -1, 255, 3)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)
            detection_color = (0, 255*(maxVal-color_threshold), 255-255*(maxVal-color_threshold))
            cv2.circle(output_img_, maxLoc, 1, detection_color, 10)
            cv2.circle(output_img_, maxLoc, 20, detection_color, 2)

        x,y,w,h = cv2.boundingRect(c)
        return [x+w/2, y+h/2]#, thresh

def test_net(cfg, net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, cfg.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=cfg.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    if cfg.max_intensity == True:
        result_img = score_text.copy()
        result_pos = get_character_position(score_text, 0.1, result_img)
        result_img = cv2.normalize(result_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        #cv2.imwrite('test_heatmap.jpg', result_test)
        return result_pos, result_img
    else:
        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if cfg.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return boxes, polys, ret_score_text

def get_position_on_original_img(cfg, pos, heatmap, img):

    heatmap_size = heatmap.shape
    img_size = img.shape

    target_w, target_h = int(img_size[0] * cfg.mag_ratio), int(img_size[1] * cfg.mag_ratio)
    width_corr = 0
    height_corr = 0
    if target_w % 32 != 0:
        width_corr = 32 - target_w % 32
    if target_h % 32 != 0:
        height_corr = 32 - target_h % 32
    
    return (int((pos[0]*2 - width_corr) / 10), int((pos[1]*2 - height_corr) / 10))

def main(cfg):

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(cfg.test_folder)

    result_folder = cfg.result_folder

    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
        os.makedirs(result_folder, exist_ok=True)
    else:
        os.makedirs(result_folder, exist_ok=True)

    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + cfg.trained_model + ')')
    if cfg.cuda:
        net.load_state_dict(copyStateDict(torch.load(cfg.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(cfg.trained_model, map_location='cpu')))

    if cfg.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if cfg.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + cfg.refiner_model + ')')
        if cfg.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(cfg.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(cfg.refiner_model, map_location='cpu')))

        refine_net.eval()
        cfg.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        if cfg.max_intensity == True:
            _position, _image = test_net(cfg, net, image, cfg.text_threshold, cfg.link_threshold, cfg.low_text, cfg.cuda, cfg.poly, refine_net)

            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = result_folder + "/res_" + filename + '_mask.jpg'
            #cv2.imwrite(mask_file, _image)

            original_file = result_folder + "/res_" + filename + '.jpg'
            _img = np.array(image[:,:,::-1])
            if _position is not None:
                circle_pos = get_position_on_original_img(cfg, _position, _image, _img)
                cv2.circle(_img, circle_pos, 1, (0,0,255), 2)
                #cv2.circle(_img, circle_pos, cfg.bbox_size, (255,0,0), 2)

                bbox_size = cfg.bbox_size/2
                cv2.rectangle(_img,(int(circle_pos[0]-bbox_size),int(circle_pos[1]+bbox_size)),(int(circle_pos[0]+bbox_size),int(circle_pos[1]-bbox_size)),(255,0,0), 2)

                res_txt = result_folder + "/res_" + filename + '.txt'
                with open(res_txt, 'w') as f:

                    p1 = (int(circle_pos[0]-bbox_size),int(circle_pos[1]-bbox_size))
                    p2 = (int(circle_pos[0]+bbox_size),int(circle_pos[1]-bbox_size))
                    p3 = (int(circle_pos[0]+bbox_size),int(circle_pos[1]+bbox_size))
                    p4 = (int(circle_pos[0]-bbox_size),int(circle_pos[1]+bbox_size))
                    f.write(str(p1[0]) + ',' + str(p1[1]) + ',' + str(p2[0]) + ',' + str(p2[1]) + ',' + str(p3[0]) + ',' + str(p3[1]) + ',' + str(p4[0]) + ',' + str(p4[1]))

                cv2.imwrite(original_file, _img)

        else:
            bboxes, polys, score_text = test_net(cfg, net, image, cfg.text_threshold, cfg.link_threshold, cfg.low_text, cfg.cuda, cfg.poly, refine_net)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = result_folder + "/res_" + filename + '_mask.jpg'
            #cv2.imwrite(mask_file, score_text)

            file_utils.saveResult(cfg, image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))

if __name__ == '__main__':
    main()