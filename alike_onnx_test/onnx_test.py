import argparse
import csv
import glob
import logging
import math
import os
import random
import time
from copy import deepcopy
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from onnxmodel import ONNXModel
from soft_detect import DKD

class ImageLoader(object):
    def __init__(self, filepath: str):
        self.images = glob.glob(os.path.join(filepath, '*.png')) + \
                      glob.glob(os.path.join(filepath, '*.jpg')) + \
                      glob.glob(os.path.join(filepath, '*.ppm'))
        self.images.sort()
        self.N = len(self.images)
        logging.info(f'Loading {self.N} images')
        self.mode = 'images'

    def __getitem__(self, item):
        filename = self.images[item]
        img = cv2.imread(filename)
        return img,filename

    def __len__(self):
        return self.N


def mnn_mather(desc1, desc2):
    sim = desc1 @ desc2.transpose()
    sim[sim < 0.75] = 0
    nn12 = np.argmax(sim, axis=1) #函数：numpy.argmax(array, axis) array：代表输入数组；axis：代表对array取行（axis=0）或列（axis=1）的最大值
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    return matches.transpose()

def plot_keypoints(image, kpts, scores, radius=2, color=(0, 0, 255)):
    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    i = 0
    greeni = 0
    for kpt in kpts:
        x0, y0 = kpt
        scores_1 = scores[i]
        i += 1
        if scores_1 >= 0.35:
            greeni += 1
        scores_1 = min(1,scores_1 + 0.2)
        color = list(color)
        color[0] = 0
        color[1] = scores_1 * 255
        color[2] = (-scores_1 + 1) * 255
        color = tuple(color)
        cv2.circle(out, (x0, y0), radius, color, -1, lineType=cv2.LINE_4)

    return out, greeni


def plot_matches(img_name,
                 image0,
                 image1,
                 kpts0,
                 kpts1,
                 scores1,
                 scores2,
                 matches,
                 match_point_write_dir,
                 radius=1,
                 color=(0, 255, 0)):

    out0, green1 = plot_keypoints(image0, kpts0, scores1, radius, color)
    out1, green2 = plot_keypoints(image1, kpts1, scores2, radius, color)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[:H1, W0:, :] = out1

    mkpts0, mkpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)

    points_out = out.copy()
    i = 0
    test_image_name = ['1614044935217943_L.png','1614044895123555_L.png','1614045104206922_L.png']

    is_write = False
    if match_point_write_dir and img_name in test_image_name:
        is_write = True
        # 测试的图片
        match_point_root_dir = os.path.join(match_point_write_dir, f"top{args.top_k}_nms{args.radius}")
        match_point_path = os.path.join(match_point_root_dir,img_name.split('.')[0])
        if not os.path.exists(match_point_path):
            os.makedirs(match_point_path)
    count_match_mnn = 0#匹配计数
    for kpt0, kpt1 in zip(mkpts0, mkpts1):
        (x0, y0), (x1, y1) = kpt0, kpt1
        mcolor = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        count_match_mnn += 1#匹配计数
        cv2.line(out, (x0, y0), (x1 + W0, y1),
                 color=mcolor,
                 thickness=1,
                 lineType=cv2.LINE_AA)
        if is_write:
            point_match = points_out.copy()
            cv2.line(point_match, (x0, y0), (x1 + W0, y1),
                     color=mcolor,
                     thickness=2,
                     lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(match_point_path,f"{i}_{img_name}"),point_match)
            i += 1


    # cv2.putText(out, str(len(mkpts0)),
    #             (out.shape[1] - 150, out.shape[0] - 50),
    #             cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

    return out, points_out, count_match_mnn, green1, green2


def double_match(img_name,img1, img2,pts1, pts2, scores1, scores2, desc1, desc2,match_point_write_dir,radius=1, color=(0, 255, 0)):
    if pts1 is None or pts2 is None or desc1 is None or desc2 is None:
        print('[alike]==>pts1 is None or pts2 is None or desc1 is None or desc2 is None')
        return

    # rows, cols = img1.shape
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(desc1, desc2)

    out1, green1 = plot_keypoints(img1, pts1, scores1, radius, color)
    out2, green2 = plot_keypoints(img2, pts2, scores2, radius, color)

    H0, W0 = img1.shape[0], img1.shape[1]
    H1, W1 = img2.shape[0], img2.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out1
    out[:H1, W0:, :] = out2
    stereo_img = out
    count_match = 0

    points_out = out.copy()
    i = 0
    test_image_name = ['1614044935217943_L.png', '1614044895123555_L.png', '1614045104206922_L.png','1614044731739682_L.png']
    is_write = False
    if match_point_write_dir and img_name in test_image_name:
        is_write = True
        # 测试的图片
        match_point_root_dir = os.path.join(match_point_write_dir, f"top{args.top_k}_nms{args.radius}")
        match_point_path = os.path.join(match_point_root_dir, img_name.split('.')[0])
        if not os.path.exists(match_point_path):
            os.makedirs(match_point_path)

    for match in matches:
        pt1 = pts1[match.queryIdx, :]
        pt2 = pts2[match.trainIdx, :]
        ptL = np.array([int(round(pt1[0])),int(round(pt1[1]))])
        ptR = np.array([int(round(pt2[0]) + W1), int(round(pt2[1]))])

        if abs(ptL[1] - ptR[1]) > 25:
            continue

        count_match += 1
        cv2.circle(stereo_img, tuple(ptL), 1, (0, 255, 0), -1, lineType=16)
        cv2.circle(stereo_img, tuple(ptR), 1, (0, 255, 0), -1, lineType=16)
        cv2.line(stereo_img, tuple(ptL), tuple(ptR),
                 (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                 thickness=1, lineType=cv2.LINE_AA)
        if is_write:
            point_match = points_out.copy()
            cv2.line(point_match, tuple(ptL), tuple(ptR),
                     color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                     thickness=2,
                     lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(match_point_path,f"{i}_{img_name}"),point_match)
            i += 1

    # cv2.putText(stereo_img, str(count_match), (stereo_img.shape[1] - 150, stereo_img.shape[0] - 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)


    return stereo_img, points_out, count_match, green1, green2
def post_deal(W,H,scores_map, descriptor_map,radius=2,top_k=2000, scores_th=0.2,n_limit=5000,sort=False):
    descriptor_map = torch.nn.functional.normalize(descriptor_map, p=2, dim=1)
    keypoints, descriptors, scores, _ = DKD(radius=radius, top_k=top_k,scores_th=scores_th, n_limit=n_limit).forward(scores_map, descriptor_map)
    keypoints, descriptors, scores = keypoints[0], descriptors[0], scores[0]
    keypoints = (keypoints + 1) / 2 * keypoints.new_tensor([[W - 1, H - 1]])
    if sort:
        indices = torch.argsort(scores, descending=True)
        keypoints = keypoints[indices]
        descriptors = descriptors[indices]
        scores = scores[indices]

    return {'keypoints': keypoints.cpu().numpy(),
            'descriptors': descriptors.cpu().numpy(),
            'scores': scores.cpu().numpy(),
            'scores_map': scores_map.cpu().numpy(),}

def pre_deal_np(img,flg=False):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img_rgb.transpose(2, 0, 1)[None] / 255.0
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = img_rgb.transpose(2, 0, 1)[None]
    b, c, h, w = image.shape
    h_ = math.ceil(h / 8) * 8 if h % 8 != 0 else h
    w_ = math.ceil(w / 8) * 8 if w % 8 != 0 else w
    if h_ > h:
        h_padding = np.zeros((b, c, h_ - h, w))
        image = np.concatenate([image, h_padding], dtype=np.float32,axis=2)
    if w_ > w:
        w_padding = np.zeros((b, c, h_, w_ - w))
        image = np.concatenate([image, w_padding], dtype=np.float32,axis=3)
    if h_ != h or w_ != w:
        flg = True
    image = image.astype(np.float32)
    return image,flg,h,w

#match_info()用来在图片中显示关键点匹配的信息以及输出信息到控制台
#   vis_img     - 特征匹配后的图像
#   points_out  - 特征点图像
#   count_match - 特征点匹配数目
#   kpts        - 左图特征点数目
#   kpts_ref    - 右图特征点数目
#   match_model - 单双向匹配模式
def match_info(vis_img, points_out, count_match, kpts, kpts_ref, match_model, green1, green2):
    count_match_img =match_model + "match:" + str(count_match)
    cv2.putText(vis_img, str(count_match_img), (20, vis_img.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # 显示匹配数量

    cv2.putText(vis_img, "Press 'q' or 'ESC' to stop.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)  # 显示按键退出

    cv2.putText(vis_img, str(len(kpts)), (int(vis_img.shape[1] / 2) - 100, vis_img.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)  # 显示左图的特征点数目

    cv2.putText(vis_img, str(len(kpts_ref)), (vis_img.shape[1] - 100, vis_img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2,
                cv2.LINE_AA)  # 显示右图的特征点数目

    cv2.putText(points_out, str(len(kpts)),
                (int(vis_img.shape[1] / 2) - 100, vis_img.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # 显示特征点数目

    cv2.putText(points_out, str(len(kpts_ref)),
                (vis_img.shape[1] - 100, vis_img.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # 显示特征点数目

    green_num1 = "green num :" + str(green1)
    # green_num2 = "green num :" + str(green2)
    cv2.putText(points_out, green_num1,
                (20, vis_img.shape[0] - 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # 显示绿色点数目

    # cv2.putText(points_out, green_num2,
    #             (int(vis_img.shape[1] / 2) + 20, vis_img.shape[0] - 50),
    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # 显示绿色点数目


    print('左图的特征点数目:' + str(len(kpts)) + ' 右图的特征点数目:' + str(len(kpts_ref)) + ' 匹配数量:' + str(
        count_match))



def run(args):
    logging.basicConfig(level=logging.INFO)
    image_loader = ImageLoader(args.input)
    version = args.version
    if not version:
        raise Exception("version is not none!")

    model = ONNXModel(args.model_path)
    logging.info("Press 'space' to start. \nPress 'q' or 'ESC' to stop!")
    image_loader2 = ImageLoader(args.input2)

    sum_net_t = []
    sum_net_matches_t = []
    sum_total_t = []  # 初始化时间列表
    # for i in range(2500, len(image_loader)):
    for i in range(0, len(image_loader)):
    # for i in range(4600, 4601):
        start = time.time()
        img, img_name = image_loader[i]
        img2, img2_name = image_loader2[i]
        if img is None or img2 is None:
            break
        img_rgb, flg1, H1, W1 = pre_deal_np(img)
        img_rgb2, flg2, H2, W2 = pre_deal_np(img2)
        start1 = time.time()
        scores_map1, descriptor_map1 = model.forward(img_rgb)
        scores_map2, descriptor_map2 = model.forward(img_rgb2)
        end1 = time.time()
        descriptor_map1 = torch.from_numpy(descriptor_map1)
        scores_map1 = torch.from_numpy(scores_map1)
        output1 = post_deal(W1, H1, scores_map1, descriptor_map1,args.radius,args.top_k,args.scores_th,args.n_limit)#后处理
        descriptor_map2 = torch.from_numpy(descriptor_map2)
        scores_map2 = torch.from_numpy(scores_map2)
        output2 = post_deal(W2, H2, scores_map2, descriptor_map2,args.radius,args.top_k,args.scores_th,args.n_limit)
        kpts = output1['keypoints']
        desc = output1['descriptors']
        kpts_ref = output2['keypoints']
        desc_ref = output2['descriptors']

        scores1 = output1['scores']
        scores2 = output2['scores']


        # try:
        #     matches = mnn_mather(desc, desc_ref)
        # except:
        #     continue
        end2 = time.time()
        img_name = os.path.basename(img_name)

        #判断是使用点积匹配还是opencv双向匹配
        if args.match_model == 'mnn':
            try:
                matches = mnn_mather(desc, desc_ref)
                vis_img, points_out, count_match, green1, green2 = plot_matches(img_name, img, img2, kpts, kpts_ref, scores1, scores2, matches,
                                                   args.match_point_write_dir)
            except:
                continue

        elif args.match_model == 'double':
            vis_img, points_out, count_match, green1, green2 = double_match(img_name,img, img2, kpts, kpts_ref, scores1, scores2, desc, desc_ref,args.match_point_write_dir)

        # vis_img, points_out = plot_matches(img_name, img, img2, kpts, kpts_ref, matches, args.match_point_write_dir)
        #vis_img, points_out, count_match = double_match(img_name,img, img2, kpts, kpts_ref, scores1, scores2, desc, desc_ref,args.match_point_write_dir)
        cv2.namedWindow(args.model)
        match_info(vis_img, points_out, count_match, kpts, kpts_ref,args.match_model, green1, green2)#


        cv2.imshow('points', points_out)
        cv2.imshow(args.model, vis_img)
        end = time.time()
        net_t = end1 - start1
        net_matches_t = end2 - start1
        total_t = end - start
        print('Use match_model:', args.match_model, 'Processed image %d (net: %.3f FPS,net+matches: %.3f FPS, total: %.3f FPS).' % (i, net_t, net_matches_t, total_t))


        if len(sum_net_t) < 102:  # 剔除最后一张和第一张  计算100张图片的平均帧率
            sum_net_t.append(net_t)
            sum_net_matches_t.append(net_matches_t)
            sum_total_t.append(total_t)
        if args.write_dir:  # 匹配的图像文件保存
            save_img_path = os.path.join(args.write_dir, f"top{args.top_k}_nms{args.radius}")
            img_name = os.path.basename(img_name)
            os.makedirs(save_img_path, exist_ok=True)
            out_file1 = os.path.join(save_img_path, "t" + img_name)
            cv2.imwrite(out_file1, points_out)
            out_file2 = os.path.join(save_img_path, "d" + img_name)
            cv2.imwrite(out_file2, vis_img)
            log_file = os.path.join(save_img_path, "log.csv")
            f = open(log_file, 'a')  # 记录图像的特征点和匹配数量
            writer = csv.writer(f)
            # writer.writerow([img_name, len(kpts), len(matches)])

        c = cv2.waitKey(1)
        if c == 32:
            while True:
                key = cv2.waitKey(1)
                if key == 32:
                    break
        if c == ord('q') or c == 27:
            break

        if i == 2100 or i == 2600 or i == 4700:
            break
    # 计算平均帧率
    avg_net_FPS = np.mean(sum_net_t[1:len(sum_net_t) - 1])
    avg_net_matches_FPS = np.mean(sum_net_matches_t[1:len(sum_net_matches_t) - 1])
    avg_total_FPS = np.mean(sum_total_t[1:len(sum_total_t) - 1])
    if args.write_dir:  # 记录图像的平均帧率
        writer.writerow([f'avg_net_FPS:{avg_net_FPS:.3f},avg_net+matches_FPS:{avg_net_matches_FPS:.3f},avg_total_FPS:{avg_total_FPS:.3f}'])
    print(f'avg_FPS：\n avg_net_FPS:{avg_net_FPS:.3f},avg_net+matches_FPS:{avg_net_matches_FPS:.3f},avg_total_FPS:{avg_total_FPS:.3f}')
    logging.info('Finished!')
    logging.info('Press any key to exit!')
    cv2.putText(vis_img, "Finished! Press any key to exit.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                cv2.LINE_AA)
    cv2.imshow(args.model, vis_img)
    cv2.waitKey()





def GetArgs():
    parser = argparse.ArgumentParser(description='ALIKE image pair Demo.')
    parser.add_argument('--input', type=str, default='',
                        help='Image directory.')
    parser.add_argument('--input2', type=str, default='',
                        help='Image directory.')
    parser.add_argument('--model', choices=['alike-t', 'alike-s', 'alike-n', 'alike-l'], default="alike-n",
                        help="The model configuration")
    parser.add_argument('--model_path', default="default", help="The model onnx file path, The default is open source model")
    parser.add_argument('--top_k', type=int, default=-1,
                        help='Detect top K keypoints. -1 for threshold based mode, >0 for top K mode. (default: -1)')
    parser.add_argument('--scores_th', type=float, default=0.2,
                        help='Detector score thr eshold (default: 0.2).')
    parser.add_argument('--n_limit', type=int, default=5000,
                        help='Maximum number of keypoints to be detected (default: 5000).')
    parser.add_argument('--radius', type=int, default=2,
                        help='The radius of non-maximum suppression (default: 2).')
    parser.add_argument('--write_dir', type=str, default='', help='Image save directory.')
    parser.add_argument('--match_point_write_dir', type=str, default='', help='Image match point save directory.')
    parser.add_argument('--version', type=str, default='', help='version')
    parser.add_argument('--match_model', choices=['double', 'mnn'], help='Choose whether the matching mode is double_match or mnn_mather ')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = GetArgs()
    run(args)

