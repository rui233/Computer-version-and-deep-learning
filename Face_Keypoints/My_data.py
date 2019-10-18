import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytest

train_boarder = 112


class FaceLandmarksDataset(Dataset):
    def __init__(self, data_file, phase='train', transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.transform = transform
        if not os.path.exists(data_file):
            print(data_file+"does not exist!")
        self.file_info = pd.read_csv(data_file, index_col=0)
        # 增加一列为正样本，人脸标签为1
        self.file_info['class'] = 1
        self.negative_samples = self.get_negative_samples(2)
        self.file_info = pd.concat([self.file_info, self.negative_samples])
        # for index, data in self.negative_samples.iterrows():
        #     img = Image.open(data['path'])
        #     rect = data['rect']
        #     draw = ImageDraw.Draw(img)
        #     draw.rectangle(rect, outline='green')
        #     # draw.point(points_zip, (255, 0, 0))
        #     # img.save(r'H:\DataSet\慧科\人脸关键点检测\result\{:d}.jpg'.format(index))
        #     plt.imshow(img)
        #     plt.show()
        self.size = len(self.file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self.file_info.iloc[idx]
        img_name = data['path']
        rect = eval(data['rect'])
        points = eval(data['points'])
        class_ = np.array([data['class']])
        # image
        img = Image.open(img_name).convert('RGB')     # this is good

        img_crop = img.crop(rect)  # this is also good, but has some shift already
        if class_ == 1:
            landmarks = np.array(points).astype(np.float32)
            landmarks = landmarks - np.array(rect[0:2])
        else:
            landmarks = np.zeros((21, 2), dtype=np.float32)
        sample = {'image': img_crop, 'landmarks': landmarks, 'label': class_}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_negative_samples(self, negative_ratio, random_border=10):
        def get_iou(rect1, rect2):
            overlap_w = min(rect1[2], rect2[2])-max(rect1[0], rect2[0]) \
                if min(rect1[2], rect2[2])-max(rect1[0], rect2[0]) > 0 else 0
            overlap_h = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]) \
                if min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]) > 0 else 0
            f = lambda a, b: (a[0]-a[2])*(a[1]-a[3])+(b[0]-b[2])*(b[1]-b[3])
            overlap_area = f(rect1, rect2) - overlap_w*overlap_h
            return overlap_w*overlap_h/overlap_area
        negative_data_info = {'path': [], 'rect': []}
        for index, rows_data in self.file_info.iterrows():
            gen_rect_num = 0
            # 如果尝试50次还没有找到合适的negative rect则放弃
            try_times = 0
            try_times_threshold = 100
            img_path = rows_data['path']
            img = Image.open(img_path)
            width, height = img.size

            rect_in_same_img = self.file_info[self.file_info['path'] == img_path]
            rects = []
            rects_w = []
            rects_h = []
            for index, rect_data in rect_in_same_img.iterrows():
                rect = eval(rect_data['rect'])
                rect_w = rect[2] - rect[0]
                rect_h = rect[3] - rect[1]
                rects_w.append(rect_w)
                rects_h.append(rect_h)
                rects.append(rect)
            max_rects_w, min_rects_w = max(rects_w), min(rects_w)
            max_rects_h, min_rects_h = max(rects_h), min(rects_h)
            while gen_rect_num < negative_ratio and try_times < try_times_threshold:
                left = np.random.randint(0, width-max_rects_w-random_border)\
                    if width-max_rects_w-random_border > 1 else 0
                top = np.random.randint(0, height-max_rects_h-random_border)\
                    if height-max_rects_h-random_border > 1 else 0
                rect_w_rand = np.random.randint(min_rects_w-1, max_rects_w)
                rect_h_rand = np.random.randint(min_rects_h-1, max_rects_h)
                rect_randw = np.random.randint(-3, 3)
                rect_randh = np.random.randint(-3, 3)
                right = left + rect_w_rand + rect_randw - 1
                bottom = top + rect_h_rand + rect_randh - 1
                rect_rand = [left, top, right, bottom]
                try_times += 1
                flag_find = True
                for rect in rects:
                    iou = get_iou(rect_rand, rect)
                    if iou > 0.3:
                        flag_find = False
                        break
                if not flag_find:
                    continue
                gen_rect_num += 1
                negative_data_info['path'].append(rows_data['path'])
                negative_data_info['rect'].append(str(rect_rand))
        data = pd.DataFrame(negative_data_info)
        data['points'] = str([0, 0])
        data['class'] = 0
        return data


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
    """
    def __call__(self, sample):
        img, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        width, height = img.size
        img_resize = np.asarray(img.resize((train_boarder, train_boarder), Image.BILINEAR), dtype=np.float32)
        img = img_resize
        if label == 1:
            landmarks[:, 0] = landmarks[:, 0]*train_boarder/width
            landmarks[:, 1] = landmarks[:, 1] * train_boarder / width
        return {'image': img, 'landmarks': landmarks, 'label': label}


class RandomHorizontalFlip(object):
    """
        Horizontally flip image randomly with given probability
        Args:
            p (float): probability of the image being flipped.
                       Default value = 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, landmarks, label = sample['image'], sample['landmarks'], sample['label']

        if np.random.random() < self.p:
            img = img[:, ::-1].copy()
            if label == 1:
                if label:
                    landmarks[:, 0] = train_boarder - landmarks[:, 0]
        return {'image': img,
                'landmarks': landmarks,
                'label': label
                }


class RandomRotate(object):
    """
        Randomly rotate image within given limits
        Args:
            p (float): probability above which the image need to be flipped. Default value = 0.25
            rotate limits by default: [-20, 20]
    """
    def __init__(self, p=0.5, a=20):
        self.p = p
        self.angle = a

    def __call__(self, sample):
        img, landmarks, label = sample['image'], sample['landmarks'], sample['label']

        if np.random.random() > self.p:
            # angle
            limit = self.angle
            angle = np.random.randint(-limit, limit)
            # print(img.dtype)
            img = Image.fromarray(img.astype('uint8')).rotate(angle, resample=Image.BILINEAR, expand=False)
            cols, rows = img.size
            if label == 1:
                # landmarks
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                landmarks_pair = np.insert(landmarks, obj=2, values=1, axis=1)
                rotated_landmarks = []
                for point in landmarks_pair:
                    rotated_landmark = np.matmul(M, point)
                    rotated_landmarks.append(rotated_landmark)
                landmarks = np.asarray(rotated_landmarks)
            img = np.asarray(img, dtype=np.float32)
        return {'image': img, 'landmarks': landmarks, 'label': label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
        Then do channel normalization: (image - mean) / std_variation
    """
    def channel_norm(self, img):
        mean = np.mean(img)
        std = np.std(img)
        pixels = (img - mean) / (std + 0.0000001)
        return pixels

    def __call__(self, sample):
        image, landmarks, label = sample['image'], sample['landmarks'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image/255.0
        landmarks = landmarks/train_boarder
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks).float(),
                'label': torch.from_numpy(label)}


def get_train_val_data():
    train_file = 'train_data.csv'
    test_file = 'val_data.csv'
    tsfm_train = transforms.Compose([
        Normalize(),                # do channel normalization
        RandomHorizontalFlip(0.5),  # randomly flip image horizontally
        RandomRotate(0.25, 10),          # randomly rotate image
        ToTensor()]                 # convert to torch type: NxCxHxW
    )
    tsfm_test = transforms.Compose([
        Normalize(),
        ToTensor()
    ])
    train_dataset = FaceLandmarksDataset(train_file, phase='train', transform=tsfm_train)
    test_dataset = FaceLandmarksDataset(test_file, phase='test', transform=tsfm_test)
    return train_dataset, test_dataset

def _test_My_data():
    train_set, val_set = get_train_val_data()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=256)
    data_loaders = {'train': train_loader, 'val': valid_loader}
    for i in range(0,10):
        sample = train_loader.dataset[i]
        img = Image.fromarray(sample['image'].astype('uint8'))
        points = sample['landmarks']
        class_ = sample['label']

        landmarks = points.astype('float').reshape(-1, 2)
        draw = ImageDraw.Draw(img)
        x = landmarks[:, 0]
        y = landmarks[:, 1]
        points_zip = list(zip(x, y))
        draw.point(points_zip, (255, 0, 0))
        # img.save(r'H:\DataSet\慧科\人脸关键点检测\result\{:d}.jpg'.format(index))
        plt.imshow(img)
        plt.show()