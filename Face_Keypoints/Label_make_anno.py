import pandas as pd
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt, array
import cv2
import numpy as np


class Data_annotation:
    def get_faces_keypoints(self):
        ROOTS = 'D:\git\learngit\Face_Keypoints\data\\'
        FOLDER = ['I', 'II']

        tmp_lines = []
        DATA_info = {'path': [], 'face_rect': [], 'face_keypoints': []}

        for f in FOLDER:
            DATA_DIR = os.path.join(ROOTS, f)
            file_path = os.path.join(DATA_DIR, 'label.txt')
            with open(file_path) as file:
                lines = file.readlines()
            tmp_lines.extend(list(map((DATA_DIR + r'/').__add__, lines)))

        for file in tmp_lines:
            file = file.strip().split()
            try:
                img = Image.open(file[0])
            except OSError:
                pass
            else:
                DATA_info['path'].append(file[0])
                DATA_info['face_rect'].append(list(map(float, file[1:5])))
                DATA_info['face_keypoints'].append(list(map(float, file[5:])))

        ANNOTATION = pd.DataFrame(DATA_info)
        ANNOTATION.to_csv('face_keypoints_annotation.csv')
        print('face_keypoints_annotation file is saved.')

    def get_train_val_data(self):
        FILE = 'face_keypoints_annotation.csv'
        DATA_info = pd.read_csv(FILE)
        DATA_info_anno = {'path': [], 'rect': [], 'points': []}

        expand_ratio = 0.25
        self.get_valid_data(DATA_info, DATA_info_anno, expand_ratio)
        data = pd.DataFrame(DATA_info_anno)
        # numpy choice, sample
        train_data = data.sample(frac=0.7, replace=False)

        val_data = data[data['path'].isin(list(set(data['path'])-set(train_data['path'])))]

        data.to_csv('train_data.csv')
        data.to_csv('val_data.csv')
        print("train_data:{:d}".format(train_data.shape[0]))
        print("val_data:{:d}".format(val_data.shape[0]))

    def get_valid_data(self, DATA_info, DATA_info_anno, expand_ratio=0.25):
        def expand_roi(rect , expand_ratio):
            left, top, right, bottom = rect
            rect_width = right-left
            rect_height = bottom-top
            left, top, right, bottom = left - expand_ratio * rect_width, \
                                       top - expand_ratio * rect_height,\
                                       right + expand_ratio * rect_width, \
                                       bottom + expand_ratio * rect_height
            return [left, top, right, bottom]

        for index, row in DATA_info.iterrows():
            is_invalid_sample = False
            img = Image.open(row.loc['path'])
            img = img.convert('RGB')
            width, height = img.size
            rect = list(map(lambda x: float(x), eval(row.loc['face_rect'])))
            rect = expand_roi(rect, expand_ratio)
            points = list(map(float, eval(row.loc['face_keypoints'])))
            x = points[0::2]
            y = points[1::2]
            points_zip = list(zip(x, y))
            # 处理Rect不超出图像边界
            rect_dstxy = [0 if i < 0 else i for i in rect]
            rect_dstx = [width if rect_dstxy[i] > width else rect_dstxy[i] for i in [0, 2]]
            rect_dsty = [height if rect_dstxy[i] > height else rect_dstxy[i] for i in [1, 3]]
            rect_dstx.extend(rect_dsty)
            rect = [rect_dstx[i] for i in [0, 2, 1, 3]]
            # 处理Points不超出Rect边界,如果超出则舍去该样本
            left, top, right, bottom = rect
            for point in points_zip:
                x, y = point
                if x < left or x > right or y < top or y > bottom:
                    is_invalid_sample = True
                    print("{:s}:Points is out of rect boundary".format(row.loc['path']))
                    break
            if is_invalid_sample:
                continue
            DATA_info_anno['path'].append(row.loc['path'])
            DATA_info_anno['rect'].append(rect)
            DATA_info_anno['points'].append(points_zip)

            # draw = ImageDraw.Draw(img)
            # draw.rectangle(rect, outline='green')
            # draw.point(points_zip, (255, 0, 0))
            # img.save(r'H:\DataSet\result\{:d}.jpg'.format(index))
            # # plt.imshow(img)
            # # plt.show()


if __name__ == '__main__':
    data_anno = Data_annotation()
    data_anno.get_faces_keypoints()
    data_anno.get_train_val_data()


