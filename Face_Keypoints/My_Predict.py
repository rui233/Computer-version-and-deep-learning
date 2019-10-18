import os
import Config
import torch
from My_net import My_Net as Net
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms

def predict_img(model, test_img_name):
    if Config.MODEL_NAME:
        if not os.path.exists(Config.MODEL_NAME):
            print("No Model!")
            return

    img_src = Image.open(test_img_name)
    width, height = img_src.size
    img_src = np.asarray(img_src.convert('RGB').resize(Config.NET_IMG_SIZE, Image.BILINEAR), dtype=np.float32)
    img = img_src.transpose((2, 0, 1))
    img = img/255.0
    img = torch.from_numpy(img).unsqueeze(0)

    model.load_state_dict(torch.load(Config.MODEL_NAME))
    model.eval()
    output_pts, output_cls = model(img)
    pred_class = output_cls.argmax(dim=1, keepdim=True).squeeze()
    output_pts = output_pts.squeeze() * Config.NET_IMG_SIZE[0]
    print(pred_class)
    if pred_class:
        x = output_pts[::2]
        y = output_pts[1::2]
        img_src = Image.fromarray(img_src.astype('uint8'))
        draw = ImageDraw.Draw(img_src)
        points_zip = list(zip(x, y))

        if len(img_src.getbands()) == 4:
            draw.point(points_zip, (255, 0, 0))
        else:
            draw.point(points_zip, 255)
    if not os.path.exists(Config.RESULT_IMGS_SAVE_PATH):
        os.mkdir(Config.RESULT_IMGS_SAVE_PATH)
    img_src.save(Config.RESULT_IMGS_SAVE_PATH+'\\'+os.path.basename(test_img_name))
    plt.imshow(img_src)
    plt.show()


def main():
    model = Net()
    test_img_name = Config.IMG_TO_PREDICT
    if Config.PHASE == 'Train' or Config.PHASE == 'train':
        print('===> Start Training')
        print('=================Finished Train===================')
    elif Config.PHASE == 'Finetune' or Config.PHASE == 'finetune':
        print('===> Finetune')
        # model.load_state_dict(torch.load(os.path.join(args.save_directory, 'aligner_epoch_28.pt')))
    elif Config.PHASE == 'Predict' or Config.PHASE == 'predict':
        print('===> Predict')
        predict_img(model, test_img_name)


if __name__ == '__main__':
    main()

