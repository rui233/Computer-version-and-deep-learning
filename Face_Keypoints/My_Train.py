import os
import Config
import torch
import torch.nn as nn
from My_data import get_train_val_data
from My_net import My_Net as Net
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms

def train(data_loaders, model, criterion):
    if Config.MODEL_SAVE_PATH:
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            os.makedirs(Config.MODEL_SAVE_PATH)

    if Config.DEVICE:
        device = torch.device("cuda" if Config.DEVICE == 'gpu' else "cpu")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    epoch = Config.EPOCH
    # 准则
    pts_criterion = criterion[0]
    cls_criterion = criterion[1]


    LR = Config.LEARNING_RATE
    opt_SGD = torch.optim.SGD(model.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizers = {"opt_SGD": opt_SGD, "opt_Momentum": opt_Momentum, "opt_RMSprop": opt_RMSprop, "opt_Adam": opt_Adam}
    optimizer = optimizers[Config.OPTIMIRZER]

    train_losses = []
    valid_losses = []

    for epoch_id in range(epoch):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                if Config.FLAG_RESTORE_MODEL and os.path.exists(Config.MODEL_NAME):
                    Config.FLAG_RESTORE_MODEL = False
                    model.load_state_dict(torch.load(Config.MODEL_NAME))
                model.train()
            else:
                model.eval()

            for batch_idx, batch in enumerate(data_loaders[phase]):

                img = batch['image'].to(device)
                landmark = batch['landmarks'].reshape(-1, 42).to(device)
                label = batch['label'].to(device)
                optimizer.zero_grad()
                # print(img.shape)
                # print(landmark.shape)
                # clear the gradients of all optimized variables

                with torch.set_grad_enabled(phase == 'train'):
                    output_pts, output_cls = model(img)

                    # due with positive samples
                    positive_mask = label == 1
                    positive_mask = np.squeeze(positive_mask)
                    len_true_positive = positive_mask.sum().item()
                    if len_true_positive == 0:
                        loss_positive = 0
                        pred_class_pos_correct = 0
                        # print(len_true_positive)
                    else:
                        loss_positive_pts = pts_criterion(output_pts[positive_mask], landmark[positive_mask])

                        # print(output_cls[positive_mask])
                        # print(label[positive_mask].squeeze(1))
                        loss_positive_cls = cls_criterion(output_cls[positive_mask], label[positive_mask].squeeze(1))
                        loss_positive = 10 * loss_positive_pts + loss_positive_cls

                        positive_pred_class = output_cls[positive_mask].argmax(dim=1, keepdim=True)
                        # print(positive_pred_class)
                        pred_class_pos_correct = positive_pred_class.eq(label[positive_mask]).sum().item()

                    # due with negative samples (no coordinates)
                    negative_mask = label == 0
                    negative_mask = np.squeeze(negative_mask)
                    len_true_negative = negative_mask.sum().item()
                    if len_true_negative == 0:
                        loss_negative = 0
                        pred_class_neg_correct = 0
                        # print(len_true_negative)
                    else:
                        # print("1:{}".format(output_cls[negative_mask]))
                        # print("2:{}".format(label[negative_mask].squeeze(1)))
                        loss_negative_cls = cls_criterion(output_cls[negative_mask], label[negative_mask].squeeze(1))
                        loss_negative = loss_negative_cls

                        negative_pred_cls = output_cls[negative_mask].argmax(dim=1, keepdim=True)
                        pred_class_neg_correct = negative_pred_cls.eq(label[negative_mask]).sum().item()

                    # sum up
                    # print(loss_positive)
                    loss = loss_positive + loss_negative

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                if batch_idx % Config.LOG_INTERVAL == 0:

                    pred_class = output_cls.argmax(dim=1, keepdim=True)
                    index_img_eval = np.random.randint(0, len(img), size=5)
                    for j in index_img_eval:
                        img_ = img[j, :, :, :] * 255
                        landmark_ = output_pts[j, :] * 112
                        img_ = Image.fromarray(img_.cpu().numpy().transpose((1, 2, 0)).astype('uint8'))
                        if pred_class[j]:
                            draw = ImageDraw.Draw(img_)
                            x = landmark_[::2]
                            y = landmark_[1::2]
                            points_zip = list(zip(x, y))
                            draw.point(points_zip, (255, 0, 0))
                        if not os.path.exists(Config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH):
                            os.mkdir(Config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH)
                        if not os.path.exists(Config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH + '\\' + phase):
                            os.mkdir(Config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH + '\\' + phase)

                        img_.save(Config.RESULT_TRAIN_LOG_IMGS_SAVE_PATH
                                  + '\\' + phase + '\\' + str(epoch_id) + '_' + str(batch_idx) + '_' + str(j) + '.jpg')

                    print('{} Epoch: {} [{}/{} ({:.0f}%)]\t\
                    Loss: {:.6f}\tloss_positive_pts: {:.6f}\t loss_positive_cls: {:.6f}\tloss_negative_cls: {:.6f}\t\
                    {} Pos acc: [{}/{} ({:.2f}%)]\tNeg acc: [{}/{} ({:.2f}%)]\tAcc: [{}/{} ({:.2f}%) {}]'.format(
                        phase,
                        epoch_id,
                        batch_idx * len(img),
                        len(data_loaders[phase].dataset),
                        100. * batch_idx / len(data_loaders[phase]),
                        # training losses: total loss, regression loss, classification loss: positive & negative samples
                        loss,
                        loss_positive_pts,
                        loss_positive_cls,
                        loss_negative_cls,
                        phase,
                        # training accuracy: positive samples in a batch
                        pred_class_pos_correct,
                        len_true_positive,
                        100. * pred_class_pos_correct / (len_true_positive+0.001),
                        # training accuracy: negative samples in a batch
                        pred_class_neg_correct,
                        len_true_negative,
                        100. * pred_class_neg_correct / (len_true_negative+0.001),
                        # training accuracy: total samples in a batch
                        pred_class_pos_correct + pred_class_neg_correct,
                        img.shape[0],
                        100. * (pred_class_pos_correct + pred_class_neg_correct) / img.shape[0], phase)
                    )
                if phase == 'train' and epoch_id % Config.SAVE_MODEL_INTERVAL == 0:
                    saved_model_name = os.path.join(Config.MODEL_SAVE_PATH, 'aligner_epoch' + '_' + str(epoch_id) + '.pt')
                    torch.save(model.state_dict(), saved_model_name)

    return loss

def main():
    print('====> Loading Datasets')
    train_set, val_set = get_train_val_data()
    # train_sampler = SubsetRandomSampler()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=Config.BATCH_TRAIN, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=Config.BATCH_VAL)
    data_loaders = {'train': train_loader, 'val': valid_loader}

    # sample = train_loader.dataset[2]
    # print(2, sample['image'].shape, sample['label'])

    print('===> Building Model')
    # For single GPU
    model = Net()

    criterion_pts = nn.MSELoss()
    weights = [1, 3]
    class_weights = torch.FloatTensor(weights)
    class_weights = class_weights.to('cuda')
    criterion_cls = nn.CrossEntropyLoss()

    if Config.PHASE == 'Train' or Config.PHASE == 'train':
        print('===> Start Training')
        _ = train(data_loaders, model, (criterion_pts, criterion_cls))
        print('=================Finished Train===================')
    elif Config.PHASE == 'Finetune' or Config.PHASE == 'finetune':
        print('===> Finetune')
        # model.load_state_dict(torch.load(os.path.join(args.save_directory, 'aligner_epoch_28.pt')))
    elif Config.PHASE == 'Predict' or Config.PHASE == 'predict':
        print('===> Predict')


if __name__ == '__main__':
    main()

