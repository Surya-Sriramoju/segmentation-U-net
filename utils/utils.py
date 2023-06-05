import numpy as np
from torchvision import transforms
import torch
from tqdm import tqdm
from dataset.CityscapesDataset import CityscapesDataset
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import time
from cityscapesscripts.helpers.labels import trainId2label as t2l
import torch.nn.functional as F
from torchmetrics import Dice

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

colormap = np.zeros((19, 3), dtype=np.uint8)
colormap[0] = [128, 64, 128]
colormap[1] = [244, 35, 232]
colormap[2] = [70, 70, 70]
colormap[3] = [102, 102, 156]
colormap[4] = [190, 153, 153]
colormap[5] = [153, 153, 153]
colormap[6] = [250, 170, 30]
colormap[7] = [220, 220, 0]
colormap[8] = [107, 142, 35]
colormap[9] = [152, 251, 152]
colormap[10] = [70, 130, 180]
colormap[11] = [220, 20, 60]
colormap[12] = [255, 0, 0]
colormap[13] = [0, 0, 142]
colormap[14] = [0, 0, 70]
colormap[15] = [0, 60, 100]
colormap[16] = [0, 80, 100]
colormap[17] = [0, 0, 230]
colormap[18] = [119, 11, 32]

def get_cityscapes_data(
    mode,
    split,
    root_dir='datasets/gtFine/',
    target_type="semantic",
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,
    num_workers=2

):
    data = CityscapesDataset(
        mode=mode, split=split, target_type=target_type,transform=transforms, root_dir=root_dir, eval=eval)

    data_loaded = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    return data_loaded

def decode_segmap(pred_labs):
    pred_labs = pred_labs.cpu().numpy()[0]

    r = pred_labs.copy()
    g = pred_labs.copy()
    b = pred_labs.copy()
    for l in range(0, 19):
        r[pred_labs == l] = colormap[l][0]
        g[pred_labs == l] = colormap[l][1]
        b[pred_labs == l] = colormap[l][2]

    rgb = np.zeros((pred_labs.shape[0], pred_labs.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    rgb  = cv2.cvtColor(rgb.astype('float32'), cv2.COLOR_RGB2BGR) 
    return rgb

def get_tensor(path, transform = None):
    frame = Image.open(path)
    if transform is not None:
            original_image = np.array(frame)
            transformed = transform(image = original_image)
            image = transformed["image"]
            image = image.unsqueeze(0)
            return image, original_image
    return None, None

def get_labels(predictions):
    predictions = torch.nn.functional.softmax(predictions, dim=1)
    pred_labels = torch.argmax(predictions, dim=1) 
    pred_labels = pred_labels.float()
    return pred_labels

def perform_segmentation(model, device, weight_path, video_path, transform = None):

    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    vid_name = 'test_video'
    time_stamp = str(time.time())
    video_output = cv2.VideoWriter(vid_name+'_'+time_stamp+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (2048, 1024))

    model.to(device)
    alpha = 0.5
    for frame_name in tqdm(sorted(os.listdir(video_path))):
        tensor_frame, original_frame = get_tensor(video_path+'/'+frame_name, transform)
        start_time = time.time()
        segmentation_map = predict_img(model, tensor_frame, transform)
        end_time = time.time()
        segmentation_map = cv2.normalize(segmentation_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # original_frame = cv2.resize(original_frame, None, fx = 0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        final_frame = cv2.addWeighted(segmentation_map, alpha,  original_frame, abs(1-alpha), 0, original_frame)
        time_taken = end_time - start_time
        fps = round(1/time_taken, 2)
        video_output.write(final_frame)
    video_output.release()
    
def predict_img(model, image, transform):
    with torch.no_grad():
        image = image.to(device)
        predictions = model(image) 
        pred_labels = get_labels(predictions)
        pred_labels = transforms.Resize((1024, 2048))(pred_labels)
        color_labels = decode_segmap(pred_labels)
        return color_labels
    
def dice_score(predicted, target, smooth=1e-5):
    dice = Dice(average='micro').to(device)
    return dice(predicted, target)

def mIOU(label, pred, num_classes=19):
    pred = F.softmax(pred, dim=1) 
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()
    each_image = list()
    og_pred = pred
    og_label = label
    for i in range(label.shape[0]):
        pred = og_pred[i]
        label = og_label[i]
        for sem_class in range(num_classes):
            pred_inds = (pred == sem_class)
            target_inds = (label == sem_class)
            if target_inds.long().sum().item() == 0:
                iou_now = float('nan')
            else: 
                intersection_now = (pred_inds[target_inds]).long().sum().item()
                union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
                iou_now = float(intersection_now) / float(union_now)
                present_iou_list.append(iou_now)
            iou_list.append(iou_now)
        each_image.append(np.mean(present_iou_list))
    return np.mean(each_image)