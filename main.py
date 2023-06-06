import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from model.unet import UNET
# from model.model import SegFormer
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.loss import FocalLoss
import train
from torchvision.datasets import Cityscapes
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
LEARNING_RATE = 1e-3
NUM_WORKERS = multiprocessing.cpu_count()//4
EPOCHS = 100

# segformer_weights = "seg_weights" + "_" + str(time.time())

# SAVE_PATH = os.path.join("weights",segformer_weights)
SAVE_PATH = 'weights'
def draw_plot(train,val, flag):
    if flag == 'loss':
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, 'b', label='Training Loss')
        plt.plot(epochs, val, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('loss.png')
    if flag == 'iou':
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, 'b', label='Training iou')
        plt.plot(epochs, val, 'r', label='Validation iou')
        plt.title('Training and Validation IOU')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.savefig('iou.png')
    if flag == 'dice':
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, 'b', label='Training Dice Score')
        plt.plot(epochs, val, 'r', label='Validation Dice Score')
        plt.title('Training and Validation Dice Score')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.savefig('Dice.png')

def main():
    train_transform = A.Compose([
    A.Resize(256, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 
    val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),])

    #### add the path to the dataset###
    root_dir = 'dataset_1/gtFine'
    
    
    train_loader = get_cityscapes_data(root_dir = root_dir, mode='fine', split='train', num_workers = NUM_WORKERS, batch_size = 4, transforms = train_transform, shuffle=True)
    val_loader = get_cityscapes_data(root_dir = root_dir,mode='fine', split='val', num_workers = NUM_WORKERS, batch_size = 1, transforms = val_transform, shuffle=True)
    print('data loaded')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "for training")
    # model = SegFormer(
    #     in_channels=3,
    #     widths=[64, 128, 256, 512],
    #     depths=[3, 4, 6, 3],
    #     all_num_heads=[1, 2, 4, 8],
    #     patch_sizes=[7, 3, 3, 3],
    #     overlap_sizes=[4, 2, 2, 2],
    #     reduction_ratios=[8, 4, 2, 1],
    #     mlp_expansions=[4, 4, 4, 4],
    #     decoder_channels=256,
    #     scale_factors=[16, 8, 4, 2],
    #     num_classes=19,
    # )
    model = UNET(3,19)
    print("Model loaded")
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    alpha = torch.tensor([0.9, 0.92, 1.1, 1.02, 0.992, 1.02, 0.97, 0.99, 1.04, 1.06, 0.897, 0.99, 1.04, 1.05, 1.0045, 0.99, 1.032, 1.0054, 1.0076]).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.2, patience=2)
    criterion = FocalLoss(alpha = alpha, gamma = 2, ignore_index = 255)


    ############## training code ######################
    # print("Training Started!")
    # loss_graph_train, miou_graph_train, dice_train_val,loss_graph_val, miou_graph_val, dice_val_scores = train.train_model(num_epochs=EPOCHS, model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_function=criterion, scheduler= scheduler, save_path = SAVE_PATH)
    # draw_plot(loss_graph_train, loss_graph_val, 'loss')
    # draw_plot(miou_graph_train, miou_graph_val,'iou')


    ############## testing in real time #################
    test_path = "stuttgart_01"
    weight_path = "weights/weights_30.pt"
    perform_segmentation(model, device, weight_path, test_path, transform = val_transform)
if __name__ == '__main__':
    main()