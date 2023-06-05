from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
from utils.utils import dice_score
from utils.utils import mIOU


def train(model, device, train_loader, optimizer, loss_function):
    current_loss = 0.0
    current_IOU = 0.0

    for batch in tqdm(train_loader):
        dice = 0
        # image, labels, _, _ = batch
        image, labels = batch
        # print(torch.unique(labels))
        image, labels = image.to(device), labels.to(device)
        
        prediction = model(image)

        # print('prediction shape: ',prediction.shape)
        # print('label shape: ', labels.shape)
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(prediction.cpu().detach().numpy()[0][0])  # Assuming data is in channel-last format

        # # Plot the corresponding label mask
        # plt.subplot(1, 2, 2)
        # plt.imshow(labels[0].cpu().detach().numpy())
        # plt.show()

        # dice += dice_score(prediction, labels)
        optimizer.zero_grad()
        loss = 0.8*loss_function(prediction, labels)  - 0.2*mIOU(labels, prediction)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()*image.size(0)
        current_IOU += mIOU(labels, prediction)

    current_loss = current_loss/len(train_loader)
    current_IOU = current_IOU/len(train_loader)
    # dice = dice/(len(train_loader))
    return current_loss, current_IOU #, dice


  
def evaluate(model, data_loader, device, loss_function):
    current_loss = 0.0
    current_IOU = 0.0
    dice = 0.0
    with torch.no_grad():
        model.eval()
        for image, labels in data_loader:
            image, labels = image.to(device), labels.to(device)
            prediction = model(image)
            loss = 0.8*loss_function(prediction, labels)  - 0.2*mIOU(labels, prediction)
            # dice += dice_score(prediction, labels)
            current_loss += loss.item()*image.size(0)
            current_IOU += mIOU(labels, prediction)
        current_loss = current_loss/len(data_loader)
        current_IOU = current_IOU/len(data_loader)
        # dice = dice/(len(data_loader))
    return current_loss, current_IOU#, dice

def train_model(num_epochs, model, device, train_loader, optimizer, loss_function,  save_path, scheduler = None, val_loader = None):
    loss_graph_train = []
    miou_graph_train = []
    loss_graph_val = []
    miou_graph_val = []
    dice_train_scores = []
    dice_val_scores = []
    model = model.to(device)
    for epoch in range(1, num_epochs+1):
        torch.cuda.empty_cache()
        model.train()
        print("Epoch "+str(epoch))
        train_loss, running_mIOU = train(model, device, train_loader, optimizer, loss_function)
        loss_graph_train.append(train_loss)
        miou_graph_train.append(running_mIOU)
        # dice_train_scores.append(dice_train)
        val_loss, val_mIOU = evaluate(model, val_loader, device, loss_function)
        loss_graph_val.append(val_loss)
        miou_graph_val.append(val_mIOU)
        # dice_val_scores.append(dice_val)
        if scheduler is not None:

            scheduler.step(val_loss)
        
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train IOU: {running_mIOU:.4f},Val Loss: {val_loss:.4f}, Val IOU: {val_mIOU:.4f}')
        if epoch%10 == 0:
            save_checkpoint(save_path=save_path, model=model, optimizer=optimizer, val_loss=0, epoch=epoch)
    return loss_graph_train, miou_graph_train, dice_train_scores, loss_graph_val, miou_graph_val, dice_val_scores


def save_checkpoint(save_path, model, optimizer, val_loss, epoch):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    file_name = save_path.split("/")[-1].split("_")[0] + "_" + str(epoch) + ".pt"
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, os.path.join(save_path, file_name))


    