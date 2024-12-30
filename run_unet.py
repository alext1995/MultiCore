from unet_modelling.specialised_models import make_refining_model
from unet_modelling.loss_functions import FocalLoss
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import sys
import numpy as np
import pandas as pd
import os
from unet_modelling.loss_functions import ssim_loss
from torch.nn import MSELoss
from torchvision import transforms as T
import pickle
from visionad_wrapper.wrapper.wrapper import run_metrics
import argparse 
from unet_modelling.loss_functions import (BCE_cl_dice_loss, 
                                           SSIM_cl_dice_loss, 
                                           get_name_critertion)

from unet_modelling.transformations import *
   
class RefiningDataset(torch.utils.data.Dataset):
    def __init__(self, heatmaps, masks, transforms=[]):
        self.heatmaps = heatmaps
        self.masks    = masks
        self.transforms = transforms

    def __len__(self,):
        return len(self.heatmaps)
    
    def __getitem__(self, idx):
        heatmaps, masks = self.heatmaps[idx], self.masks[idx]
        for t in self.transforms:
            heatmaps, masks = t(heatmaps, masks) 
        return heatmaps, masks
        
class MeanChannelNet(nn.Module):
    def __init__(self):
        super(MeanChannelNet, self).__init__()

        # This layer will compute the weighted sum of the input channels
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False)

        # Initialize the weights to 1/3 to compute the mean
        with torch.no_grad():
            self.conv.weight = nn.Parameter(torch.ones(1, 3, 1, 1) / 3)

    def forward(self, x):
        return self.conv(x)
    
def refine_dataloader(model, dataloader, device, mean, std, normalise):
    preds_out = torch.empty(0, 1, 256, 256)
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device).float()
            if normalise:
               images = (images-mean[None, :, None, None])/std[None, :, None, None]
            if isinstance(model, torch.nn.Module):
                preds = model(images)
            else:
                preds = model(images)            
            preds_out = torch.vstack((preds_out, preds.to("cpu")))
    return preds_out

def return_refined_predictions(model, regular_refining_dataloader, novel_refining_dataloader, device, mean, std, normalise):
    novel_refined   = refine_dataloader(model, novel_refining_dataloader, device, mean, std, normalise)
    regular_refined = refine_dataloader(model, regular_refining_dataloader, device, mean, std, normalise)
    return novel_refined, regular_refined

def calculate_test_metrics(metric_list, 
                           refining_model, 
                           regular_refining_dataloader, 
                           novel_refining_dataloader, 
                           test_targets,
                           device, 
                           mean, 
                           std,
                           normalise):
    novel_refined, regular_refined = return_refined_predictions(refining_model, 
                                                                regular_refining_dataloader, 
                                                                novel_refining_dataloader, 
                                                                device, 
                                                                mean, 
                                                                std,
                                                                normalise)
    
    heatmap_set_here = {"heatmap_set_novel": {"refined": novel_refined},
                        "heatmap_set_regular": {"refined": regular_refined}}
            
    score_set_here = {"image_score_set_regular": {"refined": novel_refined.max(dim=-1).values.max(dim=-1).values},
                      "image_score_set_novel": {"refined": regular_refined.max(dim=-1).values.max(dim=-1).values}}

    target_set = {"targets_novel": test_targets,}

    metric_scores = run_metrics(metric_list, heatmap_set_here, score_set_here, target_set, None, print_=False)

    scores_out = []
    for metric in metric_list:
        try:
            #print(metric_scores[metric]["metric_results"]["refined"], end="\t")
            scores_out.append(metric_scores[metric]["metric_results"]["refined"])
        except:
            #print(0, end="\t")
            scores_out.append(0)

    return scores_out

class ResidualModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ResidualModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return x.mean(1)[:,None] + self.model(x)

def main(dataset_key, data_save_key, device, save_key, num_epochs, num_models_limit, 
         transforms, 
         criterion,
         lr,
         batch_size,
         num_downs = 7,
         ):

    anomalous_preds = []
    regular_preds  = []
    training_preds = []
    for model_ind in list(range(8))[:num_models_limit]:
        if model_ind==0:
            with open(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}/train_{dataset_key}_{model_ind}.pk"), "rb") as f:
                (preds, masks) = pickle.load(f)
                masks = torch.vstack(masks)[:,None]

            with open(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}/test_{dataset_key}_{model_ind}.pk"), "rb") as f:
                (a_preds, 
                r_preds, 
                _, 
                _,
                _,
                targets_set) = pickle.load(f)
                test_targets = targets_set['targets_novel']
        else:
            with open(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}/train_{dataset_key}_{model_ind}.pk"), "rb") as f:
                (preds, _) = pickle.load(f)

            with open(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}/test_{dataset_key}_{model_ind}.pk"), "rb") as f:
                (a_preds, 
                r_preds, 
                _, 
                _,
                _,
                _) = pickle.load(f)
        anomalous_preds.append(a_preds)
        regular_preds.append(r_preds)
        training_preds.append(preds)

    training_preds  = torch.hstack(training_preds)
    anomalous_preds = torch.hstack(anomalous_preds)
    regular_preds   = torch.hstack(regular_preds)

    # create training datasets and dataloaders
    training_dataset    = RefiningDataset(training_preds, 
                                          masks,
                                          transforms=transforms)
    training_dataloader = DataLoader(training_dataset, 
                                     batch_size = batch_size, 
                                     shuffle = False)

    # load the training and testing data
    novel_refining_dataset    = RefiningDataset(anomalous_preds, 
                                                test_targets)
    novel_refining_dataloader = DataLoader(novel_refining_dataset, 
                                           batch_size = 8, 
                                           shuffle = False)

    regular_refining_dataset    = RefiningDataset(regular_preds, 
                                                  torch.zeros_like(regular_preds))
    regular_refining_dataloader = DataLoader(regular_refining_dataset, 
                                             batch_size = 8, 
                                             shuffle = False)

    from unet_modelling.unet_generator import  UnetGenerator
    refining_model = UnetGenerator(input_nc=training_preds.shape[1], output_nc=1, num_downs=num_downs).to(device )
    refining_model = ResidualModelWrapper(refining_model).to(device)
    normalise = True

    optimiser = optim.Adam(refining_model.parameters(), lr=lr)

    for ii, (images, _) in enumerate(training_dataloader):
        if ii==0:
            training_images = torch.empty((0, images.shape[1], 256, 256))
        training_images = torch.vstack((training_images, images))
        if training_images.shape[0]>800:
            break
    mean, std = torch.mean(training_images, 
                        axis=[0, 2, 3]).to(device), torch.std(training_images, 
                                                                axis=[0, 2, 3]).to(device)    
        
    metric_list = ["Imagewise_AUC", "Pixelwise_AUC", "Pixelwise_AUPRO", "PL"]

#     models = [lambda x, i=i: x[:, i][:, None] for i in range(num_models_limit)]
#     models += [lambda x: x.mean(dim=1)[:, None]]
    
#     for ii, (code, model) in enumerate(zip([f"{i}" for i in np.arange(num_models_limit)]+["mean"], 
#                                             models)):
#         scores = calculate_test_metrics(metric_list, 
#                                         model, 
#                                         regular_refining_dataloader, 
#                                         novel_refining_dataloader, 
#                                         test_targets,
#                                         device, 
#                                         mean, 
#                                         std,
#                                         normalise)
#         print(f"PC {code:<8}: {[f'{item:<6.3g}' for item in scores]}")
    
    mean_scores = calculate_test_metrics(metric_list, 
                                         lambda x: x.mean(dim=1)[:, None], 
                                         regular_refining_dataloader, 
                                         novel_refining_dataloader, 
                                         test_targets,
                                         device, 
                                         mean, 
                                         std,
                                         normalise)
   
    print("\n\n")
    test_scores = [mean_scores]
    print(f"{'Epoch ':<8}: {[f'{item:<6}' for item in ['I-AUC', 'P-AUC', 'AUPRO', 'PL']]}")
    print(f"{'Models mean '}: {[f'{item:<6.3g}' for item in mean_scores]}", " loss: ", )
    for epoch in range(100):
        mean_loss = []
        for images, masks in training_dataloader:
            images, masks = images.to(device), masks.to(device)
            
            if normalise:
                images = (images-mean[None, :, None, None])/std[None, :, None, None]

            optimiser.zero_grad()
            outputs = refining_model(images.float())
            loss = criterion(outputs, masks.float())
            loss.backward()
            mean_loss.append(loss.item())
            optimiser.step()
        
        if epoch%5==0:
            mean_loss = np.mean(mean_loss)
            scores = calculate_test_metrics(metric_list, 
                                            refining_model, 
                                            regular_refining_dataloader, 
                                            novel_refining_dataloader, 
                                            test_targets,
                                            device, 
                                            mean, 
                                            std,
                                            normalise)
        
            print(f"{'Epoch ' + str(epoch):<8}: {[f'{item:<6.3g}' for item in scores]}", " loss: ", mean_loss)
            test_scores.append(scores)

    print("\n\n")
    print(np.array(test_scores))

    df = pd.DataFrame(np.array(test_scores))
    df.columns = metric_list

    os.makedirs(os.path.join(os.getcwd(), f"results/{data_save_key}/{save_key}"), exist_ok=True)
    df.to_csv(os.path.join(os.getcwd(), f"results/{data_save_key}/{save_key}/{dataset_key}.csv"))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_key", type=str)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_models_limit", type=int, default=8)
    parser.add_argument("--mean_value", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ssim_weight", type=float, default=0.25)
    parser.add_argument("--append_save_key", type=int, default=1)
    parser.add_argument("--num_downs", type=int, default=7)
    parser.add_argument("--data_save_key", required=False, type=str, default="default_key")
    args = parser.parse_args()
   
    transforms = [
        RandomHorizontalFlipPair(p=0.5),
        RandomVerticalFlipPair(p=0.5),
        RandomAffinePair(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
        RandomRotationPair(degrees=(-45, 45)),
        RandomNoisePair(mean_range=(-args.mean_value, args.mean_value), std_range=(0.01, 0.05)),
    ]

    criterion = SSIM_cl_dice_loss(weight=args.ssim_weight)
    criterion_name  = get_name_critertion(criterion)


    dataset_key = args.dataset_key
    device = args.device
    save_key = args.save_key
    num_epochs = args.epochs
    num_models_limit = args.num_models_limit
            
    if args.append_save_key:
        save_key += f"{dataset_key}_noise{args.mean_value}_{criterion_name}_unet{args.num_downs}_ssim{args.ssim_weight}_lr{args.lr}_bs{args.batch_size}"

    main(dataset_key, args.data_save_key, device, save_key, num_epochs, num_models_limit, 
         transforms=transforms, 
         criterion=criterion,
         lr=args.lr,
             batch_size=args.batch_size,
             num_downs = args.num_downs,
         )
