from run_unet import main
import sys
import argparse 
from unet_modelling.loss_functions import (BCE_cl_dice_loss, 
                                           SSIM_cl_dice_loss, 
                                           get_name_critertion)
from unet_modelling.transformations import *

dataset_list = ['mvtec_bottle',
                'mvtec_cable',
                'mvtec_capsule',
                'mvtec_carpet',
                'mvtec_grid',
                'mvtec_hazelnut',
                'mvtec_leather',
                'mvtec_metal_nut',
                'mvtec_pill',
                'mvtec_screw',
                'mvtec_tile',
                'mvtec_toothbrush',
                'mvtec_transistor',
                'mvtec_wood',
                'mvtec_zipper']

mvtec_dataset_list = ['mvtec_bottle',
                      'mvtec_cable',
                      'mvtec_capsule',
                      'mvtec_carpet',
                      'mvtec_grid',
                      'mvtec_hazelnut',
                      'mvtec_leather',
                      'mvtec_metal_nut',
                      'mvtec_pill',
                      'mvtec_screw',
                      'mvtec_tile',
                      'mvtec_toothbrush',
                      'mvtec_transistor',
                      'mvtec_wood',
                      'mvtec_zipper']

idw_dataset_list = ["BirdStrike",
                    "CarBody",
                    "HouseStructure",
                    "HeadGasket",
                    "NozzleGuideVane",
                    "InspectionPipes",
                    "Combined",]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=False, type=str, default="mvtec")
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

    if args.datasets.lower()=="custom":
        dataset_list = dataset_list
    elif args.datasets.lower()=="mvtec":
        dataset_list = mvtec_dataset_list
    elif args.datasets.lower()=="idw":
        dataset_list = idw_dataset_list
    else:
        raise ValueError("Invalid dataset list selection. Enter 'custom', 'mvtec', or 'idw'.")
  
    transforms = [
        RandomHorizontalFlipPair(p=0.5),
        RandomVerticalFlipPair(p=0.5),
        RandomAffinePair(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
        RandomRotationPair(degrees=(-45, 45)),
        RandomNoisePair(mean_range=(-args.mean_value, args.mean_value), std_range=(0.01, 0.05)),
    ]
    
    criterion = SSIM_cl_dice_loss(weight=args.ssim_weight)
    criterion_name  = get_name_critertion(criterion)
    
    for dataset_key in dataset_list:
        save_key = args.save_key
        if args.append_save_key:
            save_key += f"{dataset_key}_noise{args.mean_value}_{criterion_name}_unet{args.num_downs}_ssim{args.ssim_weight}_lr{args.lr}_bs{args.batch_size}"
        
        print("Running dataset: ", dataset_key)
        main(dataset_key, 
             args.data_save_key, 
             args.device, 
             save_key, 
             args.epochs, 
             args.num_models_limit, 
             transforms=transforms, 
             criterion=criterion,
             lr=args.lr,
             batch_size=args.batch_size,
             num_downs = args.num_downs,
             )
