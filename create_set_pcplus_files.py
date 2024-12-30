from create_pcplus_files import main
import argparse

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
    parser.add_argument("--ad_device", required=False, type=str, default="cuda:0")
    
    parser.add_argument("--weight_min", required=False, type=float, default=0.1)
    parser.add_argument("--weight_max", required=False, type=float, default=0.5)
    
    parser.add_argument("--data_save_key", required=False, type=str, default="default_key")
    args = parser.parse_args()
    
    mask_params = {"weight_min": args.weight_min,
                "weight_max": args.weight_max,
                "weight_type": "gaussian",
                "no_mask_probability": 0.1,
                "multi_mask_factor": 0.6,
                }
        
    if args.datasets.lower()=="custom":
        dataset_list = dataset_list
    elif args.datasets.lower()=="mvtec":
        dataset_list = mvtec_dataset_list
    elif args.datasets.lower()=="idw":
        dataset_list = idw_dataset_list
    else:
        raise ValueError("Invalid dataset list selection. Enter 'custom', 'mvtec', or 'idw'.")
    for dataset_key in dataset_list:
        main(dataset_key, args.ad_device, mask_params, args.data_save_key)
