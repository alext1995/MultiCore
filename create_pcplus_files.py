from synthetic_anomalies.synthetic_anomalies import StratifiedMaskHolder
from visionad_wrapper.data_loading import get_dataloaders
from patchcoreplus.patchcoreplus import WrapperPatchCorePlus
from patchcoreplus.patchcore_models import create_model_parameters
import pickle
import sys
import torch
import time
import os

class SyntheticTrainingDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_holder = {}
        
        self.images = []
        self.paths  = []
        self.masks  = []
        self.to_pass = []
        for ind, items in enumerate(self.dataloader):
            image_batch, paths_batch, callback_info_batch = items
            self.images.append(image_batch)       
            self.paths.append(paths_batch)         
            self.masks.append(callback_info_batch)
            self.to_pass.append((image_batch, callback_info_batch,))
            
    def __len__(self,):
        return len(self.to_pass)

def make_synthetic_training_ad_dataloader(dataloader_train, 
                                          mask_holder_params,
                                          amount=0):
    stratmask = StratifiedMaskHolder(**mask_holder_params)
    dataloader_train.dataset.pre_normalisation_transform = stratmask.add_noise_to_image 
    
    with torch.no_grad():
        if amount==0:
            datas_train = [items for ind, items in enumerate(dataloader_train)]
        else:
            datas_train = []
            while len(datas_train)<amount//dataloader_train.batchsize:
                datas_train += [items for ind, items in enumerate(dataloader_train)]
            start = time.time()
            datas_train = datas_train[:(amount//dataloader_train.batchsize)]
            
    return SyntheticTrainingDataLoader(datas_train)

def main(dataset_key, ad_device, mask_params, data_save_key):
    dataloader_train, dataloader_regular_test, dataloader_novel_test = get_dataloaders(dataset_key, ad_device)

    synthetic_training_ad_dataloader = make_synthetic_training_ad_dataloader(dataloader_train,
                                                                                mask_params,
                                                                                amount=0)
    masks = synthetic_training_ad_dataloader.masks

    hyperparameters = create_model_parameters(ad_device)

    os.makedirs(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}"), exist_ok=True)
    for model_ind, parameters in enumerate(hyperparameters):

        algo_class = WrapperPatchCorePlus(**parameters)

        algo_class.enter_dataloaders(dataloader_train, 
                                     dataloader_regular_test, 
                                     dataloader_novel_test)

        algo_class.pre_eval_processing()

        preds = algo_class.eval_outputs_dataloader(synthetic_training_ad_dataloader.to_pass, 
                                                   len(synthetic_training_ad_dataloader.to_pass))
        
        if model_ind==0:
            with open(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}/train_{dataset_key}_{model_ind}.pk"), "wb") as f:
                preds = preds[0]['segmentations']
                pickle.dump((preds, masks), f)
        else:
            with open(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}/train_{dataset_key}_{model_ind}.pk"), "wb") as f:
                preds = preds[0]['segmentations']
                pickle.dump((preds, None), f)
        
        preds, masks = None, None

        heatmap_set, image_set, targets_set, paths_set = algo_class.test()
        
        # order the arrays
        arrange = np.argsort(paths_set["paths_novel"])
        paths_set["paths_novel"] = np.array(paths_set["paths_novel"])[arrange]
        targets_set['targets_novel'] = targets_set['targets_novel'][arrange]
        image_set["image_score_set_novel"]["scores"] = image_set["image_score_set_novel"]["scores"][arrange] 
        heatmap_set["heatmap_set_novel"]["segmentations"] = heatmap_set["heatmap_set_novel"]["segmentations"][arrange]

        arrange = np.argsort(paths_set["paths_regular"])
        paths_set["paths_regular"] = np.array(paths_set["paths_regular"])[arrange]
        image_set["image_score_set_regular"]["scores"] = image_set["image_score_set_regular"]["scores"][arrange]
        heatmap_set["heatmap_set_regular"]["segmentations"] = heatmap_set["heatmap_set_regular"]["segmentations"][arrange]

        
        
        regular_preds = heatmap_set["heatmap_set_regular"]["segmentations"]
        anomalous_preds = heatmap_set["heatmap_set_novel"]["segmentations"]

        regular_image_scores = image_set["image_score_set_regular"]
        anomalous_image_scores = image_set["image_score_set_novel"]
        
        if model_ind==0:
            with open(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}/test_{dataset_key}_{model_ind}.pk"), "wb") as f:
                pickle.dump((anomalous_preds, 
                                regular_preds, 
                                anomalous_image_scores, 
                                regular_image_scores,
                                paths_set,
                                targets_set), f)
        else:
            with open(os.path.join(os.getcwd(), f"data/{data_save_key}/{dataset_key}/test_{dataset_key}_{model_ind}.pk"), "wb") as f:
                pickle.dump((anomalous_preds, 
                                regular_preds, 
                                anomalous_image_scores, 
                                regular_image_scores,
                                paths_set,
                                None), f)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset_key', type=str, required=True, help='Dataset key, e.g. mvtec_bottle, BirdStrike')
    parser.add_argument('--ad_device', type=str, required=True, help='GPU identifier - etc "cuda:0"')
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
        
    main(args.dataset_key, args.ad_device, mask_params, args.data_save_key)
