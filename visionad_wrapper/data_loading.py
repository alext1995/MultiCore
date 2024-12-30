from visionad_wrapper.data.configure_dataset import load_dataset
from torchvision import transforms
from visionad_wrapper.data.load_image_data import create_dataloaders, make_unnormalise
from visionad_wrapper.data.configure_dataset import load_dataset
from PIL import Image

image_size = 256

patchcore_default_model_params =  {"results_path":"test_results", #, type=str)
                                    "seed":0, #, ,type=int, default=0, show_default=True)
                                    "log_group":"group", #, type=str, default="group")
                                    "log_project":"project", #, type=str, default="project")
                                    "backbone_names_layers": {"wideresnet50": ["layer2", "layer3"]}, # each item must be a list of layers
                                    "pretrain_embed_dimension":1024, #, type=int, default=1024)
                                    "target_embed_dimension":1024, #, type=int, default=1024)
                                    "preprocessing":"mean", #, type=click.Choice(["mean", "conv"]), default="mean")
                                    "aggregation":"mean", #, type=click.Choice(["mean", "mlp"]), default="mean")
                                    "anomaly_scorer_num_nn":5,#, ,type=int, default=5)
                                    "patchsize":3, #, ,type=int, default=3)
                                    "patchscore":"max", # , type=str, default="max")
                                    "patchoverlap":0.0 ,#, type=float, default=0.0)
                                    "patchsize_aggregate":[] ,#, "-pa", type=int, multiple=True, default=[])
                                    "faiss_on_gpu": True,#, is_flag=True)
                                    "faiss_num_workers":8, #, ,type=int, default=8)
                                    "percentage":0.01,#, "-p", type=float, default=0.1, show_default=True)
                                    "num_workers":8, #, ,default=8, type=int, show_default=True)
                                    "input_size": 256,#, default=256, type=int, show_default=True)
                                    "augment":True,#, is_flag=True)
                                    "sampler_name":"approx_greedy_coreset",
                                    }

def get_dataloaders(dataset_key, ad_device):
    dataset = load_dataset(dataset_key)

    model_parameters = {"model_params": patchcore_default_model_params,
                        "epochs": 1,
                        "training_transformations":  transforms.Compose([]),
                        "training_transformation_string": "transforms.Compose([])",
                        "identical_transforms": transforms.Compose([transforms.Resize((image_size, image_size), 
                                                                    interpolation=Image.NEAREST),
                                                                    ]),
                        "batch_size": 8,
                        "save_model_every_n_epochs": None,
                        "save_heatmaps_n_epochs": 0,
                        "evaluate_n_epochs": 0,
                        "test_batch_size": 8,
                        "use_gpu": True,
                        }

    model_parameters["pre_pro_transforms"] = transforms.Compose([transforms.Normalize(mean=dataset['mean'], 
                                                                                      std=dataset['std'])])

    dataloader_train, dataloader_regular_test, dataloader_novel_test = create_dataloaders(                                                 
                                                            pre_pro_transforms=model_parameters["pre_pro_transforms"], 
                                                            training_transformations=model_parameters["training_transformations"],
                                                            identical_transforms=model_parameters["identical_transforms"],
                                                            batch_size = 8,
                                                            shuffle = True,
                                                            device = ad_device, 
                                                            unnormalise = make_unnormalise(mean=dataset['mean'], 
                                                                                           std=dataset['std']),
                                                            test_batch_size = model_parameters["test_batch_size"],
                                                            **dataset)
    
    return dataloader_train, dataloader_regular_test, dataloader_novel_test 