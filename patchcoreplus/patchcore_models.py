ad_device = "cuda:0"

base_hyperparameters = {"results_path":"test_results", #, type=str)
                        "seed":0, #, ,type=int, default=0, show_default=True)
                        "log_group":"group", #, type=str, default="group")
                        "log_project":"project", #, type=str, default="project")
                        "backbone_names_layers": {"wideresnet50": ["layer2", "layer3"]}, # each item must be a list of layers
                        "pretrain_embed_dimension":1024, #, type=int, default=1024)
                        "target_embed_dimension":1024, #, type=int, default=1024)
                        "preprocessing":"mean", #, type=click.Choice(["mean", "conv"]), default="mean")
                        "aggregation":"mean", #, type=click.Choice(["mean", "mlp"]), default="mean")
                        "anomaly_scorer_num_nn":5,#, ,type=int, default=5)
                        "patchscore":"max", # , type=str, default="max")
                        "patchoverlap":0.0 ,#, type=float, default=0.0)
                        "patchsize_aggregate":[] ,#, "-pa", type=int, multiple=True, default=[])
                        "faiss_on_gpu": True,#, is_flag=True)
                        "faiss_num_workers":8, #, ,type=int, default=8)
                        "num_workers":8, #, ,default=8, type=int, show_default=True)
                        "input_size": 256,#, default=256, type=int, show_default=True)
                        "augment":True,#, is_flag=True)
                        "coreset_size": 25000, #, "-p", type=float, default=0.1, show_default=True)
                        "coreset_search_dimension_map": 128,
                        "nn_code": "new",
                        }

def make_params_codes(ad_device):
    params_out = []
    codes_out = []
    for distance_type in ["euclidean", "cosine", "minkowski3", "manhatten"]:
        for patchstride in [1, 3, 5]:
            for patchsize in [3, 1, 5, 7]:
                codes_out.append(f"{distance_type}, stride: {patchstride}, size: {patchsize}")
             
                params = {key:item for key, item in base_hyperparameters.items()}
                params["distance_type"] = distance_type
                params["patchsize"] = patchsize
                params["patchstride"] = patchstride
                params["device"] = ad_device
                params_out.append(params)
    return params_out, codes_out

selected_keys = [0, 25, 7, 4, 3, 5, 1, 8]

def create_model_parameters(ad_device):
    params, codes = make_params_codes(ad_device)
    return [params[key] for key in selected_keys]
        