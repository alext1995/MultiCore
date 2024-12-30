# Multicore

Multicore achieves state-of-the-art anomaly detection on the MVTec and IDW datasets. Published at British Machine Vision Conference (BMVC) 2024.

It achieves this by combining multiple Patchcore detectors with a synthetic anomaly system and a U-net.

Link to paper:

https://bmva-archive.org.uk/bmvc/2024/papers/Paper_70/paper.pdf 

## Data

The MVTec dataset can be found at: 

https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads

The IDW dataset can be found at:

 https://drive.google.com/file/d/1LEoPEhC47lmdvgzER9ubClboaatDjUHq/view?usp=sharing

Once downloaded and unzipped, enter the paths into the MVTEC_PATH and IDW_PATH variables in data/configure_dataset.py

## One dataset class

### Create the training and testing files

python3 create_pcplus_files.py --dataset_key "mvtec_bottle" --ad_device "cuda:0" --data_save_key "default_key"

### Run the U-net

python3 run_unet.py --dataset_key "mvtec_bottle" --device "cuda:3" --data_save_key "test_1_data" --save_key "bottle_test" --epochs 100 --num_models_limit 8

## Multiple datasets classes

python3 create_set_pcplus_files.py --datasets "mvtec" --ad_device "cuda:0" --data_save_key "default_key"

python3 run_set_unet.py --datasets "mvtec" --device "cuda:0" --data_save_key "test_1_data" --save_key "bottle_test" --epochs 100 --num_models_limit 8

The data_save_key is specific to the data made by the pcplusfiles, which is dependent on the synthetic noise parameters. The save_key holds the results for the Unet.

If you set --datasets to 'custom', you can use a custom list of dataset_keys. You can achieve this with the dataset_list variable in create_set_pcplus_files.py.

If --datasets is set to 'mvtec', the MVTec dataset keys will be used, if set to 'idw', the IDW dataset keys will be used

## If you have any questions or find any bugs, please reach out!

## If this work is useful to you, please cite this:
@inproceedings{Taylor_2024_BMVC,
author    = {Alexander D. J. Taylor and Jonathan James Morrison and Phillip Tregidgo and Neill D. F. Campbell},
title     = {Advancing Anomaly Detection: The IDW dataset and MC algorithm},
booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
publisher = {BMVA},
year      = {2024},
url       = {https://papers.bmvc2024.org/0070.pdf}
}

