# Unifuncgrasp

Functional Dexterous Hand Gesture Generation

## Prerequisites:

    git clone git@github.com:cxcAxxy/Unifuncgrasp.git

    conda create -n unifuncgrapnet python=3.8 -y
    conda activate unifuncgraspnet 

    pip install -r requirements.txt

## Get Started

Training

You need to modify the configuration file based on your requirements. Below are the key parameters commonly adjusted in the `config/` folder:

- `train.yaml`
  - `name`: Specify the training model name.
  - `gpu`: Set the GPU ID based on the available GPU device(s).
  - `training/max_epochs`: Define the number of training epochs.
- `model.yaml`
  - The `transformer_dim` defines the embedding size of the transformer, while `hidden_dim` specifies the dimension of the hidden layers in the CVAE network.
- `dataset/cmap_dataset.yaml`
  - `robot_names`: Provide the list of robot names to be used for pretraining.
  - `batch_size`: Set the dataloader batch size as large as possible. 
  - `object_pc_type`: Use `random` for major experiments  This parameter should remain the same during training and validation.

After updating the config file, simply run:

```
python train.py
```

## Validation

Run `python validate.py` simply using the test dataset and the trained model parameters.

```
python validate.py
```

## Dataset

You can download our prepared dataset here,For detailed information about the dataset, please refer to the paper.



## Steps to Apply our Method to a New Hand

1. Modify your hand's URDF. You can refer to an existing URDF file for guidance on making modifications.
2. Add the hand's URDF and mesh paths to `data/data_hand.Then, add the corresponding information in the `hand_asset.json` file.Generate the hand model from the URDF and mesh files created according to `HandModel/handmodel.py`.
3. Specify redundant  remove_links names in `hand_assert.json . You can visualize the links  to identify which links are irrelevant for contact.
4. Use `data_utils/generate_pc.py` to sample point clouds for each robot link and save them.
5. Then, simply replace the `robot_name` in `configs/dataset/dataset.yaml` with the corresponding hand model name.
