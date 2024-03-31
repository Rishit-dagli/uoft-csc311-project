# CSC311 Final Project

We give instructions on how to run the code for Part B of the project. The code resides in the `part_b` directory. Currently, our code is tested with Python 3.10.x, hiwever we believe it should work well with Python 3.9.x and above.

Assuming you have cloned this repository, and you are in the `part_b` directory of the repository, you can run the following command to setup the environment:

```bash
pip install -r requirements.txt
```

## Training the model

To train the model, you can use the `main.py` script. The script can be used as follows:

```bash
python main.py [OPTIONS]
```

The following options are available:

- `--project`
  - Type: `str`
  - Default: `"test"`
  - Description: Name of the project.

- `--custom_loss`
  - Action: `store_true`
  - Description: Use custom loss function.

- `--model_type`
  - Type: `str`
  - Default: `"original"`
  - Choices: `["original", "mhsa"]`
  - Description: Model architecture type: 'original' or 'mhsa'.

- `--optimizer`
  - Type: `str`
  - Default: `"adam"`
  - Description: Optimizer to use.

- `--lr_scheduler`
  - Type: `str`
  - Default: `"none"`
  - Choices: `["none", "steplr", "cosineannealing", "reducelronplateau", "cosinedecay"]`
  - Description: Type of LR scheduler: steplr, cosineannealing, reducelronplateau, cosinedecay, none.

- `--step_size`
  - Type: `int`
  - Default: `10`
  - Description: Step size for StepLR and T_max for CosineAnnealing.

- `--gamma`
  - Type: `float`
  - Default: `0.1`
  - Description: Gamma for StepLR, ReduceLROnPlateau, and factor for CosineAnnealing.

- `--patience`
  - Type: `int`
  - Default: `10`
  - Description: Patience for ReduceLROnPlateau.

- `--lr`
  - Type: `float`
  - Default: `1e-3`
  - Description: Initial learning rate.

- `--base_path`
  - Type: `str`
  - Default: `"../../data"`
  - Description: Base path for the dataset.

- `--gpu`
  - Action: `store_true`
  - Description: Use GPU for training.

- `--num_devices`
  - Type: `int`
  - Default: `1`
  - Description: Number of devices to use for training.

- `--epochs`
  - Type: `int`
  - Default: `10`
  - Description: Number of epochs.

- `--batch_size`
  - Type: `int`
  - Default: `32`
  - Description: Batch size.

- `--save_model`
  - Type: `str`
  - Default: `"model.pth"`
  - Description: Path to save the model.

- `--checkpoint_path`
  - Type: `str`
  - Default: `None`
  - Description: Path to a checkpoint to resume training from.

- `--seed`
  - Type: `int`
  - Default: `3047`
  - Description: Random seed.

### Run our model configuration

To run our best performing model configuration for each of the two model architectures we try out (Embeddings and Multi-Head Self-Attention), you can use the following commands:

```bash
# Original model
python main.py --project education_model --model_type original  --optimizer adamw --lr_scheduler none --lr 1e-1 --base_path ../data --epochs 8 --batch_size 64 --save_model model.pth --seed 3047

# Multi-Head Self-Attention model
python main.py --project education_model --model_type mhsa --optimizer adamw --lr_scheduler reducelronplateau --patience 15 --lr 1e-1 --base_path ../data --epochs 250 --batch_size 32 --save_model model.pth --seed 3047
```

You should run each of these commands with `--num_devices [NUMBER_OF_DEVICES]` if you want to use more than 1 device for the training and `--gpu` if you want to use GPU for training.

### Viewing the training logs

You can view the training logs using TensorBoard. To do this, you can run the following command:

```bash
tensorboard --logdir=logs
```

This will start a TensorBoard server that you can access by visiting `http://localhost:6006` in your browser.

## Evaluating the model

To evaluate the model on thew entire test set, the script can be used as follows:

```bash
python main.py --model_type original --base_path ../../data --batch_size 3543 --epochs 0 --checkpoint_path model.pth

python main.py --model_type mhsa --base_path ../../data --batch_size 3543 --epochs 0 --checkpoint_path model.pth
```