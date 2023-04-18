<h1 align="center">Convolutional Autoencoder for Wave Spectra Prediction</h1>

---

<p align="center">
   <img src="assets/Offshore_example.jpg" alt="Raw offshore spectra" style="height: 200px; width:200px;"/>
   <img src="assets/Offshore_proc.jpg" alt="Processed offshore spectra" style="height: 200px; width:200px;"/>
   <img src="assets/NearShore_proc.jpg" alt="Processed near shore spectra" style="height: 200px; width:200px;"/>
   <img src="assets/NearShore_example.jpg" alt="Raw near shore spectra" style="height: 200px; width:200px;"/>
</p>

---

This is a Pytorch implementation of an unsupervised image-to-image convolutional autoencoder, capable of accurately predicting near shore wave spectra from corresponding offshore spectra in the form of 64*64 spectrograms.

The network passes normalised offshore (source) and near shore (target) spectra through a 9-layer convolutional autoencoder, which generates an image as its output using a sigmoid activation in the final layer. The l1 (MAE), l2 (MSE), and Huber losses are used to compare the network's output the with known target spectrograms. 

Given a .NETCDF with raw offshore and corresponding near shore data:
1. Extract spectrograms
2. Preprocess images and create dataset
3. Train and validate a deep neural network
4. Evaluate network on unseen data/target pairs
5. Perform inference (generate prediction spectrograms) on source images

> If using Google Colab, simply run the included scripts from the Walkthrough.ipynb notebook.

Arguments are passed when running the script from the command line (or using `!python` in Colab)
* For booleans that are set to `False` by default, simply pass the flag with no argument to set to `True`

---

## 1. **Download the repository & install requirements**
   ```
    git clone https://github.com/Dedini91/WaveSpectra_CAE.git
    pip install -r requirements.txt
   ```

### 1.5. Get raw data from .NETCDF files
> Use ***"get_data.ipynb"*** to retrieve data and save spectrograms
  * Several thousand pairs of images is more than enough data, but use as much as time permits
  * Best to take small chunks of samples from throughout the ~85,000 total, for dataset diversity. 
* Download these from Colab if training locally; place in the parent folder of your project

---

## 2. **Preprocess raw data and make dataset**
> **Ensure that raw images are located in ***./data*****

  * Resize images to 64*64 pixels; greyscale; normalised to [0 1]
  * Randomly splits data into train/validation/test datasets (0.8x0.8/0.2x0.8/0.2)
  * Calculates mean and std deviation for training data
     * Run the preprocessing script to obtain the following file structure:

```
├── data/
│   ├── raw/*.jpg
│   ├── interim/*.jpg
│   └── processed/
│       ├── x_train/*.jpg           # Training data:
│       │   ├── 00000.jpg           # 80% of (80% of total samples) 
│       │   ├── 00001.jpg
│       │   ├──    ...
│       │   └── 0000n.jpg
│       ├── y_train/*
│       ├── x_val/*                 # Validation data:
│       ├── y_val/*                 # Remaining 20% of (80% of total samples)
│       ├── x_test/*                # Test/evaluation data:
│       ├── y_test/*                # 20% of total samples
│       └── red_.../*               # Prototyping folders (red_x_train... etc.)
├── CAE.py
├── ReadMe.md
├── requirements.txt
├── raw_data.NETCDF
│   ...
└──
```
* Images ready for training are located in the ***./data/processed/*** directory
  * For prototyping purposes, manually create copies of these subfolders with reduced # samples, prepending "red_" to the directory name
    * e.g. ***red_x_train/***, ***red_y_train/***, etc. 

Supported arguments: **make_dataset.py**
```python
Short Flag  Long flag           Type-Default        Description
# Required arguments:
    -d      --data              str-None            'path to parent directory of raw data folder'
```
Example:
```python
python make_dataset.py --data path/to/data_folder
```

---

## 3. **Training**

Example:
```python
python train.py -n exp_name -d "data/processed/" -b 1 -e 10 -o adamw -l l1 --reduction sum --lr 0.00003 --prototyping --verbose
or in Colab:
!python /content/gdrive/MyDrive/WaveSpectra_CAE/train.py ... ... ...
```
* The model will default to using gpu where available, unless the `--device cpu` argument is explicitly passed.

> See `python train.py --help` for a list of supported arguments and available options.

Supported arguments: **train.py**
```python
                                Type-Default        Description
# Required arguments:
    -d      --data              str-None            'path to processed dataset'
    -n      --name              str-None            'experiment name'
# Optional arguments:
            --prototyping       bool-False          'prototyping mode (train on reduced dataset)'
    -v      --verbose           bool-False          'verbose output (recommended)'
            --model_path        str-None            'path to saved model.pth file'
# Trainer arguments:
            --device            str-'cuda'          'device'
    -i      --interval          int-5               'logging interval (epochs)'
# Hyperparameters:
    --lr    --learning_rate     float-0.00005       'initial (maximum) learning rate'
            --lr_schedule       bool-True           'learning rate scheduler'
            --l1_min            float-0.00001       'minimum learning rate'
    -o      --optimizer         str-'adam'          'optimizer'
    -l      --loss_function     str-'l1'            'loss function'
            --reduction         str-'sum'           'reduction method'
    -e      --num_epochs        int-100             'number of epochs'
    -b      --batch_size        int-10              'batch size'
    -m      --momentum          float-0.9           'momentum for SGD'
# Regularisation:
            --decay             float-0.0           'weight decay rate (default off) - use with SGD or AdamW'
```
* Each experiment is logged to ***./results/exp_name/datetime/***
* Model & optimiser saved to ***./results/exp_name/datetime/model/***
* Train/Validation predictions saved to ***./results/exp_name/datetime/predictions/***

#### **Logging with tensorboard**
* Basic metrics are automatically logged to ***./results/exp_name/datetime/logs/***
* View in tensorboard: Run the following in a separate terminal on your local machine (download log folder from Colab to use):
```python
tensorboard --logdir="path/to/logs_folder/"
```
#### **Outputs**
* Folder structure for experiment
* Checkpointed model and optimizer state dicts; best performing model
* Saved logs of training and validation losses, sorted by lowest l1 loss
* Saved logs of CLI output, including argument configuration
* Prediction images for training, validation, and basic evaluation on test set
* Basic evaluation results, sorted by lowest l1 loss

<p align="center">
   <img src="assets/losses.jpg" alt="Example loss plot" style="height: 300px; width;"/>
</p>

---

## 4. **Evaluation**
Evaluates the performance of a trained model on previously unseen data (test set), giving an idea of how well it generalises.
* Pass source images located in `--img_path "./path/to/image_folder"` and paired target images `--target_path "./path/to/targets_folder"` to evaluate model located at `--model_path "./results/exp_name/datetime/model/best_model.pth"`.
```python
python evaluate.py --model_path "path/to/best_model.pth" --img_path "./path/to/image_folder" --target_path "./path/to/targets_folder" --verbose
```

> See `python evaluate.py --help` for a list of supported arguments and available options.

Supported arguments: **evaluate.py**
```python
                                Type-Default        Description
# Required arguments:
            --model_path        str-None            'path to model and optimiser state_dict.pth files'
            --img_path          str-None            'path to folder/image for evaluation and inference'
            --target_path       str-None            'path to target images for evaluation and inference'
# Optional arguments:
            --device            str-'cuda'          'prototyping mode (train on reduced dataset)'
    -v      --verbose           bool-False          'verbose output (recommended)'
            --errmaps           bool-True           'turns off error maps in output files (may speed up evaluation)'
    -l      --loss              str-'l1'            'loss function'
            --reduction         str-'sum'           'reduction method'
```
Evaluation produces single and comparison images for each sample in the test set, with error maps turned on by default. Error maps display the per-pixel absolute error as calculated by various loss functions. 
> Turning error maps off by passing `--errmaps` may improve training times. 

<p align='center'>
   <img src="assets/error_map.jpg" alt="Error map example" style="height: 400;"/>
</p>

Numerical results are also saved in .csv format ordered by lowest l1 error. These can be loaded into Excel or Python for further inspection, for identification of challenging stimuli, etc. 

Example:
   | Image ID | L1/MAE (sum) | L1/MAE (mean) | L2/MSE (sum) | L2/MSE (mean) | Huber (sum) | Huber (mean) |
   | :-- | :-: | :-: | :-: | :-: | :-: | :-: |
   | 12532.jpg | 6.345151 | 0.001549 | 0.380528 | 0.000093 | 0.190264 | 0.000046 |
   | 18572.jpg | 6.740136 | 0.001646 | 0.127139 | 0.000031 | 0.063569 | 0.000016 |
   | 17399.jpg | 7.206785 | 0.001759 | 0.357544 | 0.000087 | 0.178772 | 0.000044 |
   | 11254.jpg | 7.340772 | 0.001792 | 0.430695 | 0.000105 | 0.215348 | 0.000053 |
   | 18130.jpg | 7.443274 | 0.001817 | 0.525027 | 0.000128 | 0.262514 | 0.000064 |

---

## 5. **Inference**
Similarly, to perform inference (generate predictions) on a folder of images - *without corresponding targets*:
```python
python predict.py --model_path "path/to/best_model.pth" --img_path "./path/to/image_folder"
```

> See `python inference.py --help` for a list of supported arguments and available options.

Supported arguments: **predict.py**
```python
                                Type-Default        Description
# Required arguments:
            --model_path        str-None            'path to model and optimiser state_dict.pth files'
            --img_path          str-None            'path to folder/image for evaluation and inference'
# Optional arguments:
            --device            str-'cuda'          'prototyping mode (train on reduced dataset)'
```

---

# TODO

* Add support for:
   * Entering custom image sizes
   * Resuming training from checkpoint/transfer learning
   * Implement SSIM as both a metric and loss function

- Isolate n best and n worst examples during evaluation

- Upload more sophisticated architectures

- Test on larger datasets (10k, 20k, ..., 85k source-target pairs)

- Upload very small example dataset
