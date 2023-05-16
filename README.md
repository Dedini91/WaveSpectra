<h1 align="center">Fully Convolutional Autoencoder for Wave Spectra Prediction</h1>

---

<p align="center">
   <img src="assets/Offshore_example.jpg" alt="Raw offshore spectra" style="height: 200px; width:200px;"/>
   <img src="assets/Offshore_proc.jpg" alt="Processed offshore spectra" style="height: 200px; width:200px;"/>
   <img src="assets/NearShore_proc.jpg" alt="Processed near shore spectra" style="height: 200px; width:200px;"/>
   <img src="assets/NearShore_example.jpg" alt="Raw near shore spectra" style="height: 200px; width:200px;"/>
</p>

---

WaveSpectra is a 55-layer Fully Convolutional Neural Network (FCNN) designed to perform unsupervised image-to-image regression on offshore and corresponding near shore wave energy density spectrograms. Once trained, the network is capable of accurately predicting near shore wave conditions from offshore data. 

<p align="center">
   <img src="assets/training.gif" alt="Tracked training sample" style="height: 480px; width:640px;"/>
</p>

> If running on Colab or similar, follow the included **get_data** and **Walkthrough** notebooks

## Usage:
### **Download repository & install requirements**
   ```python
    git clone https://github.com/Dedini91/WaveSpectra.git
    pip install -r requirements.txt
   ```

Example usage:
```python
python make_balanced_dataset.py -d path/to/data_folder -n new_dataset_test -c 50 --split 80 20 10
```
```python
python train.py -n exp_name --verbose --cache -d path/to/folder/containing/npz_files -b 1 -e 30 --lr 0.00001 --track 05902 --outputs --device cuda
python train.py -n resume_exp --verbose --cache -d path/to/folder/containing/npz_files --model_path path/to/model/last.pth --track 05902 --outputs --device cuda --resume
```
```python
python evaluate.py --model_path path/to/best_model.pth -d path/to/folder/containing/npz_files --verbose --device cuda
```
```python
python predict.py -d data/x_test.npz --model_path path/to/best_model.pth
```

For each script, there are only a few required arguments. 

> Execute `script_name.py --help` to print the complete list of supported arguments.

---

## 1. **Preprocess raw data and make dataset**
> Use "get_data.ipynb" to retrieve and save raw data in compressed .npz format

* **make_balanced_dataset.py**
   1. Perform PCA, and perform constrained KMeans clustering
   2. Sample clusters equally to partition data into train/validation/test sets
 
```python
# Required arguments:           Type-Default        Description
    -d      --data              str-None            'path to parent directory of raw data folder'
    -n      --name              str-None            'dataset name'
    -s                          int-2000            'number of training samples (max 80% of total # samples)'
    -c                          int-10              'number of clusters (classes)'
```

---

## 2. **Training**
* **train.py**
   1. Training and validation of neural network
   2. Basic evaluation on test data
   3. Runs are timestamped; all outputs are saved to ***results/exp_name/datetime/***

* The model will default to using gpu where available, unless `--device cpu` is passed.

```python
# Required arguments:           Type-Default        Description
    -d      --data              str-None            'path to processed dataset'
    -n      --name              str-None            'experiment name'
```
* Each run is logged to ***./results/exp_name/datetime/***
* Model & optimiser saved to ***./results/exp_name/datetime/model/***
* Train/Validation predictions saved to ***./results/exp_name/datetime/predictions/***

#### **Logging with tensorboard**
* Basic metrics are automatically logged to ***./results/exp_name/datetime/logs/***
* View in tensorboard by running the following in a separate terminal:
```python
tensorboard --logdir="path/to/logs_folder/"
```

<p align="center">
   <img src="assets/losses.jpg" alt="Train_val_losses" style="height: 480px; width:640px;"/>
</p>

---

## 3. **Evaluation**
* **evaluate.py**
   1. Performs more in-depth evaluation on any dataset with corresponding ground truths
   2. Generates error maps to identify difficult examples
   3. Creates evaluation folder within parent run directory

Evaluates the performance of a trained model on previously unseen data (test set), giving an idea of how well it generalises.
* Pass source images located in `--img_path "./path/to/image_folder"` and paired target images `--target_path "./path/to/targets_folder"` to evaluate model located at `--model_path "./results/exp_name/datetime/model/best_model.pth"`.

```python
# Required arguments:           Type-Default        Description
            --model_path        str-None            'path to model and optimiser state_dict.pth files'
            --img_path          str-None            'path to folder/image for evaluation and inference'
            --target_path       str-None            'path to target images for evaluation and inference'
```
Numerical results are saved in .csv format ordered by lowest error (for the selected loss function). These can be loaded into Excel or Python for inspection.

---

## 4. **Inference**
* **predict.py**
   1. Perform inference on images using a trained model

```python
# Required arguments:           Type-Default        Description
            --model_path        str-None            'path to model and optimiser state_dict.pth files'
            --img_path          str-None            'path to folder/image for evaluation and inference'
```

---

# TODO

* Add support for:
   * Entering custom image sizes
   * Resuming training from checkpoint
   * Address hard-coded clustering parameters
