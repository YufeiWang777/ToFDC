# RGB+ToF Depth Completion MIPI-Challenge - NPU-CVR

This repo contains the code that can be employed to reproduce our results. 
Because the training code needs extra time to sort out, we only provide the test code. 
Should you have any questions, please contact us without hesitation.
The correspondence mailbox is wangyufei1951@gmail.com.

## How to use

### Installation
```bash
# Environment.
conda env create -f environment.yaml
conda activate ToFDC
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install Deformable Convolution V2 (DCNv2) follow https://github.com/zzangjinsun/NLSPN_ECCV20
cd model/deformconv
sh make.sh

# Install Nearest Neighbors Interpolation (NNI) follow https://github.com/Shiaoming/DensefromRGBS
cd ../..
python setup.py build_ext --inplace
```

### Results
Download the [Results](https://drive.google.com/file/d/1lpkn6LOCb9Q3nhgqd_On8Jd_dVKge7YQ/view?usp=sharing).

### Pretrained Model
Download the [Pretrained Model](https://drive.google.com/file/d/17klXTEzi-wztxqylehhAj7FhkHmIuy-b/view?usp=sharing), 
and unzip it.

### Validate
> 1. Specifying the test dataset path by `--data_folder`, for example `./test_input`.
> 2.  Moving the pretrained model to the `pretrained_model` folder in the project or specifying the path of the pretrained model by `--test_model`.
> 3.  specifying the path of ouput images by `--test_dir` (Defaults to the project directory), and the output results will be stored in the `output\depth_pred` folder under this.
> 4. We provide two test methods: using and not using the self-ensemble strategy.
```bash
# using the self-ensemble strategy 
python val.py -d data_folder -m test_model -o test_dir -c val_with_self_ensemble.yml

# not using the self-ensemble strategy 
python val.py -d data_folder -m test_model -o test_dir -c val_without_self_ensemble.yml
```

