This repository is forked from https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation. 
Some parts of the code is inspired by https://github.com/Xiangyi1996/PPNet-PyTorch

# 1. Dependencies 

The dependencies to be installed can be found in the file requirements.txt. 

# 2. Data Pre-processing 

This repository only uses the abdominal MRI dataset from the original repository. 

1. Download Combined Healthy Abdominal Organ Segmentation dataset and put the /MR folder under ./data/CHAOST2/ directory

Link: https://chaos.grand-challenge.org/

2. Converting downloaded data (T2 fold) to nii files in 3D for the ease of reading

run ./data/CHAOST2/dcm_img_to_nii.sh to convert dicom images to nifti files.

run ./data/CHAOST2/png_gth_to_nii.ipynp to convert ground truth with png format to nifti.

3. Pre-processing downloaded images
run ./data/CHAOST2/image_normalize.ipynb

4. Build class-slice indexing for setting up experiments
run ./data/CHAOST2/class_slice_index_gen.ipynb

5. Pseudolabel generation
run ./data_preprocessing/pseudolabel_gen.ipynb. You might need to specify which dataset to use in cell 2 of the notebook. 

# 3. Model Training

Configure center=x in ./examples/train_ssl_abdominal_mri.sh, where x is the number of part aware prototypes. 
run ./examples/train_ssl_abdominal_mri.sh

# 4. Model Evaluation

Configure center=x in ./examples/train_ssl_abdominal_mri.sh, where x is the number of part aware prototypes.
run ./examples/test_ssl_abdominal_mri.sh
