# Vi-Fi: Associating Moving Subjects across Vision and Wireless Sensors
[Full paper](https://winlab.rutgers.edu/~hansiiii/papers/ViFi_Paper___IPSN_2022__Camera_Ready_.pdf) (accepted by IPSN'22)

[Vi-Fi Dataset](https://sites.google.com/winlab.rutgers.edu/vi-fidataset/home)

### Directory over view:
* data_collection_preprocessing/:
  * scripts of extracting metadata from ZED .svo files
  * scripts of ground truth labeling
  * scripts of data pre-processing and organizing


* Deep_Affinity_Learning/:
  * dataset_v52_better/:
    * scripts of constructing data samples for deep affinity matrix training
    * scripts of spliting training and testing set

  * v52/:
    * scripts of model architecture
    * scripts of training and testing

* Bipartite_Association/src/:
  * MATLAB scripts to prepare and process data of the visual tracker, IMU readings and FTM to associate them.


### Execution
  #### Deep Affinity Matrix Learning
  Environment & Requirements:
  ```
  Ubuntu 18.04
  CUDA 10.2
  matplotlib==3.1.1
  numpy==1.19.5
  opencv_python==4.0.1.24
  pandas==0.25.3
  Pillow==8.4.0
  pytz==2019.3
  scikit_image==0.17.2
  scikit_learn==1.0.1
  scipy==1.1.0
  skimage==0.0
  torch==1.8.1
  torchvision==0.9.1
  torchviz==0.0.1
  ```
  
  ```bash
  cd Deep_Affinity_Learning/
  pip install requirements.txt
  virtualenv -p /usr/bin/python3 [your_env_name]
  source [your_env_name]/bin/activate
  ```
  
  To train:
  ```bash
  cd dataset_v52_better/
  python split_data_train_test.py
  cd ../v52/
  python train_v50.py  [your_dir_to_save_the_model] --fold 1 --epoch 80 --dataset [your_dir_of_train_test_dataset]/train_test_shuf_split_v2/ --lr 0.001 --batchSize 32
  ```
  To test:
  ```bash
  python tracklet_ID_assignment_ml_demo.py # need to manually change the directory of .pth in the script
  ```

  #### Bipartite Association
  Environment & Requirements:
  ```
  MATLAB 2019a or later
  Navigation Toolbox
  Curve Fitting Toolbox
  ```
  
  To prepare the data:  
  * run the script: DataPreparation.m  
  * change src_path  and sequences_path to read from your directory.
  
  To associate:  
  * run ZedMain.m



