# Instruction of the project from team YEAH

## the name of the final report:
- E4040.2020Fall.YEAH.report.jy3024.jn2722.sg3772.pdf

## the name of the main jupyter notebook:
There are 13 main jupyter notebooks named started by main for running each model. The name of the jupyter notebook is clearly stating the model that could be built in that jupyter notebook file. For instance, the main_ResidualAttention56_CIFAR10.ipynb file could be used to train and test the Attention-56 model using attention residual learning algorithm. There is also a Project_Intro.ipynb file giving the introduction of each of these 13 jupyter notebooks and the outputs we obtained as well. In the last part of Project_Intro.ipynb, 5 tables summarizing all the outputs of 13 models are stated.

The name of the 13 main jupyter notebooks are:
  - main_ImageNet.ipynb
  - main_NaiveAttention56_CIFAR10.ipynb
  - main_NaiveAttention92_CIFAR10.ipynb
  - main_ResidualAttention56_Channel_CIFAR10.ipynb
  - main_ResidualAttention56_CIFAR10.ipynb
  - main_ResidualAttention56_CIFAR100.ipynb
  - main_ResidualAttention56_Spatial_CIFAR10.ipynb
  - main_ResidualAttention92_CIFAR10.ipynb
  - main_ResidualAttention92_CIFAR10_Noise10.ipynb
  - main_ResidualAttention92_CIFAR10_Noise30.ipynb
  - main_ResidualAttention92_CIFAR10_Noise50.ipynb
  - main_ResidualAttention92_CIFAR10_Noise70.ipynb
  - main_ResidualAttention92_CIFAR100.ipynb

## instructions how to run the code:
Each model could be trained and evaluated by rerunning the code in the main jupyter notebooks with corresponding name. 

## description of key functions of each file in the project:   
  - processing: data preprocess procedure of zero-padding and random-crop for CIFAR-10 and CIFAR-100 dataset.
  - add_noise: data preprocess procedure of adding noise into training labels.
  - NAL_stage_1, NAL_stage_2, NAL_stage_3: three attention modules used in the naive attention learning model.
  - ResNet56_NAL, ResNet92_NAL: the naive attention learning 56 model and naive attention learning 92 model for CIFAR dataset using mixed attention. 
  - attention_stage_1, attention_stage_2, attention_stage_3: three attention modules used in the attention residual learning model using mixed attention.
  - channel_attention_stage_1, channel_attention_stage_2, channel_attention_stage_3: three attention modules used in the attention residual learning model using channel attention.
  - spatial_attention_stage_1, spatial_attention_stage_2, spatial_attention_stage_3: three attention modules used in the attention residual learning model using spatial attention.
  - AttentionResNet56, AttentionResNet92: the attention residual learning 56 model and attention residual learning 92 model for CIFAR dataset using mixed attention.
  - AttentionResNet56_spatial, AttentionResNet56_channel: the attention residual learning 56 model for CIFAR-10 dataset using spatial and channel attention.
  - residual_unit: Pre-activation Residual Unit with projection shortcut and Pre-activation Residual Unit with identity shortcut that would be used to build up model for CIFAR-10 and CIFAR-100 dataset.
  - AttentionResNet56_ImageNet: Create the AttentionResNet56 model for ImageNet.
  - Load_ImageNet: Load pictures into numpy array.
  - residula_unit_ImageNet:  Pre-activation Residual Unit with projection shortcut,  Pre-activation Residual Unit with identity shortcut  
  - predict_10_crop: Do 10 crop testing.

## location  of the datasets:
  - CIFAR-10: dataset is loaded using tensorflow.keras.datasets.cifar10.load_data()
  - CIFAR-100: dataset is loaded using tensorflow.keras.datasets.cifar100.load_data()
  - Tiny ImageNet 200 : http://cs231n.stanford.edu/tiny-imagenet-200.zip
## organization of directories and any other supporting information. 
All the thirteen main jupyter notebooks are under the same folder as this file. These 13 jupyter notebooks could be rerun for each model. Project_Intro.ipynb file is also under this folder, with introduction of each model with the values and figures derived and a summary of them. Project_Intro.pdf is the pdf format of Project_Intro.ipynb file. The report is also under this folder, which is the other pdf file with name of E4040.2020Fall.YEAH.report.jy3024.jn2722.sg3772.pdf. All the functions that are used to establish the different attention models are under the folder of utils. The weights of each model trained by us are saved as h5 file under save_models folder. All the plots of loss and accuracy of each model are under the images folder with names of them corresponding to the model built. Some referenced papers are listed under Reference folder. The folder of figures contains the screenshots to prove that the project has been done in the cloud. 

# e4040-2020Fall-project
Seed repo for projects for e4040-2020Fall-project
  - distributed as Github Repo and shared via Github Classroom
  - contains only README.md file
  - Students must have at least one main Jupyter Notebook, and a number of python files in a number of directories and subdirectories such as utils or similar, as demonstrated in the assignments
  - The organization of the directories has to be meaningful

# Detailed instructions how to submit this assignment/homework/project:
1. The assignment will be distributed as a github classroom assignment - as a special repository accessed through a link
2. A students copy of the assignment gets created automatically with a special name - students have to rename the repo per instructions below
3. The solution(s) to the assignment have to be submitted inside that repository as a set of "solved" Jupyter Notebooks, and several modified python files which reside in directories/subdirectories
4. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud

## (Re)naming of a project repository shared by multiple students (TODO students)
INSTRUCTIONS for naming the students' solution repository for assignments with more students, such as the final project. Students need to use a 4-letter groupID): 
* Template: e4040-2020Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2020Fall-Project-MEME-zz9999-aa9999-aa0000.

# Organization of this directory
To be populated by students, as shown in previous assignments

```
.
│   .gitignore
│   E4040.2020Fall.YEAH.report.jy3024.jn2722.sg3772.pdf
│   main_ImageNet.ipynb
│   main_NaiveAttention56_CIFAR10_sg.ipynb
│   main_NaiveAttention92_CIFAR10_jn.ipynb
│   main_ResidualAttention56_Channel_CIFAR10_sg.ipynb
│   main_ResidualAttention56_CIFAR100_sg.ipynb
│   main_ResidualAttention56_CIFAR10_sg.ipynb
│   main_ResidualAttention56_Spatial_CIFAR10_sg.ipynb
│   main_ResidualAttention92_CIFAR100_jn.ipynb
│   main_ResidualAttention92_CIFAR10_jn.ipynb
│   main_ResidualAttention92_CIFAR10_Noise10_jn.ipynb
│   main_ResidualAttention92_CIFAR10_Noise30_jn.ipynb
│   main_ResidualAttention92_CIFAR10_Noise50_jn.ipynb
│   main_ResidualAttention92_CIFAR10_Noise70_jn.ipynb
│   Project_Intro.ipynb
│   Project_Intro.pdf
│   README.md
│   
├───.github
│       .keep
│       
├───images
│       NaiveAttention56_CIFAR10.png
│       NaiveAttention56_CIFAR100.png
│       NaiveAttention92_CIFAR10.png
│       ResidualAttention56_Channel_CIFAR10.png
│       ResidualAttention56_CIFAR10.png
│       ResidualAttention56_Spatial_CIFAR10.png
│       ResidualAttention92_CIFAR10.png
│       ResidualAttention92_CIFAR100.png
│       ResidualAttention92_CIFAR10_Noise10.png
│       ResidualAttention92_CIFAR10_Noise30.png
│       ResidualAttention92_CIFAR10_Noise50.png
│       ResidualAttention92_CIFAR10_Noise70.png
│       
├───Reference
│       Attention-56-deploy.prototxt
│       Attention-92-deploy.prototxt
│       Data Augmentation using Random Image Cropping and Patching for Deep CNNs.pdf
│       Delving Deep into Rectifiers-Surpassing Human-Level Performance on ImageNet Classification.pdf
│       Residual Attention Network for Image Classification.pdf
│       scale and aspect ratio augmentation.PNG
│       Standard color augmentation.PNG
│       test time augmentation1.PNG
│       test time augmentation2.PNG
│       weight initialization.png
│       [10] Deep residual learning for image recognition.pdf
│       [20] ImageNet classification with deep CNN.pdf
│       
├───save_models
│       NaiveAttention56_CIFAR10_weights.h5
│       NaiveAttention92_CIFAR10_weights.h5
│       ResidualAttention56_channel_CIFAR10_weights.h5
│       ResidualAttention56_CIFAR100_weights.h5
│       ResidualAttention56_CIFAR10_weights.h5
│       ResidualAttention56_ImageNet.h5
│       ResidualAttention56_spatial_CIFAR10_weights.h5
│       ResidualAttention92_CIFAR100_weights.h5
│       ResidualAttention92_CIFAR10_weights.h5
│       ResidualAttention92_Noise10_CIFAR10_weights.h5
│       ResidualAttention92_Noise30_CIFAR10_weights.h5
│       ResidualAttention92_Noise50_CIFAR10_weights.h5
│       ResidualAttention92_Noise70_CIFAR10_weights.h5
│       
└───utils
        AttentionResNet_CIFAR.py
        AttentionResNet_CIFAR100.py
        AttentionResNet_ImageNet.py
        attention_block.py
        Load_ImageNet.py
        NAL_block.py
        NAL_CIFAR.py
        predict_10_crop.py
        preprocess.py
        residual_unit_CIFAR.py
        residual_unit_ImageNet.py
        

```
