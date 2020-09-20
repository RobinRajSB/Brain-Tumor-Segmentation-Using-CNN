
# Brain-Tumor-Segmentation-using-Deep-Neural-Networks

This project is made for as a proof of concept for augument the treatment of brain tumer, this is done for one of my friend, for her 🙎‍♀️ final year project.

Feel free to ask any questions/doubts 😀 

visit: www.robinrajsb.com

I have implemented this CNN using Keras 👌

Final code will be inside Notebook folder, its is names according to their uses
FinalCode.ipynb is the Final code to be executed


For accessing the dataset, you need to create account with  https://www.smir.ch/BRATS/Start2013. For free access to GPU,   refer to this Google Colab tutorial https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d 

You are free to use contents of this repo for academic and non-commercial purposes only.



### Resources
https://arxiv.org/pdf/1505.03540.pdf (this is sound and complete paper, refer to this and it's references for all questions)

### Overview 🧑‍💻
Paper poses the pixel-wise segmentation problem as classification problem. The model takes a patch around the central pixel and labels from the five categories, as defined by the dataset -

Necrosis
Edema
Non-enhancing tumor
Enhancing tumor
Everything else This way, the model goes over the entire image producing labels pixel-by-pixel.


### BRATS Dataset 🧳


I have used BRATS 2013 training dataset for the analysis of the proposed methodology. It consists of real patient images as well as synthetic images created by SMIR. Each of these folders are then subdivided into High Grade and Low Grade images. For each patient, four modalities(T1, T1-C, T2 and FLAIR) are provided. The fifth image has ground truth labels for each pixel. The dimensions of image is different in LG and HG. For HG, the dimensions are (176,261,160) and for LG are (176,196,216).



## Dataset pre-processing 
As per the requirement of the algorithm, slices with the four modalities as channels are created. For taking slices of 3D modality image, I have used 2nd dimension. At time of training/ testing, we need to generate patches centered on pixel which we would classifying. We are ignoring the border pixels of images and taking only inside pixels. Generating a dataset per slice. I am filtering out blank slices and patches. Also, slices with all non-tumor pixels are ignored.

## Model Architecture 
### TwoPathCNN
 It shows the 2 paths input patch has to go through. 1st path where 2 convolutional layers are used is the local path. The 1st convolutional layer is of size (7,7) and 2nd one is of size (3,3). Global path consist of (21,21) filter. As the local path has smaller kernel, it processes finer details because of small neighbourhood. Opposed to this, global path process in more global way. After the convolutional layer, Max-Out [Goodfellow et.al] is used. After which max-pooling is used with stride 1. I have changed the max-pooling to convolution with same dimensions. In the global path, after convolution max-out is carried out. THere is no max-pooling in the global path.After activation are generated from both paths, they are concatenated and final convolution is carried out. Then Softmax activation is applied to the output activations. Because there is no fully-connected layers in model, substantial decrease in number of parameters as well as speed-up in computation.   

![](Capture.PNG)

### Cascading Architectures
Cascading architectures uses TwoPathCNN models joined at various positions. The paper defines 3 of them - 
- InputCascadeCNN: 1st’s output joined to 2nd’s input
- LocalCascadeCNN: 1st’s output joined to 2nd’s hidden layer(local path 2nd conv input)
- MFCcascadeCNN: 1st’s output joined to 2nd’s concatenation of two paths 

![](Capture1.PNG)

## Training
### Loss function
As per the paper,Loss function is defined as ‘Categorical cross-entropy’ summed over all pixels of a slice. I have modified the loss function in 2-ways:
- The dataset per slice is being directly fed for training with mini-batch gradient descent i.e., I am calculating and back-propagating loss for much smaller number of patches than whole slice.
- For each dataset, I am calculating weights per category, resulting into weighted-loss function. This is taken as measure to skewed dataset, as number of non-tumor pixels mostly constitutes dataset. 
   
### Regularization 
The paper uses drop-out for regularization. Instead, I have used Batch-normalization,which is used for regularization also. In this paper, authors have shown that batch-norm helps training because it smoothens the optimization plane. Which helps in stable gradients and faster reaching optima. When training without regularization and weighted-loss function, I found out that model gets stuck at local optima, such that it always predicts ‘non-tumor’ label. After adding these 2, I found out increase in performance of the model.
 
## Performance
As the dataset is very large because of patch-per-pixel-wise training scheme, I am not able to train the models on all of the dataset. For now, both cascading models have been trained on 4 HG images and tested on a sample slice from new brain image. As mentioned in paper, I have computed f-measure for complete tumor region.
##### Complete score: labels 1+2+3+4 for patients data.
```


Slice Number                 F1-Score (complete) 
(HG 0027)              InputCascadeCNN  MFCcascadeCNN
105                             0.9250  0.80091
106                             0.9271  0.8029
107                             0.9269  0.8085
108                             0.9280  0.8114
109                            0.92691  0.8056
110                             0.9277  0.7986
111                             0.9291  0.7929
112                             0.9297  0.7868
113                             0.9273  0.79228
```
Thanks 🙏 to https://github.com/jadevaibhav/Brain-Tumor-Segmentation-using-Deep-Neural-networks
