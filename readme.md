# Logo Detection of Document using Siamese Network for Classification and Yolov3 for Detection

## Problems

The dataset consist of 2000 classes of Labels and the job is to classify the logo in a given document (There is only one image per class)

## Approach

### Using Yolov3 for detection of Logos in a document

I first automatically create yolo dataset for each of the logo on a couple sample document and train them on yolov3

I used yolov3 to crop out the logo in a document and send it to my classifying model (Siamese Network)

### Using Siamese Network to Classify the logos

#### Structure of the Siamese model

#### Prediction

I used the cropped logo from the Yolov3 to my Siamese model and compare it with each of the image in my dataset to return a probability that if these two image are look-alike. With that we have the probability of every logo in the dataset. Those with the highest probability are most likely to be the logo. Due to constrain in dataset, the model can return only top 10 of most probability, to increase accuracy more training and more data is needed
To speed up prediction time even further, please check out: https://medium.com/@kuzuryu71/improving-siamese-network-performance-f7c2371bdc1e


## How to run

### Create and activate environment

```
conda create --logo python==3.6.9
conda activate logo
```

### Install Requirements

```
pip install -r requirements.txt
```

### Run Detection
Run Example:

```
python classify.py --image_path Images/3_plus.png
```
