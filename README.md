# Case-Study2---Pneumothorax-detection

## 1. Business Problem

### 1.1 Description
#### Sources : 

#### a) https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data 

#### b) https://www.kaggle.com/jesperdramsch/siim-acrpneumothorax-segmentation-data

#### Problem Statement: 

We are attempting to 

a) predict the existence of pneumothorax in our test images.  

b) indicate the location and extent of the condition using masks.

### 1.2 Overview of the problem

Imagine suddenly gasping for air, helplessly breathless for no apparent reason. Could it be a collapsed lung? In the future, your entry
in this competition could predict the answer. Pneumothorax can be caused by a blunt chest injury, damage from underlying lung
disease, or most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed lung can be a life-threatening
event. Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An
accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios. AI could be used to triage chest
radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

The Society for Imaging Informatics in Medicine (SIIM) is the leading healthcare organization for those interested in the current and
future use of informatics in medical imaging. Their mission is to advance medical imaging informatics across the enterprise through
education, research, and innovation in a multi-disciplinary community.

The purpose of this competition is to identify "Pneumothorax" or a collapsed lung from chest x-rays. Pneumothorax is a condition that
is responsible for making people suddenly gasp for air, and feel helplessly breathless for no apparent reason. Pneumothorax is
visually diagnosed by radiologist, and even for a professional with years of experience; it is difficult to confirm. Neural networks and
advanced data science techniques can hopefully help capture all the latent features and detect pneumothorax consistently. So
ultimately, we want to develop a model to identify and segment pneumothorax from a set of chest radiographic images.

### 1.3 Real-world/Business objectives and constraints:

Interpretability is important.

## 2. ML/DL Problem Formulation

### 2.1 Data

#### Sources: 

#### a) https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data

#### b) https://www.kaggle.com/jesperdramsch/siim-acrpneumothorax-segmentation-data

We have dicom tain images, dicom test images, training rle.csv, sample submission.csv file

### 2.2 Mapping the real-world problem to an ML/DL problem

#### 2.2.1 Types of Deep learning problem

Classification & Segmentation Problem

#### 2.2.2. Performance Metric:

Dice coef or F1-score

## 3. Blog link
https://jayantchaudhari.medium.com/siim-acr-medical-chest-x-ray-segmentation-by-deep-learning-4bc9fdb4659

## 4. Deployment video link
https://www.youtube.com/watch?v=YQjKcl0CQD8

