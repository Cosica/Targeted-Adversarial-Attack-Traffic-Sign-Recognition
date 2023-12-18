# Targeted-Adversarial-Attack-Traffic-Sign-Recognition
Dataset: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
Model: MobileNetV2
The dataset has 42 labels. 

In this project we will make minor changes to images to make model misclassify them to our desired label. We ensure that the difference between the new images and the original image is difficult to discern with the naked eye.

The results are shown below.

On the left-moset image is the original image and its true label is also shown. 
The other images are the result of our attack. We make minor changes to the original image and produce these new images. The target label and model prediction is shown on the top of each images. 


We also shows the perturbation we made to each images below.
![Image](https://github.com/Cosica/Targeted-Adversarial-Attack-Traffic-Sign-Recognition/blob/main/results/19.png)
![Image](https://github.com/Cosica/Targeted-Adversarial-Attack-Traffic-Sign-Recognition/blob/main/results/19-1.png)
![Image](https://github.com/Cosica/Targeted-Adversarial-Attack-Traffic-Sign-Recognition/blob/main/results/37.png)
![Image](https://github.com/Cosica/Targeted-Adversarial-Attack-Traffic-Sign-Recognition/blob/main/results/37-1.png)
