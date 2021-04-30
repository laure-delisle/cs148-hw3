# CS 148 - Homework 3

**Data evaluation**

Dataset: MNIST. The training set contains 60,000 images, each of them of dimensions 28x28 with 1 channel (Gray levels). We split this set into training (85%) and validation (15%). The test set contains 10,000 images. The split is done via stratified sampling to respect the class distribution:

<img src="./images/stratified_sampling.png?raw=true" width="60%" alt="stratified sampling respects class distribution">

**Baseline model**
A baseline ConvNet trained over 10 epochs achieves a validation accuracy of 97.32%. It is interesting to note that our train loss is higher than the validation loss, probably due to regularization (dropout) used during training. For more in-depth explanations, this twitter thread by Aurélien Géron is great: https://twitter.com/aureliengeron/status/1110839223878184960.

<img src="./images/losses_basic.png?raw=true" width="40%" alt="losses vanilla model"> 
<img src="./images/accuracies_basic.png?raw=true" width="40%" alt="accuracies vanilla model">

**Data augmentations**
Selected data augmentations are: slight rotations, translations, and shears.

*Rows from top to bottom: original, rotated, translated, sheared*

<img src="./images/augmentations.png?raw=true" width="60%" alt="augmented mnist images">

Augmentations considered but not selected:
- augmentations like horitzontal flips, vertical flips and drastic rotaions would potentially change the class (a "5" flipped vertically becomes a "2", a "6" heavily rotated becomes a "9"),
- crops and blurs could denature the image enough to obfuscate it, effectively hurting the model's ability to classify. They also do not represent variations found in the dataset and seem to draw examples away from the true underlying distribution.

[TODO: show results with those augmentations, performance is hurt]