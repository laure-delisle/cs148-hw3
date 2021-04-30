# CS 148 - Homework 3

**Data evaluation**

Dataset: MNIST. The training set contains 60,000 images, each of them of dimensions 28x28 with 1 channel (Gray levels). We split this set into training (85% i.e. 51,000) and validation (15% i.e 9,000). The test set contains 10,000 images. The split is done via stratified sampling to respect the class distribution:

<img src="./images/stratified_sampling.png?raw=true" width="60%" alt="stratified sampling respects class distribution">

**Baseline model**
A baseline ConvNet trained over 10 epochs achieves a validation accuracy of 97.32%. It is interesting to note that our train loss is higher than the validation loss, probably due to regularization (dropout) used during training. For more in-depth explanations, this twitter thread by Aurélien Géron is great: https://twitter.com/aureliengeron/status/1110839223878184960.

<p float="center">
    <img src="./images/losses_basic.png?raw=true" width="40%" alt="losses vanilla model"> 
    <img src="./images/accuracies_basic.png?raw=true" width="40%" alt="accuracies vanilla model">
</p>

**Data augmentations**
Selected data augmentations are: slight rotations, translations, and shears.

*Rows from top to bottom: original, rotated, translated, sheared*

<img src="./images/augmentations.png?raw=true" width="60%" alt="augmented mnist images">

Augmentations considered but not selected:
- augmentations like horitzontal flips, vertical flips and drastic rotaions would potentially change the class (a "5" flipped vertically becomes a "2", a "6" heavily rotated becomes a "9"),
- crops and blurs could denature the image enough to obfuscate it, effectively hurting the model's ability to classify. They also do not represent variations found in the dataset and seem to draw examples away from the true underlying distribution.

<p float="center">
    <img src="./images/B_accuracies_train.png?raw=true" width="40%" alt="accuracy training with augmentation"> 
    <img src="./images/B_accuracies_val.png?raw=true" width="40%" alt="accuracy validation with augmentation">
</p>

Data augmentation hurts the training performance, as expected: it makes training harder for the model by presenting a wider variety of examples, applying different modifications to the training images at each epoch. More surprisingly, data augmentation seems to be nefast for generalization purposes as well. The validation accuracy is highest when no augmentation is applied. Digging more into this, we observe that pure translation is the augmentation scheme yielding the least decrease in accuracy at validation. CNNs being translation invariant by definition, we suspect that translations applied as augmentation may have caused the digit to be partly translated out of the 28x28 frame.

Another hypothesis supplementing above considerations is that the training set is already quite large (51,000 images), the images are quite small (28x28) yielding a small number of features, and the target is a limited number of classes. This should provide enough variety already, not requiring any data augmentation. In this specific case, data augmentation seems to be creating more issues by deforming the images too much, and providing no benefit as the training data is large.

**Improving on the Vanilla Model**
Nothing was done here, however we augmented the capacity of the filter layer from 8 to 16 filters in order to visualize at least 9 further on.

**Performance on the test set**
With our model, the accuracy on the train and test sets are 94.69% and 97.62% respectively. When trained on fractions of the training data, the model performance decreases exponentially. We observe a linear trend on a log-log plot, indicating a fat-tail (or Pareto distribution) behavior. This indicates that adding training data is exponentially beneficial for our model. This however seems at odd with the data augmentation behavior explaiend earlier, any feedback here is welcome.

<img src="./images/D_performance_fraction_data?raw=true" width="60%" alt="loss when training on fraction of data">

