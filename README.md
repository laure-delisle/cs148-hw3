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

<img src="./images/D_performance_fraction_data.png?raw=true" width="60%" alt="loss when training on fraction of data">

From 1/16 to 1/1 of the data, the train accuracies are [87.57, 90.26, 92.26, 93.85, 94.69] and test accuracies are [92.65, 94.52, 96.0, 96.73, 97.62]. We're attributing the lower accuracies on the train set to dropout being used at training. Performance on the test set indicate a pretty decent capacity to generalize on the underlying data distribution.

**Understanding the model's performance**
For this part, we are working with the vanilla model slightly modified: the first layer is composed of 16 neurons. The test accuracy of this model is 97.90%, a slight boost from the vanilla model's performance.

* Visualizing some kernels from the first layer:

Some kernels are easily identifiable in their function: top left detects dark diagonals, 3rd rows' 3rd and 4th kernels detect black top-left corners and white bottom-left corners respectively. The bottom right corner could potentially detect top-black to bottom-white transitions. In general, kernels of the first layers pick up on very trivial (!) features like lines, edges and corners. This appears to be confirmed here.

* Confusion matrix and mistakes

<img src="./images/E_conf_matrix.png?raw=true" width="60%" alt="confusion matrix">

We observe that most example are well classified (the diagonal is dark blue and all other cells are white with mostly single digit misclassifications). Some mistakes to be noted: for instance 11 "4" digits were misclassified as "9" (4>9). We explore some of these misclassification by visual inspection (4>9, 9>4, 6>0, 3>5, 3>8, 2>7).

<p float="center">
    <img src="./images/E_misclass.png?raw=true" width="40%" alt="some mistakes"> 
    <img src="./images/E_misclass_more.png?raw=true" width="40%" alt="more mistakes">
</p>

Looking at these mistakes, we understand than overly closed curves can cause misclassifications. Indeed, a 3 with closed curves is an 8, and a 4 with a closed top curve resembles a 9. A short limb can also be the root of misclassifications: a 6 with a short arm can pass for a 0, and a 2 with a short foot can be mistaken as a 7.


