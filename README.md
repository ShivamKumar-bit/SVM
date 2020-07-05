# Support Vector Machines
SVM's responsibility is to find a decision boundary to separate  different classes and maximize the margin.
![Image](https://miro.medium.com/max/1000/1*Ox4UFUKHna9BjW5gfNcQlw.png)
#### Note-
1. Boundary can be a referred as a line in 2D space, a plane in 3D space and hyperplane in other dimensions.
2. Margins are the (perpendicular) distances between the line and those dots closest to the line.
3. The closest points to your hyperplane is called **support vectors**.
![Image](https://miro.medium.com/max/1400/1*R0KTZUoPgsY0NBP2d3LFsw.png)
## SVM in the linear separable cases,you need to ensure 2 things:
- Ensure that each observation is on the correct side of the Hyperplane.
- Pick up the optimal line so that the distance from those closest dots to the Hyperplane, so-called margin, is maximized.

## SVM in non linear separable cases
In reality datasets are rarely linearly separable.So it's not possible to get 100% accuracy in your classification.
![Image](https://miro.medium.com/max/1000/1*gt_dkcA5p0ZTHjIpq1qnLQ.png)

In such cases regularization parameter , gamma and kernel plays a vital role.These are tuning parameters in SVM classifier. Varying those we can achive considerable non linear classification line with more accuracy in reasonable amount of time.
#### Kernel Trick
What kernel Trick does is,it utilizes existing features, applies some transformations, and creates new features. Those new features are the key for SVM to find the nonlinear decision boundary.In Sklearn — svm.SVC(), we can choose ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable as our kernel/transformation.

![Image](https://miro.medium.com/max/1400/1*Ha7EfcfB5mY2RIKsXaTRkA.png)

#### Regularization
Often termed as C parameter in python’s sklearn library tells the SVM optimization how much you want to avoid misclassifying each training example.
For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.
![Image](https://miro.medium.com/max/1000/1*1dwut8cWQ-39POHV48tv4w.png)  ![Image](https://miro.medium.com/max/1000/1*gt_dkcA5p0ZTHjIpq1qnLQ.png)  
low vs High regularization

#### Gamma
In simple words,with low gamma, points far away from plausible seperation line are considered in calculation for the seperation line.
Where as high gamma means the points close to plausible line are considered in calculation.
![Image](https://miro.medium.com/max/1200/1*dGDQxV8j83VB90skHsXktw.png)
![Image](https://miro.medium.com/max/1200/1*ClmsnU_yb1YtIwAAr7krmg.png)

### Let's code-

```ruby
import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()
```
We have taken iris data set from sklearn.
```ruby
print(dir(digits))
```
To get keys , just like a dictionary
``` ruby
print(digits.target) 
```
Gives your target values i.e 0,1,2....9
```ruby
print(digits.target_names) 
```
Gives names of your targets i.e 0,1,2....9
```ruby
print(digits.images) 
```
Gives you nd array 
```ruby
df = pd.DataFrame(digits.data,digits.target) 
df.head()
```
We created data frame with digits.data 
``` ruby
df['target']=digits.target 
df.head(20)
```
Appending a target column
``` ruby
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.2)
```
Using test_train_split to divide data for training and testing the model
``` ruby
from sklearn.svm import SVC
model = SVC(C=1.0,kernel='rbf') ## or model =just SVC()
```
Importing SVC classifier and setting parameters
```ruby
model.fit(X_train, y_train)
```
Training model
```ruby
model.score(X_test,y_test)
```
Checking accuracy

The entire code is available on [Github](https://github.com/ShivamKumar-bit/SVM/blob/master/SVM.ipynb)




