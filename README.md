# Developing a Neural Network Classification Model

### Name:  THARUN DP

### Register Number:212225240172

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load and Preprocess the Dataset:

Import the required libraries, load the dataset, handle missing values, encode categorical variables, and normalize the features.

### STEP 2: 

Split the Dataset:

Divide the dataset into training and testing sets.

### STEP 3: 

Build the Neural Network:

Create a Sequential model with an input layer, hidden layers using ReLU activation, and an output layer with Softmax activation for the four customer segments.

### STEP 4: 

Train the Model:

Compile the model using the Adam optimizer and categorical cross-entropy loss function, then train it using the training data.

### STEP 5: 

Evaluate and Predict:

Evaluate the model on the test dataset, measure its accuracy, and use it to predict the customer segment (A, B, C, or D) for new customers.







## PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
```
df=pd.read_csv("C:\\Users\\varal\\Downloads\\Deep Learning\\Deep Learning\\customers.csv")
df
```
<img width="1046" height="376" alt="image" src="https://github.com/user-attachments/assets/707aa7ae-a8a1-4919-964e-c6112660d57b" />

```
df=df.drop(columns=["ID"])
df
```

<img width="991" height="372" alt="image" src="https://github.com/user-attachments/assets/cd6cb4bd-d4ae-41d0-8679-81dd35526e6d" />

```
df.columns
```
<img width="627" height="121" alt="image" src="https://github.com/user-attachments/assets/212730a0-30ed-438d-865e-7dc31494cf3a" />

```
df.columns
```
<img width="573" height="106" alt="image" src="https://github.com/user-attachments/assets/e7f941ec-ed3b-4e25-b28d-a802af71080b" />

```
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
df
```
<img width="1042" height="367" alt="image" src="https://github.com/user-attachments/assets/3ab2ccf2-57d1-47fc-81ae-e20b72307835" />

```
cat_columns=['Gender','Ever_Married', 'Graduated', 'Profession',
    'Spending_Score', 'Var_1']
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])
```

```
lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])
df
```
<img width="983" height="357" alt="image" src="https://github.com/user-attachments/assets/8078285e-8222-43e3-b4d8-3f7ff3eef46c" />


```
x=df.drop(columns="Segmentation")
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)
```

```
scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)
```

```
xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)
```

```
tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)
```

```
class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,8)
        self.l4=nn.Linear(8,4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x
```

```
model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
```

```
epochs=100
for i in range(epochs):
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b.long())
        loss.backward()
        op.step()
    if i%10==0:
        print(f"Loss:{i}/{epochs}",loss.item())
```
<img width="285" height="208" alt="image" src="https://github.com/user-attachments/assets/cbce01a1-61a7-414b-a846-dc14e7bbf990" />

```
pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    act.extend(yst.numpy())
    print(act,pre)
```
<img width="446" height="348" alt="image" src="https://github.com/user-attachments/assets/edda8fab-d2e5-4c76-b946-be3810ebfe24" />


```
accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=['A','B','C','D'])
print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)
print("classification_report:\n",cl_report)
```
<img width="452" height="351" alt="image" src="https://github.com/user-attachments/assets/8cab3a35-f839-4217-85a5-b0c45a205a27" />


```
import seaborn as sns
import matplotlib.pyplot as plt
xl=['A','B','C','D']
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=xl, yticklabels=xl,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```

<img width="583" height="483" alt="image" src="https://github.com/user-attachments/assets/a46344ce-49ae-4934-85ee-7c6a2251c80b" />



### Dataset Information
<img width="992" height="372" alt="image" src="https://github.com/user-attachments/assets/897e84a1-4e1b-4702-a53f-38767faf623a" />


### OUTPUT

## Confusion Matrix
<img width="583" height="483" alt="image" src="https://github.com/user-attachments/assets/a46344ce-49ae-4934-85ee-7c6a2251c80b" />

## Classification Report
<img width="482" height="237" alt="image" src="https://github.com/user-attachments/assets/39c8205d-d4f2-4d52-88d4-683b2b8f0022" />


## RESULT
This program has been executed successfully.
