# HOUSE OF HOPE

Predictive Recovery: Data-Driven Drug and Alcohol Rehab with Correlation Matrices

## ABSTRACT

This project aims to enhance the success rates of addiction and recovery treatment programs at facilities like the Green County House of Hope. Currently, only 42.1% of residents at the House of Hope complete their treatment programs. By identifying and analyzing key factors that influence completion rates, the project seeks to direct limited resources towards residents most likely to successfully complete the program.

The first objective is to develop an intake questionnaire that will screen prospective residents and predict their probability of treatment completion. This questionnaire will capture crucial data points to inform predictions and include filters to ethically evaluate residents by comparing them to similar individuals with known outcomes.

Previous models developed for the House of Hope employed various machine learning techniques, with the most accurate results coming from Random Forest models. This project aims to improve prediction accuracy by utilizing a Multi-Layer Perceptron (MLP) Deep Neural Network (DNN), which can capture more complex patterns in the data.

The Green County House of Hope also plans to expand its operations to include services for men. To support this expansion, the project will leverage the Substance Abuse and Mental Health Data Archive dataset, which includes data on male residents. By sorting the data by gender, the project will enable more accurate predictions for the current female resident population and future male residents.

## GREEN COUNTY HOUSE OF HOPE BACKGROUND

![House of Hope](/Images/house_of_hope.jpeg)

The Green County House of Hope was established in 2020 by Paul Watkins, a former jail chaplain who also served at the Green County Drug Court in Monroe, Wisconsin. Watkins founded the House of Hope with the vision of providing women a safe and stable environment for recovery from addiction, free from the triggers that might cause a relapse.

To be considered for residency, prospective residents must be at least 18 years old and have been in recovery for a minimum of 90 days. The selection process involves an interview with the House of Hope Committee, which evaluates candidates and selects women to join the program.

Residents at the House of Hope are required to participate in group therapy sessions, such as Narcotics Anonymous, and must be either employed or actively attending job training. Daily support group participation is also mandatory, fostering a supportive community atmosphere.

The House of Hope is designed to support a maximum of four women simultaneously, allowing for intensive, personalized services. Residents can stay as long as needed to achieve independence and sustain long-term sobriety.

While many drug and alcohol rehabilitation programs have success rates of just 10%, the Green County House of Hope stands out with results exceeding four times that of typical programs. This remarkable success can be attributed to its small, personalized approach, which allows for intensive, individualized support. To further enhance their effectiveness, the House of Hope is planning to incorporate a data-driven component into their selection process. This will enable them to identify and select residents who are most likely to benefit from the program, thereby optimizing their resources and improving overall outcomes.

## SAMHSA DATASET

https://www.samhsa.gov/data/data-we-collect/samhda-substance-abuse-and-mental-health-data-archive samhsa.gov Substance Abuse and Mental Health Data Archive Get Access: NSDUH Restricted-use Data SAMHSA has partnered with the National Center for Health Statistics (NCHS) to host restricted-use National Survey on Drug Use and Health (NSDUH) data at their Federal Statistical Research Data Centers (RDCs). RDCs are secure facilities that provide access to a range of restricted-use microdata for statistical purposes. SAMHSA is the most recent federal partner to work with NCHS in making NSDUH restricted-use microdata available to approved researchers at RDC sites.

License Requirements:

Privacy of Study Respondents

Any intentional identification of an individual or organization, or unauthorized disclosure of identifiable information, violates the promise of confidentiality given to the providers of the information. Disclosure of identifiable information may also be punishable under federal law. Therefore, users of data agree to: Use these data sets solely for research or statistical purposes, and not for investigation or re- identification of specific individuals or organizations. Make no use of the identity of any individual discovered inadvertently, and report any such discovery to SAMHSA (BHSIS_HelpDesk@eagletechva.com).

Public Domain Notice

All material appearing in this document is in the public domain and may be reproduced or copied without permission from SAMHSA. Citation of the source is appreciated. This publication may not be reproduced or distributed for a fee without specific, written authorization of the Office of Communications, SAMHSA, U.S. Department of Health and Human Services.

## SAMHSA DATASET CODEBOOK

- DISYR: Year of discharge
- AGE: Age at Admission
- GENDER: Gender
- RACE: Race
- ETHNIC: Ethnicity
- MARSTAT: Marital Status
- EDUC: Education
- EMPLOY: Employment status at admission
- EMPLOY_D: Employment status at discharge
- PREG: Pregnant at admission
- VET: Veteran status
- LIVARAG: Living arrangements at addmission
- LIVARAG_D: Living arrangement at discharge
- PRIMINC: Source of income/support
- ARRESTS: Arrests in past 30 days prior to admissions
- CBSA2010: CBSA 2010 code
- REGION: Census Region
- DIVISION: Census Division
- SERVICES: Type of treatment/service at admission
- SERVICES_D: Type of treatment/service at discharge
- METHUSE: Medication-assisted opiod therapy
- DATWAIT: Days waiting to enter substance treatment
- REASON: Reason for discharge
- LOS: Lenghth of stay in treatment (days)
- PSOURCE: Referral source
- DETCRIM: Detailed criminal justice referral
- NOPRIOR: Previous substance use treatment episodes
- SUB1: Substance use at admission (primary)
- SUB1_D: Substance use discharge (primary)
- ROUTE 1: Route of administration (primary)
- FREQ1: Frequency of use at admission
- FREQ1_D: Frequency of use at discharge
- FRSTUSE1: Age of first use (primary)
- SUB2: Substance use at admission (secondary)
- SUB2_D: Substance use at discharge (secondary)
- ROUTE2: Route of administration (secondary)
- FREQ2: Frequency of use at admisssion (secondary)
- FREQ2_D: Frequency of use at discharge (secondary)
- FRSTUSE2: Age of first use (secondary)
- SUB3: Substance use at admission (tertiary)
- ROUTE3: Route of administration (tertiary)
- FREQ3: Frequency of use at admission (tertiary)
- FREQ3_D: Frequency of use at discharge (tertiary)
- FRSTUSE3: Age of first use of use (tertiary)
- IDU: Current IV drug use reported at admission
- ALCFLG: Alchol reported at admission
- COKEFLG: Cocaine/crack reported at admission
- HERFLG: Non-rx methadone reported at admission
- OPSYNFLG: Other opiates/synthetics reported at admission
- PCPFLG: PCP reported at admission
- HALLFLG: Hallucinogins reported at admission
- METHAMFLG: Methamphetamines reported at admission
- STIMFLG: Other stimulants reported at admission
- TRNQFLG: Other tranquilizers reported at admission
- BARBFLG: Barbitutates reported at admission
- SEDHPFLG: Other sedativ/hypnotics reported at admission
- INHFLG: Inhalants reported at admission
- OTCFLG: Over-the-counter medications reported at admission
- OTHERFLG: Other drug use reported at admission
- ALCDRUG: Substance use type
- DSMCRIT: DSM diagnosis (SuDS4 or SuDS19)
- PSYPROB: Co-occurring mental and substance use disorders
- HLTHINS: Health insurance
- PRIMPAY: Payment source primary (expected or actual)
- FREQ_ATND_SELF_HELP: Attendance at substance use self help groups in 30 days prior to admission
- FREQ_ATND_SELF_HELP_D: Attendatce at substance use self help groups at discharge

## DATA CLEANING AND PREPARATION

All variables in the dataset were encoded as integers corresponding to specific values, as demonstrated in the example of the target column "REASON" detailed below. Missing values in the dataset were coded as -9.

- REASON: Reason for discharge
- 1 Treatment completed
- 2 Dropped out of treatment
- 3 Terminated by facility
- 4 Transferred to another treatment program or facility
- 5 Incarcerated
- 6 Death
- 7 Other

The percentage of NaN values and the correlation coefficients for each column were calculated. Based on this analysis, the dataset was reviewed with the House of Hope team to code the null values and identify columns for removal.

The columns CASE_ID, DISYR, and STFIPS were dropped as they were not strongly correlated with the data and were feared to create noise, potentially affecting model accuracy.

The columns SERVICES and SERVICES_D were also dropped because the project's goal is to create an intake questionnaire to predict a prospective resident's treatment outcome. Since all prospective clients are required to be in recovery, these columns were deemed unnecessary by the House of Hope team.

Based on the input of the House of Hope, all null values were changed to "0" with the following exceptions:

- FREQ1, FREQ2, FREQ3: 1- "No use in the past month"
- RACE: 7- "Other single race"
- EDUC: 2- "Grade 9-11"
- EMPLOY, EMPLOY_2: 2- "Unemployed"
- PREG: 2- "No"
- VET: 2- "No"
- LIVARAG: 1- "Homeless"
- PRIMINC: 4- "Other"
- PSOURCE: 1- "Individual"
- ROUTE1: 5- "Other"
- FRSTUSE1, FRSTUSE2, FRSTUSE3: 3- "15-17 years"
- SUB1, SUB2, SUB3, SUB1_D, SUB2_D, SUB3_D: 19- "Other drugs"
- DSMCRT: 5- "Opiod Dependence"
- PSYPROB: 1- "Yes"
- PRIMPAY: 1 - "Self-pay"

> Most importantly, House of Hope wanted to change the values of the variables of SUB1, SUB2, SUB3 and SUB1_D, SUB2_D, SUB3_D from the actual drugs that the perspective resident had been using and recode them as what they perceived to be "a success" and what they deemed to be "a failure". House of Hope was unconcerned if a hard drug user continued to drink or take prescription drugs and would consider those case to be successful. As such these values were changed to "1 -success"

- 2- alcohol
- 12- Other stimulants
- 13- Benzodiazepines
- 14- Other tranquilizers
- 16- Other sedatives
- 18- Over-the-counter drugs

All other values were changed "0- failure".

Similarly, the values in the target column, "REASON" were converted to pass/fail in order to predict a binary. These two values were converted to "1- pass":

- 1- Treatment completed
- 4- transfered to another facility

These values were converted to "0- fail":

- 2- Dropped out of treatment
- 3- Terminated by facility
- 5- Incarcerated
- 6- Death
- 7- Other

The target column was dropped, and the dataset was split into training and test sets with an 80/20 ratio. The training data was then scaled using StandardScaler and prepared for model training.

## SUMMARY OF INITIAL MACHINE LEARNING MODELS

Correlation coefficents were generated for all of the dimensions in the dataset with the target column generating the following results.

- MARSTAT 0.138023
- FRSTUSE1 0.139035
- VET 0.144566
- SERVICES_D 0.152773
- SERVICES 0.154441
- IDU 0.155722
- ROUTE1 0.165564
- EMPLOY 0.177870
- FREQ_ATND_SELF_HELP 0.183493
- SUB1 0.191675
- EDUC 0.193754
- LIVARAG_D 0.206890
- EMPLOY_D 0.208258
- LIVARAG 0.209302
- ARRESTS_D 0.209523
- ARRESTS 0.209631
- FREQ1_D 0.214312
- SUB1_D 0.243122
- REASON 1.000000

Intutitively, the type and frequency of the drug of choice once the person was discharged were the primary model predicters. Also, clients who had been arrested and are presumptively on a court appointed drug treatment program have a high predictive value as well as the stablity of the client's living and employment situation.

The twenty dimensions were used to train five models with the following results:

**Model: LinearRegression**
Train score: 0.1937139216941649
Test Score: 0.19216585243099527

**Model: KNeighborsRegressor**
Train score: 0.5378006263686466
Test Score: 0.29855960172606644

**Model: RandomForestRegressor**
Train score: 0.927633723215378
Test Score: 0.481623989699713

**Model: ExtraTreesRegressor**
Train score: 1.0
Test Score: 0.4840360094581453

**Model: AdaBoostRegressor**
Train score: 0.07484674836756333
Test Score: 0.07385510906138937

Following the optimization of the model, a meeting was held with House of Hope to discuss the initial correlations and design the intake questionnaire. The aim was to develop a final dataset for model training and identify the key data to be collected in the intake questionnaire. Many of the fields that were used in the final version of the model would be included in the final training of the model were variables that were collected at discharge and could not be included in the questionnaire.

A series of iterative variable drops were preformed to optimize the model performance by reducing noise from non-correlative columns. The most correlative columns were reviewed with House of Hope and a final list of fields were selected based on the information that House of Hope found to be most valuable and had the most predictive value.

The final fields used in the creation of the model and the Python-based questionnaire were: "MARSTAT," "EMPLOY," "LIVARAG," "DAYWAIT," "SERVICES," "REASON," "FRSTUSE1," "FREQ_ATND_SELF_HELP_D," "PRIMPAY," "DIVISION," "PREG," and "METHUSE."

The questionnaire was created using a Python-based widget designed to collect information for predicting a client's outcome.

## HOUSE OF HOPE DEEP NEURAL NETWORK DATA PREPARATION

In the second phase of the project, the dataset was reevaluated, and it was decided to include all the initial data fields. This was done to determine if the Multi-Layer Perceptron or the Deep Neural Network could detect subtle patterns that might improve model accuracy, previously considered as noise by the machine learning models.

Additionally, due to the high correlation of the SUB1, SUB2, SUB3, and SUB1_D, SUB2_D, SUB3_D variables, significant changes to these columns were reassessed. In the previous model, the values in these variables were changed from the specific drug causing dependence to a "1" if House of Hope deemed the outcome a success (such as alcohol abuse or over-the-counter prescription use) or a "0" if deemed a failure (such as opioid or methamphetamine use). It was concluded that these changes might not have been interpreted as intended by the model, and that the original values could have higher predictive value.

## INTAKE QUESTIONNAIRE AND ETHICAL CONSIDERATIONS OF PREDICTIVE DRUG REHABILITATION OUTCOMES

The House of Hope study also addressed the ethical considerations of using data for admission to drug rehabilitation programs, recognizing that many variables predicting successful outcomes are beyond the client's control. It is essential to determine what is fair to use in predicting a client's outcome.

Certain variables, such as race, gender, ethnicity, and age, may be legally prohibited from being used to predict outcomes, despite their demonstrated predictive significance.

Other variables, like pregnancy, are easier to evaluate ethically. Although prioritizing pregnant clients might seem to penalize men or non-pregnant women, it can be justified by the need to protect the unborn child, an innocent life affected without fault of their own.

However, some variables are not as easily justifiable. It is questionable whether it is ethical to deny a client treatment because the drug they are dependent on has a lower probability of recovery. Similarly, prioritizing the recovery of a client in a court-ordered drug program over someone motivated to recover voluntarily raises ethical concerns.

In discussions with House of Hope, it was decided that using filters might mitigate these ethical concerns. By comparing residents with similar demographic profiles rather than the entire dataset, ethical issues in prioritizing residents using data models can be addressed more effectively.

Filters were created for each of the variables in the dataset: DSMCRIT, GENDER, AGE, RACE, EDUC, VET, ARRESTS, NOPRIOR, REGION, SUB1, SUB2, ALCFLG, PSYPROB, ALCDRUG, METHAMFLG, HLTHINS, MARSTAT, EMPLOY, LIVARAG, DAYWAIT, EMPLOY, LIVARAG, DAYWAIT, SERVICES, FRSTUSE1, FREQ_ATND_SELF_HELP, PRIPAY, DIVISION, PREG, METUSE.

A filtered dataset was created for each value of each variable to investigate whether significant changes in correlation coefficients justified the use of filters. Each value produced a different list of top correlated variables and unique correlation coefficients, confirming the significance of using filters.

A Gradio-based questionnaire was developed with a series of dropdown menus, allowing an administrator to optionally select a value for each variable. The percentage of data included for each value is displayed to the administrator, enabling them to determine if there are sufficient rows left in the dataset to generate an accurate prediction.

The collected data is then added to a dictionary, which is subsequently split into the **X_train** and **X_test** datasets.

The neural network predicts "1" for success and "0" for failure.

A correlation matrix is generated from a subset of variables that correspond to actions the prospective resident could take to improve their chances of being accepted into the program. The top five correlations are formatted into a statement that lists actionable items the client can focus on to enhance their likelihood of admission.


# GRADIENT BOOSTING CLASSIFIER

Gradient Boosting Classifier (GBC) is a powerful machine learning algorithm that combines the predictions of several base estimators to improve robustness over a single estimator. This technique is part of ensemble learning, where multiple models are trained and combined to solve complex problems and improve the accuracy and performance of the model. Here's a detailed overview to help you grasp the concept better.

**Key Concepts of Gradient Boosting Classifier**

**Ensemble Learning:**

**Boosting:** A sequential process where each new model attempts to correct the errors made by the previous models.

**Base Estimators:** Weak learners, often decision trees, that are combined to form a strong predictor.

**Gradient Descent:**

**Loss Function:** Measures the difference between the predicted and actual values. Common loss functions include Mean Squared Error for regression and Logarithmic Loss for classification.

**Gradient Descent Optimization:** Minimizes the loss function by iteratively updating the model parameters in the direction that reduces the error.

**Boosting Steps:**

**Initialization:** Start with an initial model, often a simple one like the mean of the target values.

**Iteration:** At each step, fit a new model to the residuals (errors) of the current model and update the model by adding this new model.

**Combination:** Combine the models to make the final prediction. Each model’s contribution is weighted by its performance.

**Building and Training a Gradient Boosting Classifier**

**Import Libraries**

python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

**Load and Preprocess the Data**

**Load the dataset**
data = pd.read_csv('data.csv')

**Separate features and target**
X = data.drop('target', axis=1)
y = data['target']

**Train-test split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**Standardize the features**
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Step 3: Initialize and Train the Gradient Boosting Classifier


**Initialize the Gradient Boosting Classifier**

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

**Train the model**
gbc.fit(X_train, y_train)

**Evaluate the Model**

**Make predictions**

y_pred = gbc.predict(X_test)

**Evaluate the model**
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

**Explanation of the Code**

**Data Preprocessing:** The dataset is loaded and split into training and testing sets. Features are standardized using StandardScaler.

**GBC Initialization:** A GradientBoostingClassifier is initialized with 100 estimators, a learning rate of 0.1, and a maximum depth of 3.

**Training:** The model is trained on the training data using the fit method.

**Evaluation:** The model's performance is evaluated on the test data using a confusion matrix and classification report.

**Key Points to Remember**

**Hyperparameters:** Gradient Boosting has several hyperparameters (e.g., number of estimators, learning rate, max depth) that significantly affect its performance. Tuning these hyperparameters is crucial.

**Overfitting:** Gradient Boosting can overfit if the number of estimators is too high or if the trees are too deep. Techniques like cross-validation and early stopping can help prevent overfitting.

**Computational Cost:** Training Gradient Boosting models can be computationally expensive. Efficient use of resources and optimization techniques is essential.
Gradient Boosting Classifiers are powerful models for various tasks, from binary classification to multi-class classification problems. Understanding the structure and training process is crucial for effectively applying and tuning these models.

**GRADIENT BOOSTING CLASSIFIER STATISTICAL ANALYSIS**

![Gradient Boosting ROC](/Images/GMB_ROC.png)
![Gradient Boosting Training](/Images/GBM_Train.png)


## MULTI-LEVEL PERCEPTRON (MLP) MODEL

Understanding the Multilayer Perceptron (MLP) model is fundamental in neural networks and deep learning. MLP is a type of artificial neural network (ANN) that consists of multiple layers of nodes (neurons) with each layer fully connected to the next one. Here's a detailed overview to help you grasp the concept better.

**Key Concepts of MLP**

**1. Structure of MLP**
   Input Layer: The first layer that receives the input data. Each node in this layer represents a feature in the input data.
   Hidden Layers: Intermediate layers between the input and output layers. These layers perform computations and learn the representations of the input data.
   Output Layer: The final layer that produces the prediction or classification result. The number of nodes in this layer depends on the nature of the task (e.g., one node for binary classification, multiple nodes for multi-class classification).
   
**2. Activation Functions**
   Activation functions introduce non-linearity into the network, enabling it to learn complex patterns.
   Sigmoid: σ(x)=11+e−x\sigma(x) = \frac{1}{1 + e^{-x}}σ(x)=1+e−x1​
   ReLU (Rectified Linear Unit): ReLU(x)=max⁡(0,x)\text{ReLU}(x) = \max(0, x)ReLU(x)=max(0,x)
   Tanh: tanh⁡(x)=ex−e−xex+e−x\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}tanh(x)=ex+e−xex−e−x​
   
**3. Forward Propagation**
   In forward propagation, the input data is passed through the network layer by layer. Each neuron calculates a weighted sum of its inputs, applies the activation function, and passes the result to the next layer.
   
**4. Backpropagation**
   Backpropagation is the process of training the network. It involves:
   
   *Calculating the Loss:* Using a loss function (e.g., Mean Squared Error for regression, Cross-Entropy for classification) to measure the difference between the predicted and actual values.
   
   *Computing Gradients:* Determining the gradients of the loss function with respect to the weights.
   
   *Updating Weights:* Adjusting the weights using optimization algorithms like Gradient Descent to minimize the loss.
   
**9. Training the MLP**
   Training involves multiple iterations (epochs) where forward and backpropagation steps are repeated, continuously improving the model’s weights to minimize the loss.

Import Libraries
python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
Step 2: Load and Preprocess the Data
python

**Load the dataset**

data = pd.read_csv('data.csv')

**Separate features and target**

X = data.drop('target', axis=1)
y = data['target']

**Train-test split**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**Standardize the features**

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Step 3: Initialize and Train the MLP
python

**Initialize the MLPClassifier**

mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)

**Train the model**

mlp.fit(X_train, y_train)
Step 4: Evaluate the Model
python

**Make predictions**

y_pred = mlp.predict(X_test)

**Evaluate the model**

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

**Explanation of the Code**

**Data Preprocessing:** The dataset is loaded and split into training and testing sets. Features are standardized using StandardScaler.
MLP Initialization: An MLPClassifier is initialized with one hidden layer of 100 neurons, ReLU activation function, and Adam optimizer.

**Training:** The model is trained on the training data using the fit method.

**Evaluation:** The model's performance is evaluated on the test data using a confusion matrix and classification report.
Key Points to Remember

**Hyperparameters:** MLP has several hyperparameters (e.g., number of hidden layers, number of neurons, learning rate) that significantly affect its performance. Tuning these hyperparameters is crucial.

**Overfitting:** MLP can overfit, especially with a small dataset or too many neurons. Techniques like dropout and regularization can help prevent overfitting.

**Computational Cost:** Training deep networks can be computationally expensive. Efficient use of resources and optimization techniques is essential.
MLPs are powerful models for various tasks, from simple binary classification to complex image and speech recognition problems. Understanding the structure and training process is crucial for effectively applying and tuning these models.

## MLP Statistical Analysis

![MLP Accuracy](/Images/MLP_clean_df_Model.png)

![MLP Loss](/Images/MLP_clean_df_accuracy.png)

## TensorFlow/Keras model

TensorFlow and Keras are popular frameworks for building and training neural networks. Keras is a high-level API that runs on top of TensorFlow, making it easier to build and train models. Let's break down the basics of using TensorFlow and Keras to build neural networks, particularly focusing on the core concepts and providing a simple example.

**Key Concepts of TensorFlow/Keras Models**

**Layers:** The building blocks of neural networks. Common layers include Dense (fully connected), Conv2D (convolutional), LSTM (recurrent), etc.
Models: Composed of layers. In Keras, there are two main types of models:
Sequential Model: A linear stack of layers.

**Functional API:** More flexible, allows building complex models (e.g., multi-input, multi-output).

**Compilation:** Configuring the model with loss functions, optimizers, and metrics.

**Training:** Using the fit method to train the model on data.

**Evaluation:** Using the evaluate method to assess the model's performance.

**Prediction:** Using the predict method to make predictions on new data.Building and Training a Simple Neural Network with Keras

Import Libraries
python

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
Step 2: Load and Preprocess Data
python

**Load the dataset**

data = pd.read_csv('data.csv')

**Separate features and target**

X = data.drop('target', axis=1)
y = data['target']

**Train-test split**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**Standardize the features8*

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Step 3: Build the Model
python

**Initialize the Sequential model**

model = Sequential()

**Add layers**

model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # Sigmoid activation for binary classification
Step 4: Compile the Model
python

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Step 5: Train the Model
python

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
Step 6: Evaluate the Model
python

**Note:** The model initially utilize a categorical_crossentropy optimizer that was replace by a binary_crossentropy optimizer with the hypothesis that the binary optimizer would provide better accuracy; however, the binary optimizer resulted in a 20 point drop in accuracy. As a result, the final model utilizes a categorical_crossentropy optimizer.

**Evaluate on the test set**

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

**Make predictions**

y_pred = (model.predict(X_test) > 0.5).astype("int32")

**Print classification report**

print(classification_report(y_test, y_pred))

**Explanation of the Code**

**Data Preprocessing:** The dataset is loaded, features and target are separated, and data is split into training and test sets. Features are standardized using StandardScaler.

**Model Building:** A Sequential model is initialized, and layers are added. The first layer specifies input_shape, and the final layer uses a sigmoid activation function for binary classification.

**Compilation:** The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy metric.
Training: The model is trained on the training data using the fit method, with a validation split to monitor performance on validation data.

**Evaluation:** The model's performance is evaluated on the test set using the evaluate method. Predictions are made on the test set, and a classification report is printed.

**Key Points to Remember**
Epochs and Batch Size: These hyperparameters control the training process. An epoch is one complete pass through the training data, while batch size determines the number of samples processed before updating the model's weights.

**Validation Split:** Part of the training data is used for validation to monitor the model's performance on unseen data during training.
Activation Functions: Different activation functions are used for different layers. ReLU is commonly used for hidden layers, while sigmoid is used for binary classification in the output layer.

**Additional Resources**
To learn more about TensorFlow and Keras, consider exploring the following resources:
TensorFlow Documentation
Keras Documentation
Deep Learning with Python by François Chollet

## Tensorflow/Keras Statistical Results

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓

┃ Layer (type) ┃ Output Shape ┃ Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩

│ dense_6 (Dense) │ (None, 100) │ 6,900 │
├─────────────────────────────────┼────────────────────────┼───────────────┤

│ dense_7 (Dense) │ (None, 50) │ 5,050 │
├─────────────────────────────────┼────────────────────────┼───────────────┤

│ dense_8 (Dense) │ (None, 3) │ 153 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

Total params: 12,103 (47.28 KB)
Trainable params: 12,103 (47.28 KB)
Non-trainable params: 0 (0.00 B)
Epoch 1/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 167s 1ms/step - accuracy: 0.7800 - loss: 0.4443 - val_accuracy: 0.8032 - val_loss: 0.4045

Epoch 2/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 154s 1ms/step - accuracy: 0.8057 - loss: 0.4010 - val_accuracy: 0.8079 - val_loss: 0.3987

Epoch 3/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 162s 1ms/step - accuracy: 0.8094 - loss: 0.3949 - val_accuracy: 0.8094 - val_loss: 0.3949

Epoch 4/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 156s 1ms/step - accuracy: 0.8119 - loss: 0.3918 - val_accuracy: 0.8113 - val_loss: 0.3921

Epoch 5/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 158s 1ms/step - accuracy: 0.8130 - loss: 0.3902 - val_accuracy: 0.8115 - val_loss: 0.3927

Epoch 6/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 151s 1ms/step - accuracy: 0.8130 - loss: 0.3892 - val_accuracy: 0.8123 - val_loss: 0.3913

Epoch 7/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 160s 1ms/step - accuracy: 0.8140 - loss: 0.3889 - val_accuracy: 0.8130 - val_loss: 0.3898

Epoch 8/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 157s 1ms/step - accuracy: 0.8142 - loss: 0.3879 - val_accuracy: 0.8133 - val_loss: 0.3889

Epoch 9/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 160s 1ms/step - accuracy: 0.8150 - loss: 0.3869 - val_accuracy: 0.8130 - val_loss: 0.3885

Epoch 10/10
112726/112726 ━━━━━━━━━━━━━━━━━━━━ 156s 1ms/step - accuracy: 0.8152 - loss: 0.3867 - val_accuracy: 0.8143 - val_loss: 0.3894

60389/60389 ━━━━━━━━━━━━━━━━━━━━ 59s 976us/step - accuracy: 0.8149 - loss: 0.3889

**Accuracy: 0.81**

accuracy loss val_accuracy val_loss

0 0.793142 0.421548 0.803234 0.404483

1 0.806798 0.399444 0.807931 0.398664

2 0.809866 0.394463 0.809427 0.394858

3 0.811770 0.391848 0.811342 0.392066

4 0.812810 0.390394 0.811474 0.392694

![Keras Accuracy](/Images/keras_accuracy.png)

![Keras Loss](/Images/keras_loss.png)

## PyTorch Model

PyTorch is a popular framework for building and training neural networks, known for its flexibility and ease of use. Let's break down the basics of using PyTorch to build neural networks, focusing on the core concepts and providing a simple example.

**Key Concepts of PyTorch Models**

**Tensors:** The building blocks of neural networks. Similar to NumPy arrays but with added capabilities for GPU acceleration.

**Autograd:** Automatic differentiation for building and training neural networks.

**Modules:** Composed of layers. In PyTorch, models are created by subclassing torch.nn.Module and defining the layers in the __init__ method.

**Building and Training a Neural Network with PyTorch**

**Import Libraries**

python

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
Step 2: Load and Preprocess Data

python

**Load the dataset**

data = pd.read_csv('data.csv')

**Separate features and target**

X = data.drop('target', axis=1)
y = data['target']

**Train-test split**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**Standardize the features**

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

**Convert to PyTorch tensors**

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
Step 3: Build the Model

python

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SimpleNN()

**Step 4: Compile the Model**

python

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

**Step 5: Train the Model**

python
Copy code
num_epochs = 50
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    dataset=torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

**Step 6: Evaluate the Model**

python

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = (y_pred > 0.5).float()
    test_loss = criterion(y_pred, y_test)
    test_accuracy = (y_pred == y_test).float().mean()

print(f'Test Loss: {test_loss.item()}')
print(f'Test Accuracy: {test_accuracy.item()}')

print(classification_report(y_test, y_pred)

**Explanation of the Code**

**Data Preprocessing:** The dataset is loaded, features and target are separated, and data is split into training and test sets. Features are standardized using StandardScaler and converted to PyTorch tensors.

**Model Building:** A subclass of nn.Module is created, defining layers in __init__ and implementing the forward pass in forward.
Compilation: The model is compiled with the Adam optimizer and binary cross-entropy loss function (BCELoss).

**Training:** The model is trained on the training data using a data loader, with a validation split to monitor performance on validation data.

**Evaluation:** The model's performance is evaluated on the test set using the loss function and accuracy metric. Predictions are made on the test set, and a classification report is printed.

**Key Points to Remember**

**Epochs and Batch Size:** These hyperparameters control the training process. An epoch is one complete pass through the training data, while batch size determines the number of samples processed before updating the model's weights.

**Validation Split:** Part of the training data is used for validation to monitor the model's performance on unseen data during training.

**Activation Functions:** Different activation functions are used for different layers. ReLU is commonly used for hidden layers, while sigmoid is used for binary classification in the output layer.

**Additional Resources**

To learn more about PyTorch, consider exploring the following resources:

**PyTorch Documentation**

Deep Learning with PyTorch by Eli Stevens, Luca Antiga, and Thomas Viehmann

## Statistical Analysis

**Descriptive Statistics**

Descriptive statistics are used to describe the basic features of the data in a study. They provide simple summaries about the sample and the measures. Together with simple graphics analysis, they form the basis of virtually every quantitative analysis of data.

**Descriptive Analysis Values**

**Count:** The total number of observations in the dataset or for a particular variable.

**Mean:** The average value of the observations. It is calculated by summing all the values and then dividing by the count.

**Std (Standard Deviation):**: A measure of the amount of variation or dispersion in a set of values. A low standard deviation indicates that the values tend to be close to the mean, while a high standard deviation indicates that the values are spread out over a wider range.

**Min:** The minimum value in the dataset or for a particular variable. It represents the lowest observation.

**25% (First Quartile or Q1):** The value below which 25% of the observations fall. It is also known as the first quartile.

**50% (Second Quartile or Q2):** The middle value of the dataset. It is the value below which 50% of the observations fall. In a sorted dataset, it is the value that divides the dataset into two equal halves.

**75% (Third Quartile or Q3):** The value below which 75% of the observations fall. It is also known as the third quartile.

**Max:** The maximum value in the dataset or for a particular variable. It represents the highest observation.

               EDUC       MARSTAT      SERVICES           LOS       PSOURCE  \
count  6.441469e+06  6.441469e+06  6.441469e+06  6.441469e+06  6.441469e+06   
mean   2.907929e+00  1.332656e+00  5.506965e+00  2.880227e+00  3.253722e+00   
std    9.401800e-01  1.210695e+00  1.951121e+00  2.396274e+00  2.615543e+00   
min    1.000000e+00  0.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00   
25%    2.000000e+00  1.000000e+00  4.000000e+00  1.000000e+00  1.000000e+00   
50%    3.000000e+00  1.000000e+00  7.000000e+00  1.000000e+00  2.000000e+00   
75%    3.000000e+00  2.000000e+00  7.000000e+00  5.000000e+00  6.000000e+00   
max    5.000000e+00  4.000000e+00  8.000000e+00  8.000000e+00  7.000000e+00   

            NOPRIOR       ARRESTS        EMPLOY       METHUSE       PSYPROB  \
count  6.441469e+06  6.441469e+06  6.441469e+06  6.441469e+06  6.441469e+06   
mean   5.594125e-01  7.921594e-02  2.706031e+00  1.708763e+00  1.492362e+00   
std    4.964576e-01  3.081618e-01  1.277159e+00  6.115860e-01  4.999417e-01   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00   
25%    0.000000e+00  0.000000e+00  2.000000e+00  2.000000e+00  1.000000e+00   
50%    1.000000e+00  0.000000e+00  3.000000e+00  2.000000e+00  1.000000e+00   
75%    1.000000e+00  0.000000e+00  4.000000e+00  2.000000e+00  2.000000e+00   
max    1.000000e+00  2.000000e+00  4.000000e+00  2.000000e+00  2.000000e+00   

       ...       TRNQFLG       BARBFLG      SEDHPFLG        INHFLG  \
count  ...  6.441469e+06  6.441469e+06  6.441469e+06  6.441469e+06   
mean   ...  3.483677e-04  1.178924e-03  5.058939e-03  1.152843e-03   
std    ...  1.866136e-02  3.431521e-02  7.094609e-02  3.393396e-02   
min    ...  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    ...  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
50%    ...  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
75%    ...  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
max    ...  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00   

             OTCFLG      OTHERFLG      DIVISION        REGION           IDU  \
count  6.441469e+06  6.441469e+06  6.441469e+06  6.441469e+06  6.441469e+06   
mean   1.290544e-03  3.400032e-02  4.603652e+00  2.412951e+00  2.321014e-01   
std    3.590096e-02  1.812300e-01  2.594764e+00  1.148536e+00  4.221734e-01   
min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00   
25%    0.000000e+00  0.000000e+00  2.000000e+00  1.000000e+00  0.000000e+00   
50%    0.000000e+00  0.000000e+00  5.000000e+00  3.000000e+00  0.000000e+00   
75%    0.000000e+00  0.000000e+00  7.000000e+00  3.000000e+00  0.000000e+00   
max    1.000000e+00  1.000000e+00  9.000000e+00  4.000000e+00  1.000000e+00   

            ALCDRUG  
count  6.441469e+06  
mean   1.964903e+00  
std    8.377836e-01  
min    0.000000e+00  
25%    2.000000e+00  
50%    2.000000e+00  
75%    3.000000e+00  
max    3.000000e+00  

The provided box plot visualization displays the distribution of values for multiple variables in my dataset, with each box representing a variable's interquartile range (IQR). The line inside each box represents the median, while whiskers extend to the minimum and maximum values within 1.5 times the IQR from the first and third quartiles. Outliers are displayed as individual points beyond these whiskers. The plot reveals significant variability and numerous outliers in many variables, indicating diverse data distributions and the presence of extreme values. Variables such as Education (EDUC) and Marital Status (MARSTAT) show a compact distribution with relatively fewer outliers, whereas Services (SERVICES) and Length of Stay (LOS) exhibit a wider range and more outliers, suggesting varied service usage and length of stay among individuals. Several variables, such as METHUSE, PSYPROB, TRNQFLG, BARBFLG, SEDHPFLG, and INHFLG, have distributions where values are clustered at specific points (e.g., 0 and 1), indicating their binary nature. Drug use flags, like OTCFLG and OTHERFLG, generally have low means and a high frequency of zero values, highlighting the infrequency of these drug uses. Overall, the visualization provides a comprehensive overview of my dataset's structure and variability, effectively summarizing the central tendency, spread, and outliers for each variable.

Measures of Central Tendency
Central tendency measures, such as the mean, median, and mode, are used to describe the central value of a dataset. The mean is the average value of a dataset and is sensitive to extreme values, making it useful for normally distributed data. The median is the middle value of a dataset and is robust to outliers, making it suitable for skewed data. The mode is the most frequent value in a dataset and is useful for categorical data. Together, these measures provide insights into the typical value of a dataset and its distribution.

# # Measures of Central Tendency
mean = data.mean() 
median = data.median() 
mode = data.mode() 

print("\nMeasures of Central Tendency:") 
print("Mean:") 
print(mean) 
print("\nMedian:") 
print(median) 
print("\nMode:") 
print(mode) 
Measures of Central Tendency:
Mean:
EDUC        2.907929
MARSTAT     1.332656
SERVICES    5.506965
LOS         2.880227
PSOURCE     3.253722
              ...   
OTHERFLG    0.034000
DIVISION    4.603652
REGION      2.412951
IDU         0.232101
ALCDRUG     1.964903
Length: 69, dtype: float64

Median:
EDUC        3.0
MARSTAT     1.0
SERVICES    7.0
LOS         1.0
PSOURCE     2.0
           ... 
OTHERFLG    0.0
DIVISION    5.0
REGION      3.0
IDU         0.0
ALCDRUG     2.0
Length: 69, dtype: float64

Mode:
   EDUC  MARSTAT  SERVICES  LOS  PSOURCE  NOPRIOR  ARRESTS  EMPLOY  METHUSE  \
0     3        1         7    1        1        1        0       3        2   

   PSYPROB  ...  TRNQFLG  BARBFLG  SEDHPFLG  INHFLG  OTCFLG  OTHERFLG  \
0        1  ...        0        0         0       0       0         0   

   DIVISION  REGION  IDU  ALCDRUG  
0         2       1    0        2  

[1 rows x 69 columns]
# Visualize Measures of Central Tendency
plt.figure(figsize=(20, 10)) 
plt.plot(mean, label="Mean", marker="o")
plt.plot(median, label="Median", marker="o") 
plt.plot(mode.iloc[0], label="Mode", marker="o") 
plt.title("Measures of Central Tendency") 
plt.legend() 
plt.xticks(rotation=45) 
plt.show()

# Visualize Measures of Central Tendency
# plt.figure(figsize=(20, 10)) 
# plt.plot(mean, label="Mean", marker="o")
# plt.plot(median, label="Median", marker="o") 
# plt.plot(mode.iloc[0], label="Mode", marker="o") 
# plt.title("Measures of Central Tendency") 
# plt.legend() 
# plt.show() 
The provided line plot visualization illustrates the measures of central tendency (mean, median, and mode) for various variables in my dataset, showing how they compare across these variables. For many variables, the mean, median, and mode are close to each other, indicating symmetric distributions, while notable differences for some variables suggest skewness. For instance, EDUC shows a roughly symmetric distribution with the mean, median, and mode around 3, whereas SERVICES and LOS exhibit positive skewness with higher means compared to medians and modes. Binary variables like METHUSE and PSYPROB reflect their binary nature with means and medians close to 0 or 1, and modes often at 0. Discrepancies between the measures for some variables indicate significant skewness and the presence of extreme values or outliers. Overall, the visualization provides a clear overview of the central tendency measures for each variable, highlighting the symmetry, skewness, and potential outliers in the dataset.

Measures of Dispersion
Dispersion measures, such as the range, variance, standard deviation, and interquartile range (IQR), are used to describe the spread or variability of data in a dataset. The range is the difference between the maximum and minimum values, providing a simple measure of spread. The variance and standard deviation quantify the average squared deviation of data points from the mean, with the standard deviation being the square root of the variance. The IQR is the range of values within the middle 50% of the dataset, providing a robust measure of spread that is less sensitive to extreme values.

# Measures of Dispersion
std_deviation = data.std() 
variance = data.var() 
range = data.max() - data.min() 
iqr = stats.iqr(data) 

print(f"\nStandard Deviation: {std_deviation}") 
print(f"Variance: {variance}") 
print(f"Range: {range}") 
print(f"IQR: {iqr}") 
Standard Deviation: EDUC        0.940180
MARSTAT     1.210695
SERVICES    1.951121
LOS         2.396274
PSOURCE     2.615543
              ...   
OTHERFLG    0.181230
DIVISION    2.594764
REGION      1.148536
IDU         0.422173
ALCDRUG     0.837784
Length: 69, dtype: float64
Variance: EDUC        0.883939
MARSTAT     1.465782
SERVICES    3.806874
LOS         5.742130
PSOURCE     6.841064
              ...   
OTHERFLG    0.032844
DIVISION    6.732800
REGION      1.319134
IDU         0.178230
ALCDRUG     0.701881
Length: 69, dtype: float64
Range: EDUC        4
MARSTAT     4
SERVICES    7
LOS         7
PSOURCE     6
           ..
OTHERFLG    1
DIVISION    9
REGION      4
IDU         1
ALCDRUG     3
Length: 69, dtype: int64
IQR: 2.0
# Visualize Measures of Dispersion
plt.figure(figsize=(20, 10)) 
plt.plot(std_deviation, label="Standard Deviation", marker="o") 
plt.plot(variance, label="Variance", marker="o") 
plt.plot(range, label="Range", marker="o") 
plt.plot(iqr, label="IQR", marker="o") 
plt.title("Measures of Dispersion") 
plt.legend() 
plt.xticks(rotation=45)
plt.show()

# Visualize Measures of Dispersion
# plt.figure(figsize=(20, 10)) 
# plt.plot(std_deviation, label="Standard Deviation", marker="o") 
# plt.plot(variance, label="Variance", marker="o") 
# plt.plot(range, label="Range", marker="o") 
# plt.plot(iqr, label="IQR", marker="o") 
# plt.title("Measures of Dispersion") 
# plt.legend() 
# plt.show() 
The provided line plot visualization illustrates the measures of dispersion (standard deviation, variance, range, and interquartile range) for various variables in my dataset. The standard deviation and variance measures follow similar patterns, reflecting the spread of the data around the mean, with variables like SERVICES and LOS showing high values, indicating a wide spread. The range highlights the difference between the maximum and minimum values, with some variables, such as SERVICES and LOS, having high ranges, reflecting a wide spread of values. The interquartile range (IQR) varies significantly for some variables, indicating the presence of outliers and skewed distributions. Notably, some variables exhibit extreme values in their dispersion measures, such as a variance spike around 35, indicating extreme variability. Binary variables like IDU and drug use flags have low dispersion measures, as their values are confined to a small range. Overall, the visualization provides a comprehensive overview of the dataset's variability, highlighting the spread and consistency of each variable and identifying areas with significant variability or outliers.

Shape of the Distribution
The shape of a distribution refers to its overall pattern or form, such as symmetry, skewness, or modality. Symmetric distributions are mirror images around the center, with equal tails on both sides. Skewed distributions have a longer tail on one side, indicating an imbalance in the data. Bimodal distributions have two distinct peaks, while multimodal distributions have multiple peaks. Understanding the shape of a distribution is crucial for interpreting the data and selecting appropriate statistical analyses.

# Shape of the Distribution
skewness = data.skew() 
kurtosis = data.kurt() 

print(f"\nSkewness: {skewness}") 
print(f"Kurtosis: {kurtosis}")
Skewness: EDUC        0.192350
MARSTAT     1.110967
SERVICES   -0.981715
LOS         0.858889
PSOURCE     0.527860
              ...   
OTHERFLG    5.142635
DIVISION    0.306263
REGION      0.026852
IDU         1.269141
ALCDRUG    -0.627641
Length: 69, dtype: float64
Kurtosis: EDUC        -0.173969
MARSTAT      0.305134
SERVICES    -0.502578
LOS         -0.782654
PSOURCE     -1.564304
              ...    
OTHERFLG    24.446706
DIVISION    -1.179058
REGION      -1.424036
IDU         -0.389282
ALCDRUG     -0.032413
Length: 69, dtype: float64
# Visualize Shape of the Distribution
plt.figure(figsize=(20, 10)) 
plt.plot(skewness, label="Skewness", marker="o") 
plt.plot(kurtosis, label="Kurtosis", marker="o") 
plt.title("Shape of the Distribution")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visualize Shape of the Distribution
# plt.figure(figsize=(20, 10)) 
# plt.plot(skewness, label="Skewness", marker="o") 
# plt.plot(kurtosis, label="Kurtosis", marker="o") 
# plt.title("Shape of the Distribution")
# plt.legend()
# plt.show()
The provided line plot visualization illustrates the shape of the distribution for various variables in my dataset, represented by skewness and kurtosis. Most variables have skewness close to zero, indicating relatively symmetric distributions, although some, like MARSTAT and IDU, show higher skewness, suggesting asymmetry. Similarly, kurtosis values are close to zero or negative for most variables, indicating relatively flat distributions, with the notable exception of OTHERFLG, which has an extremely high kurtosis value around 3000, indicating significant outliers or a very peaked distribution. This visualization highlights the overall symmetry and peakedness of the dataset's variables, with most showing normal-like distributions, and clearly identifies variables with significant outliers or extreme values.

Frequency Distribution
Frequency of distribution refers to how often values occur in a dataset and is essential for understanding the data's patterns and characteristics. A frequency distribution table summarizes the number of occurrences of each value or range of values in a dataset, providing insights into the data's distribution and variability. Frequency distributions can be displayed using histograms, bar charts, or frequency tables, making it easier to visualize and interpret the data.

# Frequency Distribution
print("\nFrequency Distribution:")
print(data.value_counts())
Frequency Distribution:
EDUC  MARSTAT  SERVICES  LOS  PSOURCE  NOPRIOR  ARRESTS  EMPLOY  METHUSE  PSYPROB  PREG  GENDER  VET  LIVARAG  DAYWAIT  SERVICES_D  REASON  EMPLOY_D  LIVARAG_D  ARRESTS_D  DSMCRIT  AGE  RACE  ETHNIC  PRIMINC  SUB1  SUB2  SUB3  SUB1_D  SUB2_D  SUB3_D  ROUTE1  ROUTE2  ROUTE3  FREQ1  FREQ2  FREQ3  FREQ1_D  FREQ2_D  FREQ3_D  FRSTUSE1  FRSTUSE2  FRSTUSE3  HLTHINS  PRIMPAY  FREQ_ATND_SELF_HELP  FREQ_ATND_SELF_HELP_D  ALCFLG  COKEFLG  MARFLG  HERFLG  METHFLG  OPSYNFLG  PCPFLG  HALLFLG  MTHAMFLG  AMPHFLG  STIMFLG  BENZFLG  TRNQFLG  BARBFLG  SEDHPFLG  INHFLG  OTCFLG  OTHERFLG  DIVISION  REGION  IDU  ALCDRUG
2     0        7         1    1        1        0        0       2        2        2     1       2    1        0        7           1       0         1          0          5        6    5     4       4        0     1     1     0       1       1       0       0       0       1      1      1      1        1        1        3         3         3          2       4        3                    3                      0       0        0       0       0        0         0       0        0         0        0        0        0        0        0         0       0       0         8         4       0    0          521
                                       0        0        0       2        2        2     1       2    1        0        7           1       0         1          0          5        5    5     4       4        0     1     1     0       1       1       0       0       0       1      1      1      1        1        1        3         3         3          2       4        3                    3                      0       0        0       0       0        0         0       0        0         0        0        0        0        0        0         0       0       0         8         4       0    0          474
                                                                                                                                                                                     6    5     4       4        0     1     1     0       1       1       0       0       0       1      1      1      1        1        1        3         3         3          2       4        3                    3                      0       0        0       0       0        0         0       0        0         0        0        0        0        0        0         0       0       0         8         4       0    0          458
3     0        2         1    1        0        0        4       2        1        2     1       2    1        0        2           1       0         1          0          4        10   4     4       5        1     1     1     0       1       1       1       0       0       3      1      1      1        1        1        4         3         3         -9       1        3                    5                      1       0        0       0       0        0         0       0        0         0        0        0        0        0        0         0       0       0         2         1       0    1          429
2     0        7         1    1        1        0        0       2        2        2     1       2    1        0        7           1       0         1          0          5        5    5     4       4        0     1     1     0       1       1       0       0       0       1      1      1      1        1        1        3         3         3          2       4        3                    3                      0       0        0       0       0        0         0       0        0         0        0        0        0        0        0         0       0       0         8         4       0    0          428
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ... 
3     0        2         1    6        0        0        4       2        1        2     1       2    3        0        2           1       0         1          0          5        5    5     4       4        0     0     1     0       1       1       4       4       1       3      2      2      1        1        1        5         3         3         -9       1        3                    1                      0       1        0       1       0        0         0       0        0         0        0        1        0        0        0         0       0       0         2         1       1    2            1
                                                                                                                                                                                                                                                                           0       3      2      1      1        1        1        4         6         3         -9       1        3                    1                      0       0        0       1       0        0         0       0        1         0        0        0        0        0        0         0       0       0         2         1       1    2            1
                                                                                                                                                                                                                                                                   2       1       3      2      1      1        1        1        4         4         5         -9       1        3                    1                      0       1        0       1       0        0         0       0        0         0        0        1        0        0        0         0       0       0         2         1       1    2            1
                                                                                                                                                                                                                                                                           0       3      3      1      1        1        1        5         2         3         -9       1        3                    1                      0       0        1       1       0        0         0       0        0         0        0        0        0        0        0         0       0       0         2         1       1    2            1
5     4        8         8    6        1        0        1       1        2        2     2       2    3        1        8           0       1         3          0          12       9    5     4       1        0     1     1     0       1       1       4       0       0       3      1      1      1        1        1        7         3         3         -9       4        1                    4                      0       0        0       1       0        0         0       0        0         0        0        0        0        0        0         0       0       0         2         1       1    2            1
Name: count, Length: 5711258, dtype: int64
The frequency distribution table summarizes the occurrence counts for various combinations of categorical and binary variables in my dataset. Notably, the most common profiles include combinations such as EDUC=2, MARSTAT=0, SERVICES=7, LOS=1, PSOURCE=1, NOPRIOR=1, ARRESTS=0, EMPLOY=0, METHUSE=2, and PSYPROB=2, with specific variables consistently showing up together in the most frequent profiles. For instance, the combination EDUC=2, MARSTAT=0, SERVICES=7, LOS=1, PSOURCE=1, NOPRIOR=1, ARRESTS=0, EMPLOY=0, METHUSE=2, and PSYPROB=2 appears multiple times with slightly varying counts, suggesting common demographic and treatment patterns. Variables like IDU and ALCDRUG show binary distributions, typically centered around 0, indicating that non-intravenous drug use and alcohol-related issues are less frequent in the dataset. The table effectively highlights the most common profiles and the interplay between different categorical and binary variables in the dataset.

Correlation Analysis and Human Language
The blocks of code below (reason 1, reason 2, reason 3, reason 4, and reason 5) is a series of if-else statements that checks the value of the variable reason1 and assigns a corresponding reason based on the value. The reasons are then printed in human-readable language to explain the correlation between the variables. This approach provides a structured and interpretable way to analyze correlations and present the results in a clear and understandable format. By assigning specific reasons to each correlation, the analysis becomes more accessible to non-technical audiences, enabling stakeholders to grasp the relationships between variables and make informed decisions based on the findings.

## CONCLUSIONS

In our study of predicting drug addiction rehab success, we have found that neural networks excel at forecasting outcomes in drug rehabilitation programs. These models are adept at handling the myriad complex variables that influence an individual's ability to complete drug addiction recovery treatment. Factors such as personal history, socio-economic background, and psychological conditions, which traditionally complicate predictions, are effectively managed by neural networks, leading to more accurate forecasts.

Our experiments showed that neural networks configured in both TensorFlow and PyTorch returned nearly identical results. This consistency across different platforms underscores the robustness and reliability of neural network models in this context. It also suggests that the choice between TensorFlow and PyTorch can be based on user preference or specific project requirements without compromising the accuracy of the predictions.

However, despite the promising capabilities of these models, significant ethical concerns remain regarding their application. Even with the implementation of filters designed to mitigate demographic biases, questions arise about the fairness and morality of using data models to grant or deny individuals access to treatment. The risk of perpetuating existing inequalities and potentially excluding those who might benefit from the programs based on algorithmic decisions cannot be overlooked.

The results of this study offer valuable insights that can ethically guide drug recovery programs, particularly those with limited resources, in selecting individuals who have the highest chances of successful recovery. By leveraging predictive models, these programs can optimize their resource allocation, ensuring that the individuals most likely to benefit from the treatment receive the support they need.

Moreover, this study serves as a crucial resource for individuals suffering from drug addiction. By understanding the factors that contribute to successful recovery, individuals can adopt strategies that improve their chances of overcoming addiction. The predictive models and findings from this study provide a roadmap for these individuals, empowering them with knowledge and actionable insights.

However, it is essential to acknowledge the ethical considerations that come with using data models to allocate limited healthcare resources. While predictive models offer significant benefits, they also raise important questions about fairness and equity. We hope that this study sparks a meaningful conversation about the ethics of using such models, encouraging stakeholders to consider both the potential advantages and the moral implications. By addressing these concerns, we can work towards a more just and effective approach to drug addiction treatment and recovery.
