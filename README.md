# HOUSE OF HOPE
Predictive Recovery: Data-Driven Drug and Alcohol Rehab with Correlation Matrices

## ABSTRACT

This project aims to enhance the success rates of addiction and recovery treatment programs at facilities like the Green County House of Hope. Currently, only 42.1% of residents at the House of Hope complete their treatment programs. By identifying and analyzing key factors that influence completion rates, the project seeks to direct limited resources towards residents most likely to successfully complete the program.

The first objective is to develop an intake questionnaire that will screen prospective residents and predict their probability of treatment completion. This questionnaire will capture crucial data points to inform predictions and include filters to ethically evaluate residents by comparing them to similar individuals with known outcomes.

Previous models developed for the House of Hope employed various machine learning techniques, with the most accurate results coming from Random Forest models. This project aims to improve prediction accuracy by utilizing a Multi-Layer Perceptron (MLP) Deep Neural Network (DNN), which can capture more complex patterns in the data.

The Green County House of Hope also plans to expand its operations to include services for men. To support this expansion, the project will leverage the Substance Abuse and Mental Health Data Archive dataset, which includes data on male residents. By sorting the data by gender, the project will enable more accurate predictions for the current female resident population and future male residents.

## GREEN COUNTY HOUSE OF HOPE BACKGROUND

![House of Hope](House_of_Hope-main/Images/house_of_hope.jpg)

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

* DISYR: Year of discharge
* AGE: Age at Admission
* GENDER: Gender
* RACE: Race
* ETHNIC: Ethnicity
* MARSTAT: Marital Status
* EDUC: Education
* EMPLOY: Employment status at admission
* EMPLOY_D: Employment status at discharge
*  PREG: Pregnant at admission
* VET: Veteran status
* LIVARAG: Living arrangements at addmission
* LIVARAG_D: Living arrangement at discharge
* PRIMINC: Source of income/support
* ARRESTS: Arrests in past 30 days prior to admissions
* CBSA2010: CBSA 2010 code
* REGION: Census Region
* DIVISION: Census Division
* SERVICES: Type of treatment/service at admission
* SERVICES_D: Type of treatment/service at discharge
* METHUSE: Medication-assisted opiod therapy
* DATWAIT: Days waiting to enter substance treatment
* REASON: Reason for discharge
* LOS: Lenghth of stay in treatment (days)
* PSOURCE: Referral source
* DETCRIM: Detailed criminal justice referral
* NOPRIOR: Previous substance use treatment episodes
* SUB1: Substance use at admission (primary)
* SUB1_D: Substance use discharge (primary)
* ROUTE 1: Route of administration (primary)
* FREQ1:  Frequency of use at admission
* FREQ1_D: Frequency of use at discharge
* FRSTUSE1: Age of first use (primary)
* SUB2: Substance use at admission (secondary)
* SUB2_D: Substance use at discharge (secondary)
* ROUTE2: Route of administration (secondary)
* FREQ2: Frequency of use at admisssion (secondary)
* FREQ2_D: Frequency of use at discharge (secondary)
* FRSTUSE2: Age of first use (secondary)
* SUB3: Substance use at admission (tertiary)
* ROUTE3: Route of administration (tertiary)
* FREQ3: Frequency of use at admission (tertiary)
* FREQ3_D: Frequency of use at discharge (tertiary)
* FRSTUSE3: Age of first use of use (tertiary)
* IDU: Current IV drug use reported at admission
* ALCFLG: Alchol reported at admission
* COKEFLG: Cocaine/crack reported at admission
* HERFLG: Non-rx methadone reported at admission
* OPSYNFLG: Other opiates/synthetics reported at admission
* PCPFLG: PCP reported at admission
* HALLFLG: Hallucinogins reported at admission
* METHAMFLG: Methamphetamines reported at admission
* STIMFLG: Other stimulants reported at admission
* TRNQFLG: Other tranquilizers reported at admission
* BARBFLG: Barbitutates reported at admission
* SEDHPFLG: Other sedativ/hypnotics reported at admission
* INHFLG: Inhalants reported at admission
* OTCFLG: Over-the-counter medications reported at admission
* OTHERFLG: Other drug use reported at admission
* ALCDRUG: Substance use type
* DSMCRIT: DSM diagnosis (SuDS4 or SuDS19)
* PSYPROB: Co-occurring mental and substance use disorders
* HLTHINS: Health insurance
* PRIMPAY: Payment source primary (expected or actual)
* FREQ_ATND_SELF_HELP: Attendance at substance use self help groups in 30 days prior to admission
* FREQ_ATND_SELF_HELP_D: Attendatce at substance use self help groups at discharge

## DATA CLEANING AND PREPARATION

All variables in the dataset were encoded as integers corresponding to specific values, as demonstrated in the example of the target column "REASON" detailed below. Missing values in the dataset were coded as -9.

* REASON: Reason for discharge 
* 1 Treatment completed
* 2 Dropped out of treatment
* 3 Terminated by facility
* 4 Transferred to another treatment program or facility
* 5 Incarcerated
* 6 Death
* 7 Other

The percentage of NaN values and the correlation coefficients for each column were calculated. Based on this analysis, the dataset was reviewed with the House of Hope team to code the null values and identify columns for removal.

The columns CASE_ID, DISYR, and STFIPS were dropped as they were not strongly correlated with the data and were feared to create noise, potentially affecting model accuracy.

The columns SERVICES and SERVICES_D were also dropped because the project's goal is to create an intake questionnaire to predict a prospective resident's treatment outcome. Since all prospective clients are required to be in recovery, these columns were deemed unnecessary by the House of Hope team.

Based on the input of the House of Hope, all null values were changed to "0" with the following exceptions: 

* FREQ1, FREQ2, FREQ3:  1- "No use in the past month"
* RACE:  7- "Other single race"
* EDUC: 2- "Grade 9-11"
* EMPLOY, EMPLOY_2: 2- "Unemployed"
* PREG: 2- "No"
* VET: 2- "No"
* LIVARAG: 1- "Homeless"
* PRIMINC: 4- "Other"
* PSOURCE: 1- "Individual"
* ROUTE1: 5- "Other"
* FRSTUSE1, FRSTUSE2, FRSTUSE3: 3- "15-17 years"
* SUB1, SUB2, SUB3, SUB1_D, SUB2_D, SUB3_D: 19- "Other drugs"
* DSMCRT: 5- "Opiod Dependence"
* PSYPROB: 1- "Yes"
* PRIMPAY: 1 - "Self-pay"

> Most importantly, House of Hope wanted to change the values of the variables of SUB1, SUB2, SUB3 and SUB1_D, SUB2_D, SUB3_D from the actual drugs that the perspective resident had been using and recode them as what they perceived to be "a success" and what they deemed to be "a failure".  House of Hope was unconcerned if a hard drug user continued to drink or take prescription drugs and would consider those case to be successful.  As such these values were changed to "1 -success"

* 2- alcohol
* 12- Other stimulants
* 13- Benzodiazepines
* 14- Other tranquilizers
* 16- Other sedatives
* 18- Over-the-counter drugs

All other values were changed "0- failure".

Similarly, the values in the target column, "REASON" were converted to pass/fail in order to predict a binary. These two values were converted to "1- pass":

* 1- Treatment completed
* 4- transfered to another facility

These values were converted to "0- fail":

* 2- Dropped out of treatment
* 3- Terminated by facility
* 5- Incarcerated
* 6- Death
* 7- Other

The target column was dropped, and the dataset was split into training and test sets with an 80/20 ratio. The training data was then scaled using StandardScaler and prepared for model training.

## SUMMARY OF INITIAL MACHINE LEARNING MODELS

Correlation coefficents were generated for all of the dimensions in the dataset with the target column generating the following results.

* MARSTAT                0.138023
* FRSTUSE1               0.139035
* VET                    0.144566
* SERVICES_D             0.152773
* SERVICES               0.154441
* IDU                    0.155722
* ROUTE1                 0.165564
* EMPLOY                 0.177870
* FREQ_ATND_SELF_HELP    0.183493
* SUB1                   0.191675
* EDUC                   0.193754
* LIVARAG_D              0.206890
* EMPLOY_D               0.208258
* LIVARAG                0.209302
* ARRESTS_D              0.209523
* ARRESTS                0.209631
* FREQ1_D                0.214312
* SUB1_D                 0.243122
* REASON                 1.000000

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

Filters were created for each of the variables in the dataset: MARSTAT", "EMPLOY", "LIVARAG", "DAYWAIT", "SERVICES", "FRSTUSE1", "FREQ_ATND_SELF_HELP_D", "PRIMPAY", "DIVISION", "PREG", "METHUSE".  A filtered data set was created with selection of values from each variable to investigate whether there was a significant change in the correlation coeffients in the new data set to justify the use of filters.

## MULTI-LEVEL PERCEPTRON (MLP) MODEL

Understanding the Multilayer Perceptron (MLP) model is fundamental in neural networks and deep learning. MLP is a type of artificial neural network (ANN) that consists of multiple layers of nodes (neurons) with each layer fully connected to the next one. Here's a detailed overview to help you grasp the concept better.
Key Concepts of MLP
1. Structure of MLP
Input Layer: The first layer that receives the input data. Each node in this layer represents a feature in the input data.
Hidden Layers: Intermediate layers between the input and output layers. These layers perform computations and learn the representations of the input data.
Output Layer: The final layer that produces the prediction or classification result. The number of nodes in this layer depends on the nature of the task (e.g., one node for binary classification, multiple nodes for multi-class classification).
2. Activation Functions
Activation functions introduce non-linearity into the network, enabling it to learn complex patterns.
Sigmoid: σ(x)=11+e−x\sigma(x) = \frac{1}{1 + e^{-x}}σ(x)=1+e−x1​
ReLU (Rectified Linear Unit): ReLU(x)=max⁡(0,x)\text{ReLU}(x) = \max(0, x)ReLU(x)=max(0,x)
Tanh: tanh⁡(x)=ex−e−xex+e−x\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}tanh(x)=ex+e−xex−e−x​
3. Forward Propagation
In forward propagation, the input data is passed through the network layer by layer. Each neuron calculates a weighted sum of its inputs, applies the activation function, and passes the result to the next layer.
4. Backpropagation
Backpropagation is the process of training the network. It involves:
Calculating the Loss: Using a loss function (e.g., Mean Squared Error for regression, Cross-Entropy for classification) to measure the difference between the predicted and actual values.
Computing Gradients: Determining the gradients of the loss function with respect to the weights.
Updating Weights: Adjusting the weights using optimization algorithms like Gradient Descent to minimize the loss.
5. Training the MLP
Training involves multiple iterations (epochs) where forward and backpropagation steps are repeated, continuously improving the model’s weights to minimize the loss.

Step 1: Import Libraries
python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
Step 2: Load and Preprocess the Data
python

# Load the dataset
data = pd.read_csv('data.csv')

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Step 3: Initialize and Train the MLP
python

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)

# Train the model
mlp.fit(X_train, y_train)
Step 4: Evaluate the Model
python

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
Explanation of the Code
Data Preprocessing: The dataset is loaded and split into training and testing sets. Features are standardized using StandardScaler.
MLP Initialization: An MLPClassifier is initialized with one hidden layer of 100 neurons, ReLU activation function, and Adam optimizer.
Training: The model is trained on the training data using the fit method.
Evaluation: The model's performance is evaluated on the test data using a confusion matrix and classification report.
Key Points to Remember
Hyperparameters: MLP has several hyperparameters (e.g., number of hidden layers, number of neurons, learning rate) that significantly affect its performance. Tuning these hyperparameters is crucial.
Overfitting: MLP can overfit, especially with a small dataset or too many neurons. Techniques like dropout and regularization can help prevent overfitting.
Computational Cost: Training deep networks can be computationally expensive. Efficient use of resources and optimization techniques is essential.
MLPs are powerful models for various tasks, from simple binary classification to complex image and speech recognition problems. Understanding the structure and training process is crucial for effectively applying and tuning these models.

# MLP Statistical Analysis

Accuracy: 0.81
              precision    recall  f1-score   support

           0       0.77      0.68      0.72    708990
           1       0.83      0.88      0.85   1223451

    accuracy                           0.81   1932441
   macro avg       0.80      0.78      0.79   1932441
weighted avg       0.81      0.81      0.81   1932441

## TensorFlow/Keras model

ensorFlow and Keras are popular frameworks for building and training neural networks. Keras is a high-level API that runs on top of TensorFlow, making it easier to build and train models. Let's break down the basics of using TensorFlow and Keras to build neural networks, particularly focusing on the core concepts and providing a simple example.
Key Concepts of TensorFlow/Keras Models
Layers: The building blocks of neural networks. Common layers include Dense (fully connected), Conv2D (convolutional), LSTM (recurrent), etc.
Models: Composed of layers. In Keras, there are two main types of models:
Sequential Model: A linear stack of layers.
Functional API: More flexible, allows building complex models (e.g., multi-input, multi-output).
Compilation: Configuring the model with loss functions, optimizers, and metrics.
Training: Using the fit method to train the model on data.
Evaluation: Using the evaluate method to assess the model's performance.
Prediction: Using the predict method to make predictions on new data.Building and Training a Simple Neural Network with Keras

Step 1: Import Libraries
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

# Example: Load the dataset
data = pd.read_csv('data.csv')

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Step 3: Build the Model
python

# Initialize the Sequential model
model = Sequential()

# Add layers
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
Step 4: Compile the Model
python

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Step 5: Train the Model
python

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
Step 6: Evaluate the Model
python


Note: The model initially utilize a categorical_crossentropy optimizer that was replace by a binary_crossentropy optimizer with the hypothesis that the binary optimizer would provide better accuracy; however, the binary optimizer resulted in a 20 point drop in accuracy.  As a result, the final model utilizes a categorical_crossentropy optimizer.

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Print classification report
print(classification_report(y_test, y_pred))
Explanation of the Code
Data Preprocessing: The dataset is loaded, features and target are separated, and data is split into training and test sets. Features are standardized using StandardScaler.
Model Building: A Sequential model is initialized, and layers are added. The first layer specifies input_shape, and the final layer uses a sigmoid activation function for binary classification.
Compilation: The model is compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy metric.
Training: The model is trained on the training data using the fit method, with a validation split to monitor performance on validation data.
Evaluation: The model's performance is evaluated on the test set using the evaluate method. Predictions are made on the test set, and a classification report is printed.
Key Points to Remember
Epochs and Batch Size: These hyperparameters control the training process. An epoch is one complete pass through the training data, while batch size determines the number of samples processed before updating the model's weights.
Validation Split: Part of the training data is used for validation to monitor the model's performance on unseen data during training.
Activation Functions: Different activation functions are used for different layers. ReLU is commonly used for hidden layers, while sigmoid is used for binary classification in the output layer.
Additional Resources
To learn more about TensorFlow and Keras, consider exploring the following resources:
TensorFlow Documentation
Keras Documentation
Deep Learning with Python by François Chollet

# Tensorflow/Keras Statistical Results

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_6 (Dense)                 │ (None, 100)            │         6,900 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_7 (Dense)                 │ (None, 50)             │         5,050 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_8 (Dense)                 │ (None, 3)              │           153 │
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

Accuracy: 0.81
   accuracy      loss  val_accuracy  val_loss
0  0.793142  0.421548      0.803234  0.404483
1  0.806798  0.399444      0.807931  0.398664
2  0.809866  0.394463      0.809427  0.394858
3  0.811770  0.391848      0.811342  0.392066
4  0.812810  0.390394      0.811474  0.392694

![Keras Accuracy](House_of_Hope-main/Images/keras_accuracy.png)

![Keras Loss](House_of_Hope-main/Images/keras_loss.jpg)

## GRADIENT BOOST MACHINE MODEL