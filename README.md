# HOUSE OF HOPE
Predictive Recovery: Data-Driven Drug and Alcohol Rehab with Correlation Matrices

## ABSTRACT

This project aims to enhance the success rates of addiction and recovery treatment programs at facilities like the Green County House of Hope. Currently, only 42.1% of residents at the House of Hope complete their treatment programs. By identifying and analyzing key factors that influence completion rates, the project seeks to direct limited resources towards residents most likely to successfully complete the program.

The first objective is to develop an intake questionnaire that will screen prospective residents and predict their probability of treatment completion. This questionnaire will capture crucial data points to inform predictions and include filters to ethically evaluate residents by comparing them to similar individuals with known outcomes.

Previous models developed for the House of Hope employed various machine learning techniques, with the most accurate results coming from Random Forest models. This project aims to improve prediction accuracy by utilizing a Multi-Layer Perceptron (MLP) Deep Neural Network (DNN), which can capture more complex patterns in the data.

The Green County House of Hope also plans to expand its operations to include services for men. To support this expansion, the project will leverage the Substance Abuse and Mental Health Data Archive dataset, which includes data on male residents. By sorting the data by gender, the project will enable more accurate predictions for the current female resident population and future male residents.

## GREEN COUNTY HOUSE OF HOPE BACKGROUND

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

> DISYR: Year of discharge
> AGE: Age at Admission
> GENDER: Gender
> RACE: Race
> ETHNIC: Ethnicity
> MARSTAT: Marital Status
> EDUC: Education
> EMPLOY: Employment status at admission
> EMPLOY_D: Employment status at discharge
>  PREG: Pregnant at admission
> VET: Veteran status
> LIVARAG: Living arrangements at addmission
> LIVARAG_D: Living arrangement at discharge
> PRIMINC: Source of income/support
> ARRESTS: Arrests in past 30 days prior to admissions
> CBSA2010: CBSA 2010 code
> REGION: Census Region
> DIVISION: Census Division
> SERVICES: Type of treatment/service at admission
> SERVICES_D: Type of treatment/service at discharge
> METHUSE: Medication-assisted opiod therapy
> DATWAIT: Days waiting to enter substance treatment
> REASON: Reason for discharge
> LOS: Lenghth of stay in treatment (days)
> PSOURCE: Referral source
> DETCRIM: Detailed criminal justice referral
> NOPRIOR: Previous substance use treatment episodes
> SUB1: Substance use at admission (primary)
> SUB1_D: Substance use discharge (primary)
> ROUTE 1: Route of administration (primary)
> FREQ1:  Frequency of use at admission
> FREQ1_D: Frequency of use at discharge
> FRSTUSE1: Age of first use (primary)
> SUB2: Substance use at admission (secondary)
> SUB2_D: Substance use at discharge (secondary)
> ROUTE2: Route of administration (secondary)
> FREQ2: Frequency of use at admisssion (secondary)
> FREQ2_D: Frequency of use at discharge (secondary)
> FRSTUSE2: Age of first use (secondary)
> SUB3: Substance use at admission (tertiary)
> ROUTE3: Route of administration (tertiary)
> FREQ3: Frequency of use at admission (tertiary)
> FREQ3_D: Frequency of use at discharge (tertiary)
> FRSTUSE3: Age of first use of use (tertiary)
> IDU: Current IV drug use reported at admission
> ALCFLG: Alchol reported at admission
> COKEFLG: Cocaine/crack reported at admission
> HERFLG: Non-rx methadone reported at admission
> OPSYNFLG: Other opiates/synthetics reported at admission
> PCPFLG: PCP reported at admission
> HALLFLG: Hallucinogins reported at admission
> METHAMFLG: Methamphetamines reported at admission
> STIMFLG: Other stimulants reported at admission
> TRNQFLG: Other tranquilizers reported at admission
> BARBFLG: Barbitutates reported at admission
> SEDHPFLG: Other sedativ/hypnotics reported at admission
> INHFLG: Inhalants reported at admission
> OTCFLG: Over-the-counter medications reported at admission
> OTHERFLG: Other drug use reported at admission
> ALCDRUG: Substance use type
> DSMCRIT: DSM diagnosis (SuDS4 or SuDS19)
> PSYPROB: Co-occurring mental and substance use disorders
> HLTHINS: Health insurance
> PRIMPAY: Payment source primary (expected or actual)
> FREQ_ATND_SELF_HELP: Attendance at substance use self help groups in 30 days prior to admission
> FREQ_ATND_SELF_HELP_D: Attendatce at substance use self help groups at discharge

## DATA CLEANING AND PREPARATION

All variables in the dataset were encoded as integers corresponding to specific values, as demonstrated in the example of the target column "REASON" detailed below. Missing values in the dataset were coded as -9.

> REASON: Reason for discharge 
> 1 Treatment completed
> 2 Dropped out of treatment
> 3 Terminated by facility
> 4 Transferred to another treatment program or facility
> 5 Incarcerated
> 6 Death
> 7 Other

The percentage of NaN values and the correlation coefficients for each column were calculated. Based on this analysis, the dataset was reviewed with the House of Hope team to code the null values and identify columns for removal.

The columns CASE_ID, DISYR, and STFIPS were dropped as they were not strongly correlated with the data and were feared to create noise, potentially affecting model accuracy.

The columns SERVICES and SERVICES_D were also dropped because the project's goal is to create an intake questionnaire to predict a prospective resident's treatment outcome. Since all prospective clients are required to be in recovery, these columns were deemed unnecessary by the House of Hope team.

Based on the input of the House of Hope, all null values were changed to "0" with the following exceptions: 

> FREQ1, FREQ2, FREQ3:  1- "No use in the past month"
> RACE:  7- "Other single race"
> EDUC: 2- "Grade 9-11"
> EMPLOY, EMPLOY_2: 2- "Unemployed"
> PREG: 2- "No"
> VET: 2- "No"
> LIVARAG: 1- "Homeless"
> PRIMINC: 4- "Other"
> PSOURCE: 1- "Individual"
> ROUTE1: 5- "Other"
> FRSTUSE1, FRSTUSE2, FRSTUSE3: 3- "15-17 years"
> SUB1, SUB2, SUB3, SUB1_D, SUB2_D, SUB3_D: 19- "Other drugs"
> DSMCRT: 5- "Opiod Dependence"
> PSYPROB: 1- "Yes"
> PRIMPAY: 1 - "Self-pay"

Most importantly, House of Hope wanted to change the values of the variables of SUB1, SUB2, SUB3 and SUB1_D, SUB2_D, SUB3_D from the actual drugs that the perspective resident had been using and recode them as what they perceived to be "a success" and what they deemed to be "a failure".  House of Hope was unconcerned if a hard drug user continued to drink or take prescription drugs and would consider those case to be successful.  As such these values were changed to "1 -success"

> 2- alcohol
> 12- Other stimulants
> 13- Benzodiazepines
> 14- Other tranquilizers
> 16- Other sedatives
> 18- Over-the-counter drugs

All other values were changed "0- failure".

Similarly, the values in the target column, "REASON" were converted to pass/fail in order to predict a binary. These two values were converted to "1- pass":

> 1- Treatment completed
> 4- transfered to another facility

These values were converted to "0- fail":

> 2- Dropped out of treatment
> 3- Terminated by facility
> 5- Incarcerated
> 6- Death
> 7- Other

The target column was dropped, and the dataset was split into training and test sets with an 80/20 ratio. The training data was then scaled using StandardScaler and prepared for model training.

## SUMMARY OF INITIAL MACHINE LEARNING MODELS

Correlation coefficents were generated for all of the dimensions in the dataset with the target column generating the following results.

> MARSTAT                0.138023
> FRSTUSE1               0.139035
> VET                    0.144566
> SERVICES_D             0.152773
> SERVICES               0.154441
> IDU                    0.155722
> ROUTE1                 0.165564
> EMPLOY                 0.177870
> FREQ_ATND_SELF_HELP    0.183493
> SUB1                   0.191675
> EDUC                   0.193754
> LIVARAG_D              0.206890
> EMPLOY_D               0.208258
> LIVARAG                0.209302
> ARRESTS_D              0.209523
> ARRESTS                0.209631
> FREQ1_D                0.214312
>SUB1_D                 0.243122
> REASON                 1.000000

Intutitively, the type and frequency of the drug of choice once the person was discharged were the primary model predicters. Also, clients who had been arrested and are presumptively on a court appointed drug treatment program have a high predictive value as well as the stablity of the client's living and employment situation.

The twenty dimensions were used to train five models with the following results:

> Model: LinearRegression
> Train score: 0.1937139216941649
> Test Score: 0.19216585243099527

> Model: KNeighborsRegressor
> Train score: 0.5378006263686466
> Test Score: 0.29855960172606644

> Model: RandomForestRegressor
> Train score: 0.927633723215378
> Test Score: 0.481623989699713

> Model: ExtraTreesRegressor
> Train score: 1.0
> Test Score: 0.4840360094581453

> Model: AdaBoostRegressor
> Train score: 0.07484674836756333
> Test Score: 0.07385510906138937

Following the optimization of the model, a meeting was held with House of Hope to discuss the initial correlations and design the intake questionnaire. The aim was to develop a final dataset for model training and identify the key data to be collected in the intake questionnaire. Many of the fields that were used in the final version of the model would be included in the final training of the model were variables that were collected at discharge and could not be included in the questionnaire. Furthermore, House of Hope chose to collect demographic information such as AGE, RACE and ETHNICITY that did not show great correlation with the target variable.

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

