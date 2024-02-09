# ChangeLog

## Changes

### Project Goals
<p> The goal of the notebook being followed is to give an overview of the Support Vector Machine machine learning algorithm. One task this notebook completes which inspired the current project is the classification of a number of images of famous people into the correct person in the image.</p>
<br>
![Categories](Images/dataexploration.png)

<br>
<g> The goal of this project has been changed to the classification of a number of images of people. The classes to be identified are not wearing glasses, wearing glasses or wearing sunglasses. This project will attempt to use Support Vector Machine machine learning techniques to make these classifications based on the input images.

### Data Source
<p> The source notebook retrieves the data it uses for its predictions from included datasets within the sklearn platform. This project instead sources its data from kaggle using twos datasets. Two datasets are used to boost the amount of data available to try and increase prediction acccuracy. The two datasets used are as follows:

1. **Glasses and Covering**
    - **Description**: This dataset was designed for the training of facial recognition. This contains a total of 2537 images of faces where the images are aligned and cropped around the face. This dataset contains 5 classes, plain (no glasses), glasses (with glasses), sunglasses (with sunglasses), sunglasses-imagenet (additional sunglasses) and coverings( faces with various coverings eg. masks). The image class is defined by the directory it is in within the dataset.
    - **Usage**: As this project is attempting to find only 3 of the classes contained in the dataset, only plain, glasses, sunglasses and sunglasses-imagenet will be used.
    - **Source Quality**: This dataset was sourced from Kaggle and has a high usability score from Kaggle users.
    - **Link**: [here](https://www.kaggle.com/datasets/mantasu/glasses-and-coverings)

2. **People with and without glasses dataset**
     - **Description** This dataset contains 4920 color images split into two folders glaases and  and without_mask to denote their class. These classes are images where a face is not wearing glasses and is wearing glasses.
     - **Source Quality**: The data was sourced from Kaggle and has a low usability score. This means that it will have to be checked in preprocessing to ensure it's data quality.
     - **Link**: [here](https://www.kaggle.com/datasets/saramhai/people-with-and-without-glasses-dataset)

### Data Preprocessing
<p> As this project is using a different dataset, some data exploration and preprocessing is required to make sure the data is clean and ready for modelling. The reference notebook uses native sklearn functions to perform data exploration on it's image set as seen below. </p>
![Data exploration](Images/dataexploration.png)
<p> In this project, custom data exploration is involved. First a function was created to gather metadata on the contents of each data seperately as below. This function counted the size of each class in terms of images, validated that each file had a picture suffix such as .jpg and counted any files that did not have the correct format.</p>

```
from collections import defaultdict

#Create function to explore datasets
def getfolderinfo(FOLDER_PATH,folders):
    # Define a default dict to store folder properties
    folder_info = defaultdict(dict)
    for folder in folders:
        path = os.path.join(FOLDER_PATH, folder)
        length = len(os.listdir(path))
        folder_info[folder]['length'] = length
        suffix_count = defaultdict(int)
        not_image = 0
        # Check files in folder
        for img in os.listdir(path):
            # Check suffix
            suffix = img.split('.')[-1]
            # Define these suffix as valid image files
            if suffix.lower() in ['jpg', 'jpeg', 'png', 'gif']:
                suffix_count[suffix] += 1
            else:
                not_image += 1

            # Save folder info            
            folder_info[folder]['suffix_frequency'] = suffix_count
            folder_info[folder]['not_image'] = not_image

    return folder_info
```
### Dataset - Glasses and Covering
<p> Function was used to analyse contents of dataset. The results below show that the dataset is balanced in terms of numbers of images so there is not imbalance in the data which might cause bias.</p>

![alt text](Images/ds1_counts.png)

<p><p> All files in folders are images with most files being jpg in this dataset with small exception. This should cause no problems going forward.</p>

![alt text](Images/filetypes.png)

#### Checking for null values
<p> Some checks are made to ensure that the data is clean and there are no null values in the dataset. This check showed that there are none </p>
```
df[df.isnull().any(axis=1)].count()
```

#### Dropping unneeded columns
<p> The dataset included three commas at the end of each row which created 3 unneeded columns once parsed. This columns were dropped. </p>

```
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
```
### Model Creation

#### Feature Preparation
<p> This project prepares the data for modelling in a different way given the data source and the fact this is a binary classfication algorithm with a single feature variable. The correct variables are placed in X and y arrays and reshape is used to flatten both arrays.</p>

#### Target Variable Encoding
<p> This project uses a label encoder to encode categorical target variable. O for ham and 1 for spam </p>

```
from sklearn.preprocessing import LabelEncoder
labels = ['ham','spam']
lab_encoder = LabelEncoder()
y = lab_encoder.fit_transform(y)
```

### Model Evaluation

#### Classification Report
<p> Here this project does additional metrics to evaluate the model by preparing a classification report </p> 

#### Evaluation Function
<p> This project creates an evaluation function which is designed to test the variance of different models across different subsets of the data. This function takes as parameters, the X and y arrays, the number of iterations that should be run, the training/test split size, the text encoding function, the model types and a boolean that states whether random random_states should be used or not </p>

```
def evaluate_variance(X, y, num_iterations, test_size, textEncoder, modelType, random_state=True):
    accuracy_scores = []

    if not random_state:
        num_iterations = 10
        
    for i in range(num_iterations):

        random_states = [12345, 54321, 32145, 43125, 23145, 0, 10, 20, 30, 40, 50]

        if random_state:
            random_states = np.random.randint(1, 100000, size=num_iterations)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_states[i])
        
        # Create and fit the model
        model = make_pipeline(textEncoder, modelType)
        model.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Run ", i, " with random state: ", random_states[i], ": ",accuracy)
        accuracy_scores.append(accuracy)

    # Calculate the average accuracy and variance
    average_accuracy = np.mean(accuracy_scores)
    variance_accuracy = np.var(accuracy_scores)
        
    return average_accuracy, variance_accuracy
```

#### Feature Engineering
<p> The next difference here is that some basic feature engineering is conducted to try and increase model accuracy. </p>

1. Ignore Case: All words in X are made lowercase. This had no effect on accuracy of algorithm.
2. Removing Punctuation: All punctuation is removed from X. This lowered accuracy from 0.9539 to 0.9451. This implies punctuation is useful in the detection of Spam messages. In particular removing punctuation led to more false positives for spam messages.

#### Model Selection
<p> This project also tried to experiment with different encodings and algorithm variations to try and get the highest accuracy possible for predictions.</p>

1. CountVectorizer for text encoding: Switched the text encoding method to a simple CountVectorizer which counts word frequencies like TFid but does not add a weight to word importance. This boosted accuracy from .9539 to .9825 and greatly reduced false positive rates for Spam messages.
2. CountVectorizer ignoring frequency: Using the binary=True option on CountVectorizer makes all non-zero frequencies = 1. So this only uses a binary consideration of whether a word is within a message or not. With this option, accuracy dropped from .9825 to .9817. This result shows that word frequency, perhaps surprisely, while useful does not have that great an impact on model accuracy.
3. BernoulliDB: As BernoulliNB is designed for binary values, the next test was to switch to this model while keeping CountVectorizer with binary set to True. Even though Bernoulli is designed to work with binary values, it performs worse than Multinomial Naive Bayes with accuracy of .96

<p>Overall the best performance was achieved using a combination of CountVectorizer for text encoding and MultinomialNB </p>

### Deployment
<p> Another change to this project is that a simple webapp was created to demonstrate the deployment of such a model. As such at the end of this Jupyter notebook, pickle was used to dump the finetuned model to a file. This file was then used to build an online webapp using the model to predict if entered text is SPAM. You can find the website [here](http://roadlesswalked.pythonanywhere.com/), please feel free to try it out. </p>

