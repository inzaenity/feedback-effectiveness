# feedback-effectiveness

## Abstract
Using Machine learning models to evaluate and give feedback for the effectiveness of argumentative writing. 
The goal is to classify a text as Ineffective, Adequate, Effective given its discourse lead, Position, Claim, Evidence, Counterclaim, Rebuttal and Concluding Statement.

For this project I explore Traditional Algorithms and CNNs
 * Explore four popular machine learning algorithms, namely logistic regression, multinomial naive Bayes, kNN and decision trees.
 * Explore of Convolutional Neural Networks.

## Usage

**Running the code**
 ```
 python3 main.py
 ```
**Files Walkthrough**
data/train.csv - the training data

main.py - **Run this for section 2 results.** This writes to best_algorithms.csv and populates the final_model_training and outer_cv_results directories.
hyperfinder.py - code for a general class that find hyperparameters for a given algorithm
best_algorithms.csv - overall results file for section 2
final_model_training/ - directory of hyperparameter randomised search results, as csv files
outer_cv_results/ - directory of outer cross validation results

cnn_main.py - **Run this for section 3 results.** This plots training histories of the best models for each discourse type, and writes to cnn_final_models.csv.
cnn.py - defines CNN class
embedding.py - code for loading GloVe embeddings
preprocess.py - code for text preprocessing, including zero padding sequences
cnn_final_models.csv - results for best CNN model found for each discourse type
cnn_graphs/ - directory of training histories of the best models for each discourse type
glove.6B.50d.txt - GloVe embedding file,  from https://nlp.stanford.edu/projects/glove/ which is inside the glove.6B.zip
contractions.csv - small dataset of English contractions





