# Predictive-Modeling-for-Code-Review-Question-Intent-Classification


SCHOOL OF COMPUTER SCIENCE
COMP 8130
Topics in Software Engineering
This is an individual assignment. Cheating and plagiarism in any part of the
assignment will result in zero for the whole assignment
In this second assignment, you need to develop a classifier to predict the intention in
the code review questions. Select one multi-class prediction model from sklearn or
any other machine learning libraries.
Deadline: October 20, 2023 (11.59 PM)
Total Marks: 10
Reference Paper: Predicting Communicative Intention in Code Review
Questions
Dataset: icsme-questions-labeled.xlsx (available on the course website)
Note: Include at least one new feature other than only considering the words appearing
in those questions (you can consider more features!).
Input: A specific question in the code review comments
Labels: Consider the top-level categories
1. Suggestions
2. Requests
3. Attitudes and emotions
4. Hypothetical scenario
5. Rhetorical questions

   
Examples from the provided dataset

The dataset will be available with the repository as well

Tasks: 
1. Generate a model with words as the feature, with text preprocessing step
2. Generate a model with words as the feature + your feature, without text
preprocessing step
3. Generate a model with words as the feature + your feature, with text preprocessing
step
4. Generate a model with words as the feature + your feature, with resampling and
text preprocessing step
Submission instructions
Submit the code as a Jupyter notebook, data file(2) to train and test the technique,
a document describing the experiment process (same format that you used for
submitting project proposal) and the results of four models (Not more than four
pages).
The document must contain an introduction describing the problem description,
dataset description, experiment process, study results and references.
