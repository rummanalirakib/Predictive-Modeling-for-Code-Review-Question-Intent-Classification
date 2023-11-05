import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import precision_recall_fscore_support
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler


def QuestionDevider(text):
    question = []
    sentence = ""
    index=0
    for i in range(0, len(text)):
        if text[i] == '.':
            if i-1>0 and i-2>0 and text[i-1]=='s' and text[i-2]=='v':
                continue
            sentence = ""
        elif text[i] == '?':
            sentence += "?"
            question.insert(index, sentence)
            sentence=""
            index+=1
        else:
            sentence+=text[i]
    return question

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenization (split text into words)
    words = text.split()
    
    # Remove punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Rejoin the words to form the preprocessed text
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text

def LabelsDevider(labels):
    label_mapping = {
        'suggestion': 1,
        'action': 2,
        'confirmation': 2,
        'information': 2,
        'rationale': 2,
        'clarification': 2,
        'opinion': 2,
        'criticism': 3,
        'anger': 3,
        'surprise': 3,
        'hypothetical scenario': 4,
        'rhetorical question': 5,
        'request for opinion': 2,
        'request for information': 2,
        'request for rationale': 2,
        'request for confirmation': 2,
        'request for clarification': 2,
        'request for action': 2,
        'discarded': 2,
    }
    
    labels = labels.lower()
    for label, value in label_mapping.items():
        if label==labels:
            return value


excel_file_path = 'Assignment-2_icsme-questions-labeled.xlsx' 
workbook = openpyxl.load_workbook(excel_file_path)
worksheet = workbook.active

# Define a list to store red-marked questions
red_marked_questions = []
questionsColumn = []
labelsColumn = []
count1=0
index=0
count2=0
flag=1
tempQuestionCell=""
# Iterate through rows and extract red-marked questions
for row in worksheet.iter_rows(values_only=True):
    if flag == 1:
        flag=0
        continue;
    question_cell = row[2]
    if count1 > 0 and tempQuestionCell==question_cell:
        labelsColumn.insert(index, LabelsDevider(row[3]))
        count2+=1
        questionsColumn.insert(index, red_marked_questions[count2])
        count1-=1
        index+=1
        continue
    else:
        count2=0
    red_marked_questions = QuestionDevider(question_cell)
    count1=len(red_marked_questions)
    tempQuestionCell=question_cell
    if count1>0:
        count1-=1
        labelsColumn.insert(index, LabelsDevider(row[3]))
        questionsColumn.insert(index, red_marked_questions[count2])
        index+=1

questionsColumn = [preprocess_text(text) for text in questionsColumn]
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
Question_tf_idf = tfidf_vectorizer.fit_transform(questionsColumn)

# Create a feature matrix that combines TF-IDF values and question length
questionLength = np.array([len(question.split()) for question in questionsColumn])
X_train_test_combined = hstack([Question_tf_idf, questionLength.reshape(-1, 1)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train_test_combined, labelsColumn, test_size=0.3, random_state=42, stratify=None)

randomOverSampler = RandomOverSampler(random_state=42)

# Resample the training data
X_train_resampled, y_train_resampled = randomOverSampler.fit_resample(X_train, y_train)

svc_model = SVC(kernel='linear')
svc_model.fit(X_train_resampled, y_train_resampled)

# Make predictions using SVC model
y_pred = svc_model.predict(X_test)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:\n", report)