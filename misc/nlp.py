# Importing the libraries
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
from xgboost import XGBClassifier

nltk.download('wordnet')
nltk.download('omw-1.4')

# To count the iterations
from tqdm import tqdm

# Importing the dataset
dataset = pd.read_csv('Comments.csv')


def removeHTMLTags(review):
    soup = BeautifulSoup(review, 'lxml')
    return soup.get_text()


def removeApostrophe(review):
    phrase = re.sub(r"won't", "will not", review)
    phrase = re.sub(r"can\'t", "can not", review)
    phrase = re.sub(r"n\'t", " not", review)
    phrase = re.sub(r"\'re", " are", review)
    phrase = re.sub(r"\'s", " is", review)
    phrase = re.sub(r"\'d", " would", review)
    phrase = re.sub(r"\'ll", " will", review)
    phrase = re.sub(r"\'t", " not", review)
    phrase = re.sub(r"\'ve", " have", review)
    phrase = re.sub(r"\'m", " am", review)
    return phrase


def removeAlphaNumericWords(review):
    return re.sub("\S*\d\S*", "", review).strip()


def removeSpecialChars(review):
    return re.sub('[^a-zA-Z]', ' ', review)


def scorePartition(x):
    if x < 3:
        return 0
    return 1


def doTextCleaning(review):
    review = removeHTMLTags(review)
    review = removeApostrophe(review)
    review = removeAlphaNumericWords(review)
    review = removeSpecialChars(review)
    # Lower casing
    review = review.lower()
    # Tokenization
    review = review.split()
    # Removing Stopwords and Lemmatization
    lmtzr = WordNetLemmatizer()
    review = [lmtzr.lemmatize(word, 'v') for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    return review


# Generalizing the score
# actualScore = dataset['Score']
# positiveNegative = actualScore.map(scorePartition)
# dataset['Score'] = positiveNegative

# creating the document corpus
corpus = []
for index, row in tqdm(dataset.iterrows()):
    review = doTextCleaning(row['Comment'])
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

# Creating a tranform
cv = CountVectorizer(ngram_range=(1, 3), max_features=5000)
X = cv.fit_transform(corpus).toarray()
y = dataset['Label'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)


# Predict the sentiment for new review
def predictNewReview():
    newReview = ""
    print("Confusion Matrix",cm)
    #print("Prediction", y_pred)
    counter=0
    # for i in y_pred:
    #     if y_pred[counter]==1:
    #         pass
    #     counter+=1
#
    data=cv.inverse_transform(X_test)
    print(y_pred[0])
    print(data[0])
#
# for i in range(100):
#     if y_pred[i] == 1: print("true")





    newReview = input("Type the Review: ")
    if newReview == '':
        print('Invalid Review')
    else:
        newReview = doTextCleaning(newReview)
        new_review = cv.transform([newReview]).toarray()
        prediction = classifier.predict(new_review)
        # score = TextBlob(newReview).sentiment.polarity
        # print("Dictionary:", score)
        print(prediction)
        if prediction[0] == 1:
            print("Positive Comment")
        if prediction[0] == -1:
            print("Negative Comment")


if __name__ == '__main__':
    predictNewReview()
