# spam-detector

built this after getting annoyed at how many spam messages i was seeing. wanted to understand how spam filters actually work under the hood, so i trained one myself.

uses naive bayes with tfidf vectorization - turns out this simple combination works surprisingly well for text classification. accuracy comes out around 97-98% on the UCI SMS spam dataset.

## what it does

paste any message - SMS, email, whatever - and it tells you if it's spam or not, with a confidence score.

## stack

- python
- scikit-learn - naive bayes classifier + tfidf vectorizer
- pandas - data loading and prep
- streamlit - ui

## how it works

```
raw message
  -> tfidf vectorizer converts text to numbers
  -> naive bayes calculates spam probability
  -> result + confidence score shown
```

tfidf (term frequency-inverse document frequency) weighs words by how often they appear in a message vs how common they are across all messages. words like "FREE", "WINNER", "CLAIM" get high weight in spam messages.

naive bayes works well here because it assumes word independence - not always true in reality but good enough for spam detection and fast to train.

## setup

```bash
git clone https://github.com/sudharsanai/spam-detector.git
cd spam-detector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

first run downloads the dataset and trains the model - takes about 10 seconds. after that it loads instantly from the saved model file.

## dataset

UCI SMS spam collection - 5574 messages, 13% spam, 87% ham. standard benchmark dataset for SMS spam classification.

## accuracy

around 97-98% on test set. false positive rate is low which matters more than raw accuracy - you don't want legitimate messages getting flagged.

## author

sudharsan m
