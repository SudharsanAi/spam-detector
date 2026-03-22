import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import urllib.request

def train_model():
    # using the UCI SMS spam dataset - classic benchmark for this kind of thing
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

    try:
        df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    except Exception:
        # fallback - use a small hardcoded dataset if no internet
        # not great but at least the app won't crash
        data = {
            "label": ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam",
                      "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"],
            "message": [
                "Hey, are you free tomorrow?",
                "WINNER! You have been selected for a cash prize of $1000!",
                "Can you pick me up at 6pm?",
                "FREE entry! Text WIN to 12345 to claim your prize now!",
                "Meeting moved to 3pm, see you there",
                "Urgent! Your account has been compromised. Click here immediately!",
                "What do you want for dinner tonight?",
                "Congratulations! You've won a holiday. Call 0800 now!",
                "I'll be late, traffic is bad",
                "Your loan has been approved! Claim $5000 now, no credit check!",
                "Did you watch the game last night?",
                "You have been chosen to receive a free gift card worth $500",
                "Can we reschedule our meeting?",
                "Double your income working from home! Limited offer!",
                "Don't forget mom's birthday is next week",
                "You are a WINNER of our weekly prize draw!"
            ]
        }
        df = pd.DataFrame(data)

    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    # tfidf works better than count vectorizer for this - tried both
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(df["message"])
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"model accuracy: {acc:.2%}")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("model saved")

if __name__ == "__main__":
    train_model()
