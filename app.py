import streamlit as st
import pickle
import os
from train import train_model

st.set_page_config(page_title="Spam Detector", page_icon="🚫", layout="centered")
st.title("🚫 Spam Detector")
st.caption("paste a message and find out if it's spam - trained on real SMS data")

# train model if not already saved
# didn't want to commit the model file so just retrain on first run
if not os.path.exists("model.pkl"):
    with st.spinner("first run - training model, takes a few seconds..."):
        train_model()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.divider()

text = st.text_area("message to check", placeholder="e.g. Congratulations! You've won a free iPhone. Click here to claim...", height=150)

if st.button("check", type="primary"):
    if not text.strip():
        st.warning("enter a message first")
    else:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        spam_score = round(proba[1] * 100, 1)
        ham_score = round(proba[0] * 100, 1)

        if prediction == 1:
            st.error(f"**SPAM** - {spam_score}% confidence")
        else:
            st.success(f"**NOT SPAM** - {ham_score}% confidence")

        # show the breakdown
        col1, col2 = st.columns(2)
        col1.metric("spam probability", f"{spam_score}%")
        col2.metric("not spam probability", f"{ham_score}%")

st.divider()
st.markdown("**try these examples:**")

examples = [
    "WINNER!! You have been selected to receive a $1000 gift card. Call now!",
    "hey are you coming to the meeting tomorrow at 3pm?",
    "FREE entry in 2 a weekly competition to win FA Cup final tickets!",
    "can you pick up some groceries on your way home",
]

for ex in examples:
    if st.button(ex[:60] + "..." if len(ex) > 60 else ex, key=ex):
        st.session_state["example"] = ex
        st.rerun()

if "example" in st.session_state:
    st.info(f"selected: {st.session_state['example']}")
