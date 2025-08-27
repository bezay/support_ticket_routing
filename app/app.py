import streamlit as st
import joblib
import os

st.set_page_config(page_title="AI Enquiry Classifier", page_icon="📨")

@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train it first.")
        st.stop()
    return joblib.load(model_path)

st.title("📨 AI Enquiry Classification: Support Ticket Routing")
st.markdown("Type a customer message and I'll predict the category.")

model_path = os.environ.get("MODEL_PATH", "models/ticket_router.joblib")
model = load_model(model_path)

user_text = st.text_area("Customer message", height=160, placeholder="e.g., I was charged twice this month, please refund the extra charge.")
if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        pred = model.predict([user_text])[0]
        proba = None
        # LogisticRegression supports predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([user_text])[0]
            classes = model.classes_
            st.subheader("Prediction")
            st.write(f"**{pred}**")
            st.subheader("Confidence by class")
            for cls, p in sorted(zip(classes, proba), key=lambda x: x[1], reverse=True):
                st.write(f"- {cls}: {p:.3f}")
        else:
            st.subheader("Prediction")
            st.write(f"**{pred}**")
            st.info("Probability scores unavailable for this model.")