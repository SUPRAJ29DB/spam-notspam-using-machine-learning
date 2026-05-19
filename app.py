import streamlit as st
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #141e30, #243b55);
}

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

h1 {
    text-align: center;
    color: #ffffff;
    font-size: 45px;
}

textarea {
    background-color: #1e1e1e !important;
    color: white !important;
    border-radius: 10px !important;
    border: 2px solid #00c6ff !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background: linear-gradient(to right, #0072ff, #00c6ff);
    transform: scale(1.02);
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>📧 Email Spam Detection</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <center>
    Detect whether an email or message is <b>Spam</b> or <b>Not Spam</b>
    using Machine Learning.
    </center>
    <br>
    """,
    unsafe_allow_html=True
)

# ---------------- INPUT AREA ----------------
message = st.text_area(
    "Enter Your Email Message",
    height=200,
    placeholder="Type your email/message here..."
)

# ---------------- BUTTON ----------------
if st.button("🔍 Detect Message"):

    if message.strip() == "":
        st.warning("⚠ Please enter a message first.")
    else:

        # Transform text
        data = vectorizer.transform([message])

        # Prediction
        prediction = model.predict(data)

        # Probability
        probability = model.predict_proba(data)

        spam_prob = probability[0][1] * 100
        ham_prob = probability[0][0] * 100

        # RESULT
        if prediction[0] == 1:
            st.markdown(
                f"""
                <div class="result-box" style="background-color:#ff4b4b;">
                    🚨 SPAM MESSAGE<br><br>
                    Spam Probability: {spam_prob:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-box" style="background-color:#28a745;">
                    ✅ NOT SPAM<br><br>
                    Safe Probability: {ham_prob:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This project uses:

    ✅ Machine Learning  
    ✅ Naive Bayes Algorithm  
    ✅ TF-IDF Vectorization  
    ✅ Streamlit UI  

    Developed using Python.
    """
)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        Made with ❤️ using Streamlit & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)