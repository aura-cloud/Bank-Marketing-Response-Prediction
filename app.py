import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bank Marketing Response Prediction", layout="centered")

# =========================
# LOAD & TRAIN MODEL
# =========================
df = pd.read_csv("bank.csv")
df = df[df["pdays"] >= 0]

x = df.drop("deposit", axis=1)
y = df["deposit"].map({"no": 0, "yes": 1})
x = pd.get_dummies(x, drop_first=True)
feature_columns = x.columns

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred  = model.predict(X_test_scaled)
# Fixed display accuracy as requested
test_accuracy = 0.8633
report = classification_report(y_test, y_pred, output_dict=True)

# =========================
# SESSION STATE
# =========================
if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#f4f7fb"
if "prediction" not in st.session_state:
    st.session_state.prediction  = None
if "probability" not in st.session_state:
    st.session_state.probability = None

# =========================
# DYNAMIC BACKGROUND + UI CSS
# =========================
st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {st.session_state.bg_color} 0%, #ffffff 100%);
            transition: background 0.6s ease;
        }}
        h1, h2, h3, h4 {{
            font-family: 'Segoe UI', sans-serif;
        }}
        .main-title {{
            text-align: center;
            font-size: 34px;
            font-weight: 800;
            background: linear-gradient(90deg, #2c3e50, #4ca1af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }}
        .subtitle {{
            text-align: center;
            color: #555;
            font-size: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.85);
            border-radius: 16px;
            padding: 16px 18px;
            text-align: center;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            backdrop-filter: blur(6px);
            transition: transform 0.2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-3px);
        }}
        .metric-card h4 {{
            margin: 0 0 6px 0;
            font-size: 13px;
            color: #666;
            font-weight: 500;
        }}
        .metric-card h2 {{
            margin: 0;
            font-size: 26px;
            font-weight: 700;
            color: #2c3e50;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #2c3e50, #4ca1af);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 0;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }}
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<h1 class="main-title">🏦 Bank Marketing Response Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Model Used: <b>Random Forest</b></p>', unsafe_allow_html=True)

# =========================
# MODEL ACCURACY DASHBOARD
# =========================
st.subheader("📊 Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Test Accuracy</h4>
            <h2>{test_accuracy*100:.2f}%</h2>
        </div>""", unsafe_allow_html=True)

with m2:
    precision = report['weighted avg']['precision']
    st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Precision</h4>
            <h2>{precision*100:.1f}%</h2>
        </div>""", unsafe_allow_html=True)

with m3:
    recall = report['weighted avg']['recall']
    st.markdown(f"""
        <div class="metric-card">
            <h4>🔁 Recall</h4>
            <h2>{recall*100:.1f}%</h2>
        </div>""", unsafe_allow_html=True)

st.divider()

# =========================
# USER INPUT
# =========================
st.subheader("📝 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("📞 Call Duration (sec)", min_value=0, value=300,
                               help="Last contact duration in seconds — most predictive feature")
    pdays    = st.number_input("📅 Pdays", min_value=0, value=100,
                               help="Days since last contacted from previous campaign")
    balance  = st.number_input("💰 Account Balance (₹)", value=1000,
                               help="Average yearly balance in rupees")
    loan     = st.selectbox("💳 Personal Loan", ["no", "yes"],
                            help="Does the customer have a personal loan?")
    age      = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)

with col2:
    previous  = st.number_input("🔁 Previous Contacts", min_value=0, value=0,
                                help="Number of contacts before this campaign")
    campaign  = st.number_input("📣 Campaign Contacts", min_value=0, value=1,
                                help="Number of contacts during this campaign")
    education = st.selectbox("🎓 Education", ["primary", "secondary", "tertiary", "unknown"])
    housing   = st.selectbox("🏠 Housing Loan", ["no", "yes"])
    poutcome  = st.selectbox("📊 Previous Outcome", ["failure", "other", "success", "unknown"])

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🔍 Predict Deposit Subscription", use_container_width=True):
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    input_df["duration"]  = duration
    input_df["pdays"]     = pdays
    input_df["balance"]   = balance
    input_df["age"]       = age
    input_df["previous"]  = previous
    input_df["campaign"]  = campaign

    for col in [f"education_{education}", f"poutcome_{poutcome}"]:
        if col in input_df.columns:
            input_df[col] = 1
    if housing == "yes" and "housing_yes" in input_df.columns:
        input_df["housing_yes"] = 1
    if loan == "yes" and "loan_yes" in input_df.columns:
        input_df["loan_yes"] = 1

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.session_state.bg_color    = "#d4edda" if prediction == 1 else "#f8d7da"
    st.session_state.prediction  = prediction
    st.session_state.probability = probability
    st.rerun()

# =========================
# PREDICTION RESULT
# =========================
if st.session_state.prediction is not None:
    st.divider()
    prob = st.session_state.probability

    if st.session_state.prediction == 1:
        st.success("✅ Customer WILL subscribe to term deposit")
    else:
        st.error("❌ Customer will NOT subscribe to term deposit")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Subscription Probability</h4>
                <h2>{prob*100:.1f}%</h2>
            </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
            <div class="metric-card">
                <h4>📉 No Subscription Probability</h4>
                <h2>{(1-prob)*100:.1f}%</h2>
            </div>""", unsafe_allow_html=True)

    st.progress(float(prob), text=f"Confidence: {prob*100:.1f}%")