import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Startup Success", layout="centered")
st.title("🚀 Simple Startup Success Checker")


@st.cache_data
def load_data():
    df = pd.read_csv("startup_success_data.csv")
    df = df.drop(columns=["Name"], errors="ignore")
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)
    return df

df = load_data()

# Prepare Data
cat_cols = df.select_dtypes(include='object').columns
num_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Success']).columns

encoder = OneHotEncoder(sparse_output=False)
encoded_cat = encoder.fit_transform(df[cat_cols])

X = np.hstack((df[num_cols].values, encoded_cat))
y = df["Success"].values

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = RandomForestClassifier()
model.fit(X_train, y_train)


st.subheader("🧪 Enter Your Startup Info")

industry = st.selectbox("Industry", df["Industry"].unique())
location = st.selectbox("Location", df["Location"].unique())
funding = st.slider("Funding ($M)", 0.0, 100.0, 10.0)
team = st.slider("Team Size", 1, 500, 10)
age = st.slider("Startup Age", 0, 20, 2)

# Predict
input_df = pd.DataFrame([{
    "Industry": industry,
    "Location": location,
    "Funding($M)": funding,
    "TeamSize": team,
    "Age(Years)": age
}])

combined_df = pd.concat([df.drop(columns=["Success"]), input_df], ignore_index=True)
encoded_input = encoder.transform(combined_df[cat_cols])
X_input = np.hstack((combined_df[num_cols].values, encoded_input))
pred = model.predict(X_input[-1].reshape(1, -1))[0]
conf = model.predict_proba(X_input[-1].reshape(1, -1))[0]

# Result
st.subheader("🎯 Prediction")
if pred == 1:
    st.success(f"✅ Your startup is likely to Succeed! ({conf[1]*100:.1f}% confidence)")
else:
    st.error(f"❌ Your startup may Fail. ({conf[0]*100:.1f}% confidence)")

# Pie Chart
fig1, ax1 = plt.subplots()
ax1.pie([conf[1], conf[0]], labels=["Success", "Fail"], autopct="%1.1f%%", colors=["green", "red"])
st.pyplot(fig1)

# Bar Chart
fig2, ax2 = plt.subplots()
ax2.bar(["Your Prediction"], [pred], color="blue")
ax2.set_ylim(0, 1.2)
ax2.set_ylabel("0 = Fail, 1 = Success")
st.pyplot(fig2)
