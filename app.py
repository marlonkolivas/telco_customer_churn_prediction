# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve

# ======================
# 1. Load & Preprocess Data
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_customer_churn.csv")
    df = df.drop(columns=['customerID'])
    df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    cat_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    num_cols = df_encoded.select_dtypes(include=['int64','float64']).columns.tolist()
    num_cols.remove('Churn')
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    return df, X, y

raw_df, X, y = load_data()

# ======================
# 2. Sidebar Settings
# ======================
st.sidebar.header("App Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
threshold = st.sidebar.slider("Churn Probability Threshold", 0.0, 1.0, 0.5, 0.01)

# ======================
# 3. Train/Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ======================
# 4. Train Selected Model
# ======================
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=200, random_state=42)
else:
    model = GradientBoostingClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:,1]
y_pred = (y_proba >= threshold).astype(int)

# ======================
# 5. Tabs for Organization
# ======================
tab1, tab2, tab3, tab4 = st.tabs(["Metrics & Evaluation", "EDA & Visualization", "Feature Importance", "Customer Prediction"])

# ----------------------
# Tab 1: Metrics & Evaluation
# ----------------------
with tab1:
    st.header(f"{model_choice} Model Performance")
    st.write(f"Threshold: {threshold}")
    st.write(f"Train Accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
    st.write(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    st.write(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig_cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = px.area(x=fpr, y=tpr, title='ROC Curve', labels={'x':'False Positive Rate','y':'True Positive Rate'})
    st.plotly_chart(fig_roc)

# ----------------------
# Tab 2: EDA & Visualization
# ----------------------
with tab2:
    st.header("Exploratory Data Analysis")
    st.write(raw_df.describe())
    st.write("Churn Distribution")
    fig_churn = px.histogram(raw_df, x="Churn", color="Churn", title="Churn Distribution")
    st.plotly_chart(fig_churn)

    # Churn by categorical features
    cat_cols = raw_df.select_dtypes(include=['object']).columns
    selected_cat = st.selectbox("Select Categorical Feature to Visualize Churn", cat_cols)
    fig_cat = px.histogram(raw_df, x=selected_cat, color="Churn", barmode='group', title=f"Churn vs {selected_cat}")
    st.plotly_chart(fig_cat)

# ----------------------
# Tab 3: Feature Importance
# ----------------------
with tab3:
    st.header("Feature Importance / Coefficients")
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values(by='importance', ascending=False)
        fig_feat = px.bar(feat_df.head(10), x='importance', y='feature', orientation='h', title="Top 10 Feature Importances")
        st.plotly_chart(fig_feat)
    else:  # Logistic Regression
        coefs = model.coef_[0]
        feat_df = pd.DataFrame({'feature': X_train.columns, 'coefficient': coefs}).sort_values(by='coefficient', ascending=False)
        fig_feat = px.bar(feat_df.head(10), x='coefficient', y='feature', orientation='h', title="Top 10 Features Increasing Churn Probability")
        st.plotly_chart(fig_feat)

    # SHAP Explanation
    st.subheader("SHAP Summary (Global Feature Impact)")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')

# ----------------------
# Tab 4: Customer-Level Prediction (Final Optimized Version)
# ----------------------
with tab4:
    st.header("Predict Churn for a Custom Customer")

    st.write("""
    Enter a customer's details below. The app will preprocess your inputs 
    exactly as during model training (encoding and scaling) before prediction.
    """)

    # Use one random sample as reference
    sample_customer = raw_df.sample(1).iloc[0]
    st.subheader("Customer Input Form")

    col1, col2 = st.columns(2)
    input_data = {}

    for i, col in enumerate(raw_df.columns):
        if col == "Churn":
            continue
        elif raw_df[col].dtype == "object":
            if i % 2 == 0:
                input_data[col] = col1.selectbox(f"{col}", raw_df[col].unique(), index=0)
            else:
                input_data[col] = col2.selectbox(f"{col}", raw_df[col].unique(), index=0)
        else:
            default_val = float(sample_customer[col]) if not pd.isna(sample_customer[col]) else 0.0
            if i % 2 == 0:
                input_data[col] = col1.number_input(f"{col}", value=default_val)
            else:
                input_data[col] = col2.number_input(f"{col}", value=default_val)

    # ✅ Safe and optimized feature alignment
    @st.cache_data(show_spinner=False)
    def align_features(_X_columns, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns the new customer DataFrame to match training feature columns.
        Prevents hashing issues and handles missing/extra columns efficiently.
        """
        X_cols = list(_X_columns)

        missing_cols = [col for col in X_cols if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in X_cols]

        # Add missing columns with 0 values
        if missing_cols:
            df_missing = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, df_missing], axis=1)

        # Drop any unexpected columns
        if extra_cols:
            df = df.drop(columns=extra_cols)

        # Match exact column order
        df = df[X_cols].copy()
        return df

    # Prediction process
    if st.button("Predict Churn Probability"):
        new_customer = pd.DataFrame([input_data])

        # --- Apply same preprocessing as training ---
        new_encoded = pd.get_dummies(new_customer, drop_first=True)
        new_encoded = align_features(X.columns, new_encoded)

        # Predict churn probability
        churn_prob = model.predict_proba(new_encoded)[:, 1][0]
        churn_pred = int(churn_prob >= threshold)

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Predicted Churn Probability:** {churn_prob:.2%}")
        if churn_pred == 1:
            st.error("⚠️ This customer is likely to CHURN.")
        else:
            st.success("✅ This customer is likely to STAY.")

