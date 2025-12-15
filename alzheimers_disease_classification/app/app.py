import os
import hashlib
import pandas as pd
import joblib
import streamlit as st


# ==========================
# User storage helpers
# ==========================

def get_users_file_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    return os.path.join(project_root, "users.csv")


def init_users_file():
    """Create users.csv if it doesn't exist."""
    users_path = get_users_file_path()
    if not os.path.exists(users_path):
        df = pd.DataFrame(columns=["username", "password_hash"])
        df.to_csv(users_path, index=False)


def hash_password(password: str) -> str:
    """Return SHA-256 hash of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def load_users() -> pd.DataFrame:
    users_path = get_users_file_path()
    if not os.path.exists(users_path):
        init_users_file()
    return pd.read_csv(users_path)


def save_users(df: pd.DataFrame):
    users_path = get_users_file_path()
    df.to_csv(users_path, index=False)


def username_exists(username: str) -> bool:
    df = load_users()
    return username in df["username"].values


def validate_login(username: str, password: str) -> bool:
    df = load_users()
    password_hash = hash_password(password)
    match = df[(df["username"] == username) & (df["password_hash"] == password_hash)]
    return not match.empty


def create_user(username: str, password: str) -> bool:
    if username_exists(username):
        return False
    df = load_users()
    new_row = {"username": username, "password_hash": hash_password(password)}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_users(df)
    return True


def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()


# ==========================
# Model / preprocessing loader
# ==========================

@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    model_dir = os.path.join(project_root, "models")
    data_path = os.path.join(project_root, "data", "alzheimers_disease_data.csv")

    model_path = os.path.join(model_dir, "gradient_boosting_model.pkl")
    imputer_path = os.path.join(model_dir, "imputer.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(data_path)

    # Drop only if they exist
    drop_cols = [c for c in ["PatientID", "DoctorInCharge"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    X = df.drop(columns=["Diagnosis"])
    feature_cols = list(X.columns)
    medians = X.median(numeric_only=True).to_dict()

    return model, imputer, scaler, feature_cols, medians


# ==========================
# Page: Home
# ==========================

def show_home_page():
    username = st.session_state.username or "User"
    st.subheader("🏠 Home")

    st.markdown(
        f"""
        ### Welcome, **{username}** 👋

        This portal is part of an **Alzheimer’s Disease Classification** project.

        #### What this dashboard does

        - Uses a trained **Gradient Boosting model** to estimate whether a patient
          is likely to have Alzheimer's Disease.
        - Uses **clinical, cognitive and lifestyle features** as inputs.
        - Provides:
          - A **prediction** (Alzheimer’s / No Alzheimer’s)
          - An **estimated probability** (0–100%)
          - A **simple 5-year risk trend** based on age increasing.

        #### How to use

        1. Open the **Prediction** page from the left sidebar.
        2. Adjust the main sliders for key features (Age, MMSE, etc.).
        3. Optionally open **Advanced features** for full control.
        4. Click **Predict Alzheimer Risk** to see the result and short guidance.
        5. After at least one prediction, visit **Future 5-year Risk** to view
           a basic risk trend over the next 5 years.

        """
    )


# ==========================
# Page: Prediction
# ==========================

def show_prediction_page(model, imputer, scaler, feature_cols, medians):
    st.subheader("🔍 Alzheimer Risk Prediction")

    st.write("Adjust the key sliders below, optionally edit advanced features, then click **Predict Alzheimer Risk**.")

    view_mode = st.radio(
        "Select result view",
        ["Simple", "Detailed"],
        horizontal=True,
    )

    # Prepare dict with defaults
    input_data = {col: float(medians.get(col, 0.0)) for col in feature_cols}

    # -----------------------------
    # Important features with sliders
    # -----------------------------
    st.markdown("### Main clinical inputs")

    col1, col2 = st.columns(2)

    important_fields = ["Age", "BMI", "MMSE", "FunctionalAssessment", "ADL"]

    with col1:
        if "Age" in feature_cols:
            input_data["Age"] = st.slider(
                "Age (years)",
                min_value=40,
                max_value=100,
                value=int(medians.get("Age", 70)),
            )

        if "BMI" in feature_cols:
            input_data["BMI"] = st.slider(
                "BMI",
                min_value=10.0,
                max_value=40.0,
                value=float(medians.get("BMI", 24.0)),
            )

        if "MMSE" in feature_cols:
            input_data["MMSE"] = st.slider(
                "MMSE Score (0–30)",
                min_value=0,
                max_value=30,
                value=int(medians.get("MMSE", 20)),
            )

    with col2:
        if "FunctionalAssessment" in feature_cols:
            input_data["FunctionalAssessment"] = st.slider(
                "Functional Assessment",
                min_value=0.0,
                max_value=30.0,
                value=float(medians.get("FunctionalAssessment", 15.0)),
            )

        if "ADL" in feature_cols:
            input_data["ADL"] = st.slider(
                "ADL (Daily Living Activities)",
                min_value=0.0,
                max_value=10.0,
                value=float(medians.get("ADL", 5.0)),
            )

    # -----------------------------
    # Advanced features (all remaining)
    # -----------------------------
    with st.expander("⚙️ Advanced features (optional)"):
        adv_col1, adv_col2 = st.columns(2)
        for i, col in enumerate(feature_cols):
            if col in important_fields:
                continue
            default_val = float(medians.get(col, 0.0))
            target_col = adv_col1 if i % 2 == 0 else adv_col2
            with target_col:
                input_data[col] = st.number_input(col, value=default_val)

    # -----------------------------
    # Prediction button
    # -----------------------------
    if st.button("Predict Alzheimer Risk"):
        # Prepare data
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_cols]

        # Preprocess
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Predict
        pred_class = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0, 1]

        # Save for future risk page
        st.session_state.last_input = input_data
        st.session_state.last_proba = float(pred_proba)

        st.subheader("Prediction Result")

        if pred_class == 1:
            st.error(
                f"**Prediction:** Alzheimer's Disease (Positive)\n\n"
                f"**Estimated Probability:** {pred_proba:.2%}"
            )

            st.markdown("#### 🛡️ General precautions (for brain health)")
            st.markdown(
                """
                These are general brain-health suggestions. They are **not** personal medical advice:

                - 🩺 Consult a doctor or neurologist for proper clinical evaluation.
                - 💊 Keep other conditions (diabetes, blood pressure, cholesterol) under control.
                - 🧠 Stay mentally active: reading, puzzles, learning new skills.
                - 🚶‍♂️ Be physically active as advised by your doctor.
                - 👥 Stay connected with family and friends.
                - 😴 Maintain good sleep and try to manage stress.
                """
            )

        else:
            st.success(
                f"**Prediction:** No Alzheimer's Disease (Negative)\n\n"
                f"**Estimated Probability:** {pred_proba:.2%}"
            )

            st.markdown("#### 🌱 Healthy brain habits")
            st.markdown(
                """
                Even with a negative prediction, healthy habits are important:

                - 🧠 Keep learning and challenging your brain.
                - 🚶‍♀️ Do regular physical activity (after medical advice).
                - 🥗 Eat a balanced, nutritious diet.
                - 👥 Maintain social connections.
                - 😴 Sleep well and manage stress where possible.
                """
            )

        if view_mode == "Detailed":
            st.markdown(
                """
                **Model explanation (simple):**

                - The model looks at the features you entered (age, cognitive scores,
                  daily living activities, etc.).
                - It compares them with patterns it learned from the training data.
                - It outputs a probability and a class (Positive / Negative).
                - This is meant as a **learning tool**, not a clinical decision.
                """
            )


# ==========================
# Page: Future 5-year risk
# ==========================

def show_future_risk_page(model, imputer, scaler, feature_cols):
    st.subheader("📈 Future 5-Year Risk (Simple Scenario)")

    if "last_input" not in st.session_state:
        st.info(
            "No previous prediction found. Please go to the **Prediction** page, "
            "run at least one prediction, and then return here."
        )
        return

    input_data = st.session_state.last_input

    if "Age" not in feature_cols:
        st.warning("The dataset does not contain an 'Age' feature, so the 5-year view cannot be created.")
        return

    base_age = input_data["Age"]
    st.write(f"Using last input with starting Age = **{base_age}** years.")

    years = list(range(0, 6))  # now + 5 years
    ages = [base_age + y for y in years]
    probs = []

    for new_age in ages:
        modified_input = input_data.copy()
        modified_input["Age"] = new_age
        row_df = pd.DataFrame([modified_input])
        row_df = row_df[feature_cols]

        row_imputed = imputer.transform(row_df)
        row_scaled = scaler.transform(row_imputed)
        proba = model.predict_proba(row_scaled)[0, 1]
        probs.append(float(proba))

    proj_df = pd.DataFrame(
        {
            "Year (from now)": years,
            "Age": ages,
            "Estimated Risk Probability": probs,
        }
    )

    st.write("Scenario: risk if **only age increases** and all other features stay the same.")
    st.dataframe(proj_df.style.format({"Estimated Risk Probability": "{:.2%}"}), use_container_width=True)

    st.line_chart(
        proj_df.set_index("Year (from now)")["Estimated Risk Probability"],
        height=300,
    )

    st.caption(
        "This is a simple 'what-if' risk view, not a real forecast over time."
    )


# ==========================
# Main app
# ==========================

def main():
    st.set_page_config(
        page_title="Alzheimer's Dashboard",
        page_icon="🧠",
        layout="centered",
    )

    # Session state setup
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    st.title("🧠 Alzheimer's Project Portal")

    # If logged in → show dashboard with pages
    if st.session_state.logged_in:
        st.write(f"Welcome, **{st.session_state.username}**")
        st.button("Logout", on_click=logout)

        # Load model artifacts once
        model, imputer, scaler, feature_cols, medians = load_artifacts()

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to",
            ["Home", "Prediction", "Future 5-year Risk"],
        )

        if page == "Home":
            show_home_page()
        elif page == "Prediction":
            show_prediction_page(model, imputer, scaler, feature_cols, medians)
        elif page == "Future 5-year Risk":
            show_future_risk_page(model, imputer, scaler, feature_cols)

        return

    # If not logged in → show login / signup
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Sign up"])

    # Login
    with tab1:
        st.subheader("Login to your account")

        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login_username.strip() == "" or login_password.strip() == "":
                st.error("Please enter both username and password.")
            elif validate_login(login_username, login_password):
                st.success("Login successful ✅")
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # Signup
    with tab2:
        st.subheader("Create a new account")

        signup_username = st.text_input("Choose a username", key="signup_username")
        signup_password = st.text_input("Choose a password", type="password", key="signup_password")
        signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm")

        if st.button("Sign up"):
            if signup_username.strip() == "" or signup_password.strip() == "":
                st.error("Username and password cannot be empty.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match.")
            elif username_exists(signup_username):
                st.error("Username already exists. Please choose another one.")
            else:
                ok = create_user(signup_username, signup_password)
                if ok:
                    st.success("Account created successfully ✅. You can now login.")
                else:
                    st.error("Something went wrong creating the user. Please try again.")


if __name__ == "__main__":
    init_users_file()
    main()
