# streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import io
import time

# ---------- SETUP ----------
# Configure the Streamlit app
st.set_page_config(page_title="GymBot AI Trainer", layout="centered")

# Initialize session state variables for first-time users and app memory
if "first_visit" not in st.session_state:
    st.session_state.first_visit = True
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "user_X" not in st.session_state:
    st.session_state.user_X = []
    st.session_state.user_y = []
if "bot_name" not in st.session_state:
    st.session_state.bot_name = "GymBot"
if "last_raw_input" not in st.session_state:
    st.session_state.last_raw_input = None
if "last_weighted_input" not in st.session_state:
    st.session_state.last_weighted_input = None
if "last_hidden_output" not in st.session_state:
    st.session_state.last_hidden_output = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_history" not in st.session_state:
    st.session_state.last_history = None

# ---------- TUTORIAL ----------
def tutorial():
    # Display the onboarding guide on first visit
    st.markdown("""
    ### 👋 Welcome to your AI Gym Decision Trainer!
    This AI simulates your brain deciding: *Should I go to the gym today?*

    - You'll enter how you feel, how crowded the gym is, and how much time you have.
    - You'll also set how important each factor is to you.
    - The AI will show you how it's thinking — including its hidden neurons and final decision.
    - If you disagree, you can teach it what you'd prefer and it'll learn!
    """)
    st.success("Click below to begin!")
    if st.button("🚀 Start Using the App"):
        st.session_state.first_visit = False
        st.rerun()

# ---------- DATA FUNCTIONS ----------
def generate_training_data():
    # Generate initial synthetic training data
    np.random.seed(42)
    X = np.random.rand(150, 3)
    weights = np.array([0.6, -0.3, 0.5])
    y = (X @ weights > 0.4).astype(int).reshape(-1, 1)

    # Add manually curated examples
    X_manual = np.array([
        [0.8, 0.2, 0.3],
        [1.0, 0.1, 0.5],
        [0.9, 0.3, 0.4],
        [0.7, 0.1, 0.6],
        [1.0, 0.0, 0.9]
    ])
    y_manual = np.array([[1], [1], [1], [1], [1]])

    # Combine auto + manual data
    X = np.vstack((X, X_manual))
    y = np.vstack((y, y_manual))

    # Include learner-submitted training data
    if st.session_state.user_X:
        X = np.vstack((X, np.vstack(st.session_state.user_X)))
        y = np.vstack((y, np.vstack(st.session_state.user_y)))

    return X, y

def build_model():
    # Define the architecture: input → hidden layer → output
    inputs = Input(shape=(3,), name="input_layer")
    x = Dense(6, activation="relu", name="hidden_layer")(inputs)
    outputs = Dense(1, activation="sigmoid", name="output_layer")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, X, y):
    # Train the model with historical + learner data
    history = model.fit(X, y, epochs=100, verbose=0)
    return model, history

def predict_and_explain(model, input_data):
    # Extract hidden layer output and final prediction
    hidden_model = Model(inputs=model.input,
                         outputs=model.get_layer("hidden_layer").output)
    hidden_output = hidden_model.predict(input_data, verbose=0).flatten()
    final_output = float(model.predict(input_data, verbose=0)[0][0])
    return hidden_output, final_output

def show_loss_chart(history):
    # Visualise training loss curve
    buf = io.BytesIO()
    plt.figure()
    plt.plot(history.history["loss"], label="Loss")
    plt.title("Training Loss Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    st.image(buf)

# ---------- APP UI ----------
# Show tutorial if new user
if st.session_state.first_visit:
    tutorial()
    st.stop()

# Title area
st.title(f"🤖 {st.session_state.bot_name} - Gym Decision AI")

# Sidebar for input configuration
with st.sidebar:
    st.header("🎛️ Settings")
    st.text_input("🤖 Name your AI assistant:", value=st.session_state.bot_name, key="bot_name")
    scenario_labels = {
        "energised": st.text_input("Label for 'energised' input:", "How energised do you feel?"),
        "crowded": st.text_input("Label for 'crowded' input:", "How crowded is the gym?"),
        "time": st.text_input("Label for 'time' input:", "How much free time do you have?")
    }
    weight_labels = {
        "energised": st.text_input("Importance of energised:", "How important is feeling energised?"),
        "crowded": st.text_input("Importance of crowdedness:", "How much does a crowded gym affect you?"),
        "time": st.text_input("Importance of time:", "How important is having time?")
    }
    decision_labels = {
        "go": st.text_input("Label for GO decision:", "💪 Go to the gym!"),
        "stay": st.text_input("Label for STAY decision:", "😴 Stay home today.")
    }

# Input sliders
st.subheader("🧍 Your Situation")
cols = st.columns(3)
energised = cols[0].slider(scenario_labels["energised"], 0.0, 1.0, 0.7)
crowded = cols[1].slider(scenario_labels["crowded"], 0.0, 1.0, 0.3)
time = cols[2].slider(scenario_labels["time"], 0.0, 1.0, 0.5)

st.subheader("⚖️ What Matters to You?")
cols2 = st.columns(3)
w_energised = cols2[0].slider(weight_labels["energised"], 0.0, 1.0, 0.6)
w_crowded = cols2[1].slider(weight_labels["crowded"], -1.0, 1.0, -0.3)
w_time = cols2[2].slider(weight_labels["time"], 0.0, 1.0, 0.4)

# Ask button triggers training and prediction
with st.form("ask_form"):
    submitted = st.form_submit_button("💬 Submit to " + st.session_state.bot_name)

    if submitted:
        with st.spinner("🤖 Thinking hard about your gym situation..."):
            raw_input = np.array([energised, crowded, time])
            weights = np.array([w_energised, w_crowded, w_time])
            weighted_input = (raw_input * weights).reshape(1, -1)

            X, y = generate_training_data()
            model = build_model()
            model, history = train_model(model, X, y)

            hidden_output, prediction = predict_and_explain(model, weighted_input)
            decision = decision_labels["go"] if prediction > 0.5 else decision_labels["stay"]

            # Save for later use
            st.session_state.last_raw_input = raw_input
            st.session_state.last_weighted_input = weighted_input
            st.session_state.last_hidden_output = hidden_output
            st.session_state.last_prediction = prediction
            st.session_state.last_history = history

            # Save chat
            st.session_state.chat_log.append({
                "user": raw_input.tolist(),
                "weights": weights.tolist(),
                "decision": decision,
                "confidence": round(prediction, 2)
            })

# ---------- APP UI ----------
if st.session_state.chat_log:
    st.markdown("### 💬 Conversation History")
    for i, chat in enumerate(reversed(st.session_state.chat_log[-5:])):
        with st.chat_message("user"):
            st.markdown(f"**You:** {chat['user']}, weights: {chat['weights']}")
        with st.chat_message("assistant"):
            st.markdown(f"**{st.session_state.bot_name}:** {chat['decision']} (confidence: {chat['confidence']})")

    # If prediction was made, show explanation
    if st.session_state.last_raw_input is not None:
        st.markdown("### 🧠 Latest AI Reasoning")
        st.code(f"Hidden layer activations: {st.session_state.last_hidden_output}", language="python")
        st.markdown("**Weighted input vector:**")
        st.code(st.session_state.last_weighted_input.flatten())
        st.markdown("**Loss Curve:**")
        show_loss_chart(st.session_state.last_history)

        # feedback and training
        with st.form("teach_form"):
            teach = st.radio("Teach your AI?", ["No", "Yes"], key="teach_option")
            label = None
            if teach == "Yes":
                label = st.radio("What should the correct decision have been?", [decision_labels["go"], decision_labels["stay"]], key="true_label")
                submit_teach = st.form_submit_button("📚 Submit Training Feedback")
                if submit_teach:
                    label_binary = 1 if label == decision_labels["go"] else 0
                    st.session_state.user_X.append(st.session_state.last_raw_input.reshape(1, -1))
                    st.session_state.user_y.append(np.array([[label_binary]]))
                    st.success("✅ Added to training! Try again and see how it improves.")
            else:
                st.form_submit_button("📚 Submit")

# Footer
st.markdown("---")
