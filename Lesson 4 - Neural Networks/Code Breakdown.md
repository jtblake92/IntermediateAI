# 🏋️‍♂️ Streamlit App: GymBot – Your AI Gym Decision Coach

This project is an interactive Streamlit web app that simulates a neural network deciding **whether you should go to the gym** based on your mood, time, and how crowded the gym is. It's designed as a playful introduction to neural networks and reinforcement through feedback.

---

## 🧠 What It Does

The app trains a **neural network with 1 hidden layer** to make decisions like:

> "Should I go to the gym today?"

You input:
- How energised you feel
- How crowded the gym is
- How much time you have
- AND how important each of those are to you

Then the AI will:
- Show its thought process (hidden neurons + weighted input)
- Make a prediction
- Learn from your feedback if you disagree

---

## 🗂 Files Included

### `streamlit_app.py`
This is the full Streamlit app:
- Interactive sliders for input
- Sidebar customisation for labels and weights
- Neural network (3 input → 6 hidden → 1 output)
- Displays hidden layer activations and prediction confidence
- Lets users teach the AI via feedback

### `run_lab.py`
A simplified script for:
- Building and training the same neural network on predefined scenarios
- Plotting training loss
- Testing preset gym scenarios
- Good for unit tests or standalone experimentation

---

## ⚙️ How It Works

### Neural Network Architecture
```text
Inputs:     [energised, crowded, time]
Weights:    Your preferences
Hidden:     Dense(6, relu)
Output:     Dense(1, sigmoid) → decision
```

### Training Process
- Uses synthetic + curated + user-labeled data
- Trains in real-time on every submission
- Tracks training loss and displays it
- Learns progressively from user feedback

### User Features
- Name your AI (e.g. GymBot, CoachGPT)
- Rephrase input labels and decision outcomes
- Submit feedback to teach the AI what it *should* have said
- View session history and neural reasoning

---

## 📈 Visual Feedback

- Loss curve shows how training progressed
- Weighted inputs and hidden outputs displayed
- Confidence score on each prediction

---

## 🎯 Example Use Cases

- Show beginners how AI makes decisions
- Teach neural network basics via real-life analogies
- Introduce AI fine-tuning through reinforcement
- Explore decision boundaries interactively

---

## ✅ Summary

This app demonstrates:
- Real-time neural decision making
- Dynamic user-driven training
- Interpretable AI outputs
- Feedback loops in AI learning

> 🧠 A fun way to learn neural networks, feedback loops, and how decisions evolve with data!

---

## 🚀 Run It Yourself

To launch the app locally:

```bash
pip install streamlit tensorflow matplotlib
streamlit run streamlit_app.py
```

Or run `run_lab.py` to test the model in a script version.