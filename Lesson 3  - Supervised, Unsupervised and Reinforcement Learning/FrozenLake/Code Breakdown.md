# ❄️ FrozenLake Q-Learning – Undeveloped vs Developed AI Agents

This project explores **Q-learning reinforcement learning** using the `FrozenLake-v1` environment from OpenAI Gym. You'll experience the difference between an **undeveloped agent** (trained briefly) and a **developed agent** (trained over many episodes).

---

## 🎮 What is FrozenLake?

FrozenLake is a simple 4x4 grid world where an agent must learn to reach a goal without falling into holes. It's a perfect sandbox to understand:

- States and actions
- Rewards
- Exploration vs. exploitation
- Q-table learning

---

## 📂 Project Breakdown

We have two scripts:

### 1. `frozenlake_qlearning.py` – *Undeveloped Agent*
- Trains over **5,000 episodes** (relatively short)
- Q-table values are still incomplete in early episodes
- Performance is inconsistent
- May often fail during test simulations
- Lower confidence in state-action decisions

### 2. `frozenlake_trained.py` – *Developed Agent*
- Trains over **10,000+ episodes**
- Q-values are refined for most states
- Success rate increases significantly
- Exploits learned strategies more effectively
- Learns loop-breaking and adapts over time

---

## 🧠 Q-Learning Recap

The agent learns via trial-and-error by updating a **Q-table**, where:

```python
Q[state, action] += learning_rate * (reward + discount * max(Q[new_state]) - Q[state, action])
```

- `epsilon` controls exploration rate
- Rewards are sparse (1 for success, 0 otherwise)

---

## 🎥 Video Recording

Both scripts support video recording using `RecordVideo`:

- Episodes **0**, **halfway**, and **final** are recorded in `./videos/`
- These show agent progress over time
- You can compare early vs. late behaviour visually

---

## 📊 Rewards Plot

Each script also plots a **moving average** of reward per episode:

- Helps you visualise learning progress
- Stable increase means successful learning

---

## 🧪 Simulation Differences

### Undeveloped Agent (Low Training)
- High exploration rate (`epsilon ≈ 1`)
- Q-table has low confidence
- Often loops or fails to reach goal
- Video: Agent appears lost or random

### Developed Agent (High Training)
- Lower exploration rate (`epsilon ↓`)
- Refined Q-table with higher certainty
- Faster to reach goal, avoids holes
- Video: Agent moves confidently with fewer steps

---

## 📝 Summary

| Aspect            | Undeveloped Agent        | Developed Agent            |
|-------------------|--------------------------|-----------------------------|
| Training Episodes | ~5,000                   | ~10,000+                    |
| Exploration       | High                     | Low (decayed epsilon)       |
| Q-table           | Sparse / Unreliable      | Well-learned / Stable       |
| Performance       | Inconsistent              | High success rate           |
| Use Case          | Demonstrate early learning | Showcase mature policy    |

---

## 🧩 Try It Yourself

- Change number of episodes to see how it affects learning
- Try with `is_slippery=True` for a harder challenge
- Modify reward structure or grid layout
- Visualise Q-table using heatmaps

---

> ✨ Watching your AI agent go from "random walker" to "goal getter" is one of the most satisfying parts of RL!