from gym_decision_neuralnetwork import create_dataset, build_model, plot_loss, test_model

# Step 1: Load data
X, y = create_dataset()

# Step 2: Build model
model = build_model(input_dim=X.shape[1])

# Step 3: Train model
history = model.fit(X, y, epochs=200, verbose=0)

# Step 4: Visualize loss
plot_loss(history)

# Step 5: Try test scenarios
test_model(model)
