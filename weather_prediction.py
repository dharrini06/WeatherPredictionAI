# ================================================================
# PROJECT: Predicting Weather Conditions Using Probabilistic Reasoning
# ================================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import logging

# ================================================================
# Step 1.1: Silence pgmpy INFO logs
# ================================================================
logging.getLogger("pgmpy").setLevel(logging.WARNING)

# ================================================================
# Step 2: Generate Synthetic Weather Dataset
# ================================================================
def generate_weather_data(n=2000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    data = []
    for _ in range(n):
        outlook = random.choices(['Sunny', 'Overcast', 'Rain'], [0.45, 0.25, 0.30])[0]
        temp = random.choices(['Hot', 'Mild', 'Cool'], [0.35, 0.45, 0.20])[0]
        humidity = random.choices(['High', 'Normal'], [0.55, 0.45])[0]
        wind = random.choices(['Weak', 'Strong'], [0.65, 0.35])[0]

        # Conditional logic
        if outlook == 'Sunny' and humidity == 'High':
            play = 'No'
        elif outlook == 'Rain' and wind == 'Strong':
            play = 'No'
        elif outlook == 'Overcast':
            play = 'Yes'
        elif temp == 'Cool' and humidity == 'Normal':
            play = 'Yes'
        else:
            play = random.choice(['Yes', 'No'])

        data.append([outlook, temp, humidity, wind, play])
    return pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play'])


weather_data = generate_weather_data()
print("Sample Weather Data:\n", weather_data.head())

# ================================================================
# Step 3: Encode categorical data for ML part (silences FutureWarning)
# ================================================================
data_encoded = weather_data.copy()
mapping = {
    'Sunny': 0, 'Overcast': 1, 'Rain': 2,
    'Hot': 0, 'Mild': 1, 'Cool': 2,
    'High': 0, 'Normal': 1,
    'Weak': 0, 'Strong': 1,
    'No': 0, 'Yes': 1
}
data_encoded = data_encoded.replace(mapping).infer_objects(copy=False)

# ================================================================
# Step 4: Split data for Naive Bayes comparison
# ================================================================
X = data_encoded[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = data_encoded['Play']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ================================================================
# Step 5: Train Naive Bayes Classifier
# ================================================================
nb_model = CategoricalNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

print("\n--- MACHINE LEARNING MODEL (Naive Bayes) ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ================================================================
# Step 6: Bayesian Network Model
# ================================================================
model = DiscreteBayesianNetwork([
    ('Outlook', 'Play'),
    ('Temperature', 'Play'),
    ('Humidity', 'Play'),
    ('Wind', 'Play')
])
model.fit(weather_data, estimator=MaximumLikelihoodEstimator)

# ================================================================
# Step 7: Inference Setup
# ================================================================
infer = VariableElimination(model)
print("\n--- PROBABILISTIC REASONING MODEL (Bayesian Network) ---")
print("Conditional Probability Table for 'Play':")
print(model.get_cpds('Play'))

# ================================================================
# Step 8: Visualize Bayesian Network Structure
# ================================================================
plt.figure(figsize=(7, 5))
G = nx.DiGraph()
G.add_edges_from(model.edges())
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=4000, node_color='lightblue', arrowsize=20)
plt.title("Bayesian Network Structure", fontsize=14)
plt.show()

# ================================================================
# Step 9: Function to Predict Using Bayesian Inference
# ================================================================
def predict_play(Outlook=None, Temperature=None, Humidity=None, Wind=None):
    evidence = {}
    if Outlook: evidence['Outlook'] = Outlook
    if Temperature: evidence['Temperature'] = Temperature
    if Humidity: evidence['Humidity'] = Humidity
    if Wind: evidence['Wind'] = Wind
    return infer.query(variables=['Play'], evidence=evidence, show_progress=False)

# ================================================================
# Step 10: Predict a Few Scenarios
# ================================================================
print("\n--- Bayesian Network Predictions ---")
scenarios = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'},
    {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Wind': 'Strong'},
    {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak'}
]

for i, s in enumerate(scenarios, 1):
    pred = predict_play(**s)
    print(f"🌤️ Scenario {i}: {s}")
    print(pred)
    print("-" * 60)

# ================================================================
# Step 11: Heatmap Visualization of CPT
# ================================================================
sns.heatmap(pd.DataFrame(model.get_cpds('Play').values), annot=True, cmap="YlGnBu")
plt.title("Conditional Probability Table - 'Play'")
plt.xlabel("Play States")
plt.ylabel("Probability Index")
plt.show()

# ================================================================
# Step 12: Save Outputs
# ================================================================
weather_data.to_csv("advanced_weather_data.csv", index=False)
print("✅ Data saved to 'advanced_weather_data.csv'")

# ================================================================
# END PROJECT CODE
# ================================================================
