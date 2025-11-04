import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from tkinter import *
from tkinter import messagebox

# Load dataset
data = pd.read_csv("weather.csv")
data.columns = data.columns.str.strip()

# Encode categorical columns
encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Split data
X = data[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = data['Play']

# Train model
model = CategoricalNB()
model.fit(X, y)

# GUI setup
root = Tk()
root.title("🌤 Weather Prediction using Naïve Bayes")
root.geometry("480x480")
root.config(bg="#E8F0F2")

Label(root, text="🌦 Weather Prediction System",
      font=("Segoe UI", 18, "bold"), bg="#E8F0F2", fg="#2C3E50").pack(pady=20)

frame = Frame(root, bg="#E8F0F2")
frame.pack(pady=10)

# Dropdown options
outlook_options = ["Sunny", "Overcast", "Rainy"]
temperature_options = ["Hot", "Mild", "Cool"]
humidity_options = ["High", "Normal"]
windy_options = ["False", "True"]

# Default placeholder text
outlook_var = StringVar(value="Choose an option")
temp_var = StringVar(value="Choose an option")
humidity_var = StringVar(value="Choose an option")
windy_var = StringVar(value="Choose an option")

def create_dropdown(label_text, variable, options):
    Label(frame, text=label_text, bg="#E8F0F2", fg="#34495E",
          font=("Segoe UI", 11, "bold")).pack(pady=5)
    om = OptionMenu(frame, variable, *options)
    om.config(bg="#D1E7DD", fg="#1B4332", font=("Segoe UI", 10, "bold"),
              activebackground="#95D5B2", width=15, borderwidth=2, relief=RIDGE)
    om.pack(pady=2)

# Create dropdowns
create_dropdown("Outlook:", outlook_var, outlook_options)
create_dropdown("Temperature:", temp_var, temperature_options)
create_dropdown("Humidity:", humidity_var, humidity_options)
create_dropdown("Windy:", windy_var, windy_options)

# Prediction
def predict_weather():
    if ("Choose an option" in [outlook_var.get(), temp_var.get(), humidity_var.get(), windy_var.get()]):
        messagebox.showwarning("⚠️ Input Missing", "Please select all weather conditions before predicting.")
        return

    input_data = pd.DataFrame({
        'Outlook': [outlook_var.get()],
        'Temperature': [temp_var.get()],
        'Humidity': [humidity_var.get()],
        'Windy': [windy_var.get()]
    })

    for col in input_data.columns:
        input_data[col] = encoders[col].transform(input_data[col])

    pred = model.predict(input_data)[0]
    result = "✅ Play (Good Weather)" if pred == 1 else "❌ Don’t Play (Bad Weather)"
    messagebox.showinfo("🌤 Prediction Result", f"Predicted Result:\n\n{result}")

# Predict button
Button(root, text="🔍 Predict Weather", command=predict_weather,
       bg="#2E8B57", fg="white", font=("Segoe UI", 12, "bold"),
       activebackground="#3CB371", relief="ridge", padx=10, pady=5,
       cursor="hand2").pack(pady=25)

Label(root, text="Developed using Python • Naïve Bayes • Tkinter",
      bg="#E8F0F2", fg="#5F6A6A", font=("Segoe UI", 9, "italic")).pack(side=BOTTOM, pady=10)

root.mainloop()
