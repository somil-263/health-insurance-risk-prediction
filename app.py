import gradio as gr
import joblib
import pandas as pd
import spaces

# Saved model aur feature list load karna
bundle = joblib.load("risk_model.joblib")
model = bundle["model"]
features = bundle["features"]

@spaces.GPU
def predict_risk(age, bmi, systolic_bp, ldl, hba1c, chronic_count, smoker, alcohol_freq):
    data = {col: 0 for col in features}
    data["age"] = age
    data["bmi"] = bmi
    data["systolic_bp"] = systolic_bp
    data["ldl"] = ldl
    data["hba1c"] = hba1c
    data["chronic_count"] = chronic_count

    if smoker == "Former" and "smoker_1" in data:
        data["smoker_1"] = 1
    elif smoker == "Current" and "smoker_2" in data:
        data["smoker_2"] = 1

    if alcohol_freq == "Occasional" and "alcohol_freq_1" in data:
        data["alcohol_freq_1"] = 1
    elif alcohol_freq == "Weekly" and "alcohol_freq_2" in data:
        data["alcohol_freq_2"] = 1
    elif alcohol_freq == "Daily" and "alcohol_freq_3" in data:
        data["alcohol_freq_3"] = 1

    df_input = pd.DataFrame([data])[features]
    score = float(model.predict(df_input)[0])
    score = max(0.0, min(1.0, score))

    category = "High Risk 🔴" if score >= 0.5 else "Low / Moderate Risk 🟢"
    return f"Predicted Risk Score: {score:.4f}  |  Status: {category}"

demo = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(18, 90, value=45, step=1, label="Age (Years)"),
        gr.Slider(15.0, 50.0, value=26.5, step=0.1, label="BMI"),
        gr.Slider(80, 200, value=120, step=1, label="Systolic Blood Pressure (mmHg)"),
        gr.Slider(50, 250, value=110, step=1, label="LDL Cholesterol (mg/dL)"),
        gr.Slider(4.0, 14.0, value=5.5, step=0.1, label="HbA1c (%)"),
        gr.Slider(0, 10, value=1, step=1, label="Chronic Conditions Count"),
        gr.Dropdown(["Never", "Former", "Current"], value="Never", label="Smoking Status"),
        gr.Dropdown(["None", "Occasional", "Weekly", "Daily"], value="None", label="Alcohol Frequency"),
    ],
    outputs=gr.Textbox(label="Underwriting Assessment Result"),
    title="Health Insurance Underwriting Risk Predictor",
    description="Predicts patient underwriting risk score using clinical vitals & lifestyle indicators (HistGradientBoostingRegressor)."
)

demo.launch()
