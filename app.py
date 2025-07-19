import gradio as gr
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

# Prediction function
def predict_clv(quantity, unit_price, country):
    try:
        input_df = pd.DataFrame({
            "Quantity": [quantity],
            "UnitPrice": [unit_price],
            "Country": [country]
        })

        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        if prediction is None or len(prediction) == 0:
            return "Prediction failed. No output from model."

        return f"Predicted CLV: ₹{round(float(prediction[0]), 2)}"

    except Exception as e:
        import traceback
        return f"Error:\n{traceback.format_exc()}"

# Gradio UI
clv_app = gr.Interface(
    fn=predict_clv,
    inputs=[
        gr.Number(label="Quantity"),
        gr.Number(label="Unit Price (₹)"),
        gr.Textbox(label="Country")
    ],
    outputs=gr.Textbox(label="Predicted CLV"),
    title="Customer Lifetime Value (CLV) Predictor",
    description="Enter Quantity, Unit Price, and Country to predict CLV."
)

# Run app
if __name__ == "__main__":
    clv_app.launch()
