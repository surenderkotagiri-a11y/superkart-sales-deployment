from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback
import io

app = Flask(__name__)

# Load the unified pipeline (Preprocessor + Tuned Estimator)
MODEL_PATH = 'superkart_model.joblib'
try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"[*] Successfully loaded unified pipeline model from {MODEL_PATH}")
except Exception as e:
    pipeline = None
    print(f"[!] Error loading model artifact: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health-check endpoint."""
    status = "healthy" if pipeline is not None else "degraded"
    return jsonify({
        "status": status,
        "model_loaded": pipeline is not None
    }), 200

@app.route('/v1/predict', methods=['POST'])
def predict_single():
    """Accepts a single product/store payload and returns the predicted sales."""
    if pipeline is None:
        return jsonify({"error": "Model is not loaded on server."}), 503

    try:
        # Retrieve JSON input
        data = request.get_json(force=True)

        # Convert singular JSON object to DataFrame
        input_df = pd.DataFrame([data])

        # Run prediction directly through the unified pipeline
        prediction = pipeline.predict(input_df)[0]

        return jsonify({"predicted_sales": float(prediction)}), 200

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

@app.route('/v1/predictbatch', methods=['POST'])
def predict_batch():
    """Accepts a CSV file of inputs and returns index-mapped batch predictions."""
    if pipeline is None:
        return jsonify({"error": "Model is not loaded on server."}), 503

    try:
        # Get file from request files dictionary
        if 'file' not in request.files:
            return jsonify({"error": "No 'file' key found in the request files."}), 400

        file = request.files['file']
        # Read CSV bytes into a pandas DataFrame
        csv_data = file.read().decode('utf-8')
        input_df = pd.read_csv(io.StringIO(csv_data))

        # Predict batch data
        predictions = pipeline.predict(input_df)

        # Map index to predicted values for JSON return
        pred_map = {int(i): float(val) for i, val in enumerate(predictions)}
        return jsonify(pred_map), 200

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

if __name__ == '__main__':
    # Listen on port 7860 to match standard Codespaces container settings
    app.run(host='0.0.0.0', port=7860, debug=True)
