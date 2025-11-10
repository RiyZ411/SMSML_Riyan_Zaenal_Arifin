from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
from prometheus_client import (
    start_http_server, Counter, Histogram, Gauge, Summary
)
import time

# Load model dari MLflow
model = mlflow.pyfunc.load_model("Monitoring dan Logging/exported_model")

# Inisialisasi metrik Prometheus
PREDICT_COUNTER = Counter('predict_requests_total', 'Total prediction requests')
PREDICT_ERRORS = Counter('predict_errors_total', 'Total prediction errors')
REQUEST_LATENCY = Histogram('predict_request_duration_seconds', 'Latency of prediction requests')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of current active requests')
PAYLOAD_SIZE = Histogram('payload_size_bytes', 'Payload size of prediction requests')
SUCCESSFUL_RESPONSES = Counter('predict_success_total', 'Total successful predictions')
MODEL_VERSION = Gauge('model_version_info', 'Model version being served')
AVG_INPUT_FEATURE = Summary('avg_input_feature', 'Average of the first input feature')
LAST_PREDICTION_VALUE = Gauge('last_prediction_value', 'Last prediction output')
TOTAL_BATCH_SIZE = Counter('total_batch_size', 'Total items predicted in batch')

MODEL_VERSION.set(1.0)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time()
def predict():
    ACTIVE_REQUESTS.inc()
    try:
        PREDICT_COUNTER.inc()
        data_json = request.get_json()
        PAYLOAD_SIZE.observe(len(str(data_json)))

        # Ubah ke DataFrame
        df = pd.DataFrame(data_json["data"], columns=data_json["columns"])

        # Prediksi
        predictions = model.predict(df)

        batch_size = len(df)
        TOTAL_BATCH_SIZE.inc(batch_size)
        SUCCESSFUL_RESPONSES.inc()

        try:
            pred_value = predictions[0]
            if isinstance(pred_value, (int, float)):
                LAST_PREDICTION_VALUE.set(float(pred_value))
        except:
            pass

        try:
            first_value = df.iloc[0, 0]
            if isinstance(first_value, (int, float)):
                AVG_INPUT_FEATURE.observe(float(first_value))
        except:
            pass

        return jsonify(predictions.tolist())

    except Exception as e:
        PREDICT_ERRORS.inc()
        return jsonify({'error': str(e)}), 500

    finally:
        ACTIVE_REQUESTS.dec()

if __name__ == '__main__':
    start_http_server(8001)
    app.run(port=5000)