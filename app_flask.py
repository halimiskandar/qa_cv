
from flask import Flask, request, jsonify
import uuid
import time

app = Flask(__name__)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()

    inference_id = str(uuid.uuid4())

    product_id = request.form.get("product_id")
    stage_key = request.form.get("stage_key")

    images = request.files.getlist("images[]")

    if not product_id or not stage_key or len(images) == 0:
        return jsonify({
            "error": "INVALID_REQUEST",
            "message": "missing product_id, stage_key, or images[]"
        }), 400

    # placeholder inference
    results = [
        {
            "label": "READY_TO_SEND",
            "confidence": 0.91
        }
    ]

    response = {
        "inference_id": inference_id,
        "results": results,
        "aggregate_label": "READY_TO_SEND",
        "aggregate_confidence": 0.91,
        "inference_ms": round((time.time() - start) * 1000, 2),
        "model_version": "v1.0.0",
        "raw_debug_payload": {
            "content_sent": True
        }
    }

    # placeholder logging
    print({
        "inference_id": inference_id,
        "content_sent": response
    })

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
