"""
Convert fake_lstm.keras and sentiment_lstm.keras to ONNX format.
Run once: python convert_lstm_to_onnx.py
"""
import os
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MODELS = [
    ("fake_lstm.keras",      "fake_lstm.onnx",      100),
    ("sentiment_lstm.keras", "sentiment_lstm.onnx", 100),
]


def convert(keras_path, onnx_path, max_len):
    print(f"\nConverting {keras_path} → {onnx_path} ...")
    model = tf.keras.models.load_model(keras_path)

    # Save as SavedModel then convert — more reliable for Bidirectional LSTM
    saved_model_dir = keras_path.replace(".keras", "_saved_model_tmp")
    model.export(saved_model_dir)

    input_signature = [tf.TensorSpec([None, max_len], tf.int32, name="input_1")]
    onnx_model, _ = tf2onnx.convert.from_saved_model(
        saved_model_dir,
        input_names=["input_1"],
        output_names=None,
        opset=13,
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Saved → {onnx_path}")

    # Clean up temp SavedModel
    import shutil
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    # Quick sanity check
    sess  = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, max_len), dtype=np.int32)
    out   = sess.run(None, {"input_1": dummy})
    print(f"  Sanity check output shape: {out[0].shape}  value: {out[0][0][0]:.4f}")


if __name__ == "__main__":
    for keras_name, onnx_name, max_len in MODELS:
        keras_path = os.path.join(DATA_DIR, keras_name)
        onnx_path  = os.path.join(DATA_DIR, onnx_name)
        if not os.path.exists(keras_path):
            print(f"  SKIP — {keras_path} not found")
            continue
        convert(keras_path, onnx_path, max_len)

    print("\nDone.")
