"""Custom MLflow pyfunc wrapper around a Hugging Face sentiment model (Lab 9 style).
Student: Yashodha Kamble | USN: 1MS25CI149"""
import mlflow.pyfunc
import pandas as pd


LABEL_MAP = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}   # textattack models return LABEL_0/1


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with a `text` column. Output: DataFrame with text, label (POSITIVE/NEGATIVE) and score."""

    def load_context(self, context):
        from transformers import pipeline
        cfg = context.model_config or {}
        self.max_length = int(cfg.get("max_length", 128))
        self.batch_size = int(cfg.get("predict_batch_size", 8))
        self.clf = pipeline("text-classification", model=context.artifacts["hf_model_dir"], device="cpu")

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        res = self.clf(texts, truncation=True, max_length=self.max_length, batch_size=self.batch_size)
        return pd.DataFrame({"text": texts,
                             "prediction": [LABEL_MAP.get(r["label"].upper(), r["label"].upper()) for r in res],
                             "score": [round(float(r["score"]), 4) for r in res]})
