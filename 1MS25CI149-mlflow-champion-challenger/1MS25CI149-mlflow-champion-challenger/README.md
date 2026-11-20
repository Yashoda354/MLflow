# 1MS25CI149 – Champion–Challenger Model Comparison with Hugging Face and MLflow
**Student:** Yashodha Kamble · **USN:** 1MS25CI149 · **Section:** B · Skill Lab-1 (CISL-1)

## 1. Task and models
| Item | Value |
|---|---|
| Task | Binary sentiment classification (POSITIVE / NEGATIVE) |
| Dataset | SST-2 validation split (`stanfordnlp/sst2`, Hugging Face Hub) |
| Subset | 200 sentences: 100 negative and 100 positive, seed 149 → `data/eval_inputs.csv` |
| Model A – current Champion | `distilbert-base-uncased-finetuned-sst-2-english` (~67M, table row 1) |
| Model B – Challenger | `textattack/bert-base-uncased-SST-2` (~110M, table row 3) |
| Inference settings | Hugging Face `pipeline("text-classification")`, CPU, max_length 128, truncation on, batch_size 1 (so latency is per sentence) |

Pairing "1 vs 3" is one of the suggested classification pairings in the brief. Model B returns `LABEL_0` / `LABEL_1`, so both models are mapped to `NEGATIVE` / `POSITIVE` (`LABEL_MAP` in `evaluate.py` and `hf_wrapper.py`) before any metric is computed.

## 2. How to run
```bash
pip install -r requirements.txt
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
# in a second terminal:
python evaluate.py                        # runs A and B on the same 200 sentences, logs everything, makes the decision
python register_champion.py               # registers both versions, sets @champion / @challenger aliases
python predict_from_mlflow.py             # loads models:/1MS25CI149-sentiment-model@champion and classifies data/new_data.csv
python predict_from_mlflow.py challenger  # stretch goal: same script, other alias, no code change
python register_champion.py --swap        # optional: flip the two aliases, then rerun predict_from_mlflow.py unchanged
```
MLflow experiment: `1MS25CI149-hf-sentiment-champion-challenger` · Registered model: `1MS25CI149-sentiment-model`
(To use another tracking server, set `MLFLOW_TRACKING_URI`. No run IDs are hard-coded.)

## 3. Champion rule (written before the final run)
- **Primary metric:** macro-F1 (%) on the shared 200-sentence set. **Higher is better.**
- **Margin:** B must beat A by **at least 2.0 macro-F1 points**. With 200 sentences one sentence is worth 0.5 points, so 2 points means at least 4 more sentences classified correctly. Anything smaller could come from a handful of borderline sentences.
- **Guardrail:** B is rejected if its average latency is **more than 10×** A's. **Lower latency is better.**
- **Tie-breaker:** if |Δ macro-F1| < 2.0, keep the **faster** model.
- Secondary metrics (reported, not used for the decision): accuracy ↑, precision (positive) ↑, recall (positive) ↑, recall (negative) ↑, total inference time ↓.

The README logged as an artifact in the `compare_sentiment` run is the version that existed when `evaluate.py` ran (rule only, no results), which shows the rule was fixed before the run.

## 4. Results
| Metric | A: DistilBERT-SST2 | B: BERT-base-SST2 | B − A |
|---|---:|---:|---:|
| Accuracy ↑ | 91.00 | 90.50 | −0.50 |
| **Macro-F1 ↑ (primary)** | **91.00** | **90.50** | **−0.50** |
| Precision (positive) ↑ | 91.00 | 90.91 | −0.09 |
| Recall (positive) ↑ | 91.00 | 90.00 | −1.00 |
| Recall (negative) ↑ | 91.00 | 91.00 | 0.00 |
| Avg latency (ms/sentence) ↓ | 53.81 | 85.10 | 1.58× |
| Total inference (s) ↓ | 10.76 | 17.02 | +6.26 |
| Parameters | 67.0M | 109.5M | +42.5M |

A gets 182 of 200 right, B gets 181. They disagree on 11 sentences (`outputs/disagreements.csv`): A is right on 6 of them and B on 5. Seven are commented in `outputs/qualitative_examples.csv`.

Plots: `outputs/metric_comparison.png` (every metric + latency), `outputs/confusion_matrices.png` (both models side by side), `outputs/confidence_distribution.png` (confidence of correct vs wrong predictions).

## 5. Decision – Champion stays: Model A (`distilbert-base-uncased-finetuned-sst-2-english`)
The Challenger does not improve on the Champion. Its macro-F1 is 90.50 against 91.00, so it is actually 0.50 points worse, which is exactly one sentence out of 200. That is far below the 2.0-point margin fixed before the run. The latency guardrail does not decide anything here, because BERT-base is only 1.58× slower, well inside the 10× limit. The tie-breaker does: since |Δ macro-F1| < 2.0, the faster model is kept, and DistilBERT answers in about 54 ms per sentence against 85 ms. Model A therefore keeps the `@champion` alias (version 1) and BERT-base is registered as version 2 with `@challenger`.

The difference is not meaningful at this sample size. The two models disagree on only 11 sentences and split them almost evenly (6 to A, 5 to B). Both make most of their errors on sarcasm and mixed sentences such as "hilariously inept and ridiculous" or "positively anti-darwinian", and both are often very confident when they are wrong (see the red bars near 1.0 in the confidence plot).

Trade-offs: BERT-base has about 1.6× the parameters (110M vs 67M) and is about 1.6× slower on CPU, for no gain in accuracy. DistilBERT is smaller, faster and at least as accurate on this data, so it remains the better choice for deployment. On the 8 new sentences, the Champion got all 8 right and the Challenger 7 (it called new_005, a positive library review, negative at 0.57 confidence).

## 6. Limitations and declaration
- Only 200 SST-2 sentences (short movie-review phrases), so ±2 macro-F1 points is within noise. Results may differ on other domains such as tweets or product reviews.
- Both models were fine-tuned on the SST-2 training split, so this measures in-domain performance. The evaluation uses the validation split, which neither model was trained on.
- Latency was measured on a 2-core CPU with batch size 1 and varies between runs. The ratio is more reliable than the absolute milliseconds.
- `predict_from_mlflow.py` never calls Hugging Face directly. It loads the model through the alias URI, and the pyfunc wrapper (`hf_wrapper.py`) calls transformers inside MLflow. The version and HF model ID are read from the registry tags.
- New data (`data/new_data.csv`, 8 sentences about campus life, including sarcasm in new_004, negation in new_007 and new_008, and a mixed review in new_006) was written by me and is not from SST-2.
- AI assistant used: Hark (AI assistant) helped write and debug the scripts and README. I can explain every line.

## 7. Repository structure
```
data/        eval_inputs.csv, new_data.csv
outputs/     predictions_*.csv, metrics_summary*.json, model_comparison.csv, disagreements.csv, qualitative_examples.csv,
             metric_comparison.png, confusion_matrices.png, confusion_matrix_*.png, confidence_distribution.png,
             new_data_predictions*.csv, model_runs.json, *_terminal.txt
screenshots/ MLflow runs, compare view, summary run, registry aliases, predict_from_mlflow.py and register terminal output
evaluate.py, hf_wrapper.py, register_champion.py, predict_from_mlflow.py, requirements.txt, .gitignore
```
