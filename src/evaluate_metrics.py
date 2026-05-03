import json
import sys
from rouge_score import rouge_scorer

def parse_risk_label(metin: str) -> int:
    """
    Parses the LLM output to extract a binary risk label (0/1).
    Mapped levels:
      - HIGH / VERY HIGH / MEDIUM => 1
      - LOW => 0
    """
    if metin is None:
        return -1

    t = metin.lower()
    if any(x in t for x in ["cok yuksek risk", "cok yüksek risk", "yuksek risk", "yüksek risk", "orta risk"]):
        return 1
    if any(x in t for x in ["dusuk risk", "düşük risk"]):
        return 0

    # Fallback heuristic
    pos_keywords = ["siddetli", "kritik", "tehlikeli", "ciddi", "yuksek", "yüksek"]
    neg_keywords = ["dusuk", "düşük", "stabil", "guvenli", "güvenli", "minimal"]

    pos_found = any(x in t for x in pos_keywords)
    neg_found = any(x in t for x in neg_keywords)

    if pos_found and not neg_found: return 1
    if neg_found and not pos_found: return 0
    return 0

def check_factual_consistency(girdi: dict, cikti: str) -> tuple[float, list]:
    """
    Verifies if numerical input values are preserved in the model output.
    """
    if cikti is None:
        return 0.0, []

    values_to_check = []
    for category in girdi.values():
        for val in category.values():
            values_to_check.append(str(val))

    missing = []
    for val in values_to_check:
        if val in cikti:
            continue
        if val.endswith(".0") and val[:-2] in cikti:
            continue
        missing.append(val)

    score = ((len(values_to_check) - len(missing)) / len(values_to_check)) * 100 if values_to_check else 0.0
    return score, missing

def classification_metrics(y_true: list, y_pred: list) -> dict:
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred):
        if pred == -1: continue
        if   true == 1 and pred == 1: tp += 1
        elif true == 0 and pred == 1: fp += 1
        elif true == 0 and pred == 0: tn += 1
        elif true == 1 and pred == 0: fn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1  = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1, "Total": total}

def calculate_metrics(results_path: str, label: str = "Evaluation", verbose: bool = True):
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {results_path} not found.")
        return

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    y_true, y_pred, consistency_scores = [], [], []

    if verbose:
        print(f"\n{'='*60}\n{label.upper()}\n{'='*60}")

    for i, row in enumerate(data):
        input_data = row.get("Girdi_Verisi", {})
        output_text = row.get("Model_Ciktisi", None)
        true_label = row.get("Gercek_Yangin_Durumu", -1)
        pred_label = parse_risk_label(output_text)

        y_true.append(true_label)
        y_pred.append(pred_label)

        c_score, missing = check_factual_consistency(input_data, output_text)
        consistency_scores.append(c_score)

        if verbose:
            res_str = "CORRECT" if true_label == pred_label else "WRONG"
            print(f"Sample {i+1} (ID: {row.get('id')}): {res_str} | Consistency: {c_score:.1f}%")

    m = classification_metrics(y_true, y_pred)
    print(f"\n--- {label} Summary ---")
    print(f"Accuracy : {m['Accuracy']:.3f}")
    print(f"F1-Score : {m['F1']:.3f}")
    print(f"Recall   : {m['Recall']:.3f}")
    print(f"Avg Consistency: {sum(consistency_scores)/len(consistency_scores):.1f}%")
    print("-" * 30)

if __name__ == "__main__":
    show_details = "-s" not in sys.argv
    calculate_metrics("../outputs/llm_results.json", "Main Dataset", verbose=show_details)
    calculate_metrics("../outputs/edge_cases_results.json", "Edge Cases", verbose=show_details)