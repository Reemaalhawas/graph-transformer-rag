import re

import numpy as np
import pandas as pd


def compute_intermediate_metrics(correct_nodes: dict[str, list[int]], predicted_nodes: dict[str, np.ndarray]):
    precisions = []
    recalls = []
    hit_at_1 = []
    num_guesses = []
    f1s = []

    ids = correct_nodes.keys()
    all_correct_nodes = [correct_nodes[id] for id in ids]
    all_predicted_nodes = [predicted_nodes.get(id, []) for id in ids]

    for predicted_nodes, correct_nodes in zip(all_predicted_nodes, all_correct_nodes):

        num_correct_predictions = len(set(predicted_nodes).intersection(set(correct_nodes)))

        if num_correct_predictions == 0:
            precisions.append(0)
            recalls.append(0)
            hit_at_1.append(0)
            num_guesses.append(len(predicted_nodes))
            f1s.append(0)
        else:
            prec = num_correct_predictions / len(set(predicted_nodes))
            precisions.append(prec)
            rec = num_correct_predictions / len(set(correct_nodes))
            recalls.append(rec)
            f1s.append(2 / (1 / prec + 1 / rec))
            hit_at_1.append(1 * (predicted_nodes[
                                     0] in correct_nodes))  # only makes sense if ordered (compare hist@first vs hits@last)
            num_guesses.append(len(predicted_nodes))

    print(f"F1:              {np.mean(f1s)}")
    print(f"Precision:       {np.mean(precisions)}")
    print(f"Recall:          {np.mean(recalls)}")
    print(f"Exact hit@1:     {np.mean(hit_at_1)}")
    print(f"Exact hit@any:   {np.mean(np.array(precisions) != 0)}")
    print(f"Num predictions: {np.mean(num_guesses)}")

def compute_metrics(eval_output, skip_invalid_hit=True):
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    all_hit = []
    all_exact_hit = []
    all_exact_hit_at_any = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_num_preds = []
    for pred, label in zip(df.pred.tolist(), df.label.tolist()):
        pred = pred.split('[/s]')[0].strip().split('|')
        try:
            hit = re.findall(pred[0], label)
        except Exception as e:
            print(f'Label: {label}')
            print(f'Pred: {pred}')
            print(f'Exception: {e}')
            print('------------------')
            if skip_invalid_hit:
                continue
            else:
                hit = []

        all_hit.append(len(hit) > 0)

        label = label.split('|')
        exact_hit = 1 * (pred[0] in label)
        matches = set(pred).intersection(set(label))
        precision = len(matches) / len(set(pred))
        recall = len(matches) / len(set(label))
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        all_exact_hit.append(exact_hit)
        all_exact_hit_at_any.append(1 * (precision > 0))
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_num_preds.append(len(pred))

    dataset_len = len(df.label.tolist())
    hit = sum(all_hit) / dataset_len
    exact_hit = sum(all_exact_hit) / dataset_len
    exact_hit_at_any = sum(all_exact_hit_at_any) / dataset_len
    precision = sum(all_precision) / dataset_len
    recall = sum(all_recall) / dataset_len
    f1 = sum(all_f1) / dataset_len
    num_preds = sum(all_num_preds) / dataset_len

    print(f'F1:              {f1:.4f}')
    print(f'Precision:       {precision:.4f}')
    print(f'Recall:          {recall:.4f}')
    print(f'Substring hit@1: {hit:.4f}')
    print(f'Exact hit@1:     {exact_hit:.4f}')
    print(f'Exact hit@any:   {exact_hit_at_any:.4f}')
    print(f'Num predictions: {num_preds:.4f}')

