# torch
import torch
from torch.utils.data import DataLoader
# huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
from datasets import load_dataset
# miscellaneous
from tqdm import tqdm
import os
# mine
from dataset import PuncDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_confusion_matrix(pred, label, mask, num_labels):
    out = torch.empty((num_labels, num_labels))
    for pred_i in range(num_labels):
        for label_i in range(num_labels):
            out[pred_i][label_i] = torch.logical_and(torch.logical_and(pred == pred_i, label == label_i), mask).sum().item()
    return out


# def get_F1_from_confusion_matrix_arabic(matrix, return_all=False):
#     # if matrix.size() != (2, 2):
#     #     print(f"Warning: Confution matrix has size {matrix.size()}, expected (2, 2). Returning None.")
#     #     return None
#     # TN, FN, FP, TP = matrix.reshape((-1,))

#     TN = matrix[0, 0]
#     TP = matrix.diag().sum() - TN
#     FN = matrix[0, :].sum() - TN
#     FP = matrix.sum() - (TN + TP + FN)

#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     F1 = 2 * precision * recall / (precision + recall)
#     if return_all:
#         return F1, precision, recall
#     return F1

def get_F1_from_confusion_matrix(matrix, positive_label=1, return_all=False):
    epsilon = 1e-6

    TP = matrix[positive_label, positive_label]
    FP = matrix[positive_label, :].sum() - TP
    FN = matrix[:, positive_label].sum() - TP
    TN = matrix.sum() - (TP + FP + FN)

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    F1 = 2 * precision * recall / (precision + recall + epsilon)

    if return_all:
        return F1, precision, recall
    return F1


def get_mF1_from_confusion_matrix(matrix, return_list=False, return_all=False):
    num_labels = matrix.size(0)
    F1_list, pre_list, rec_list = [], [], []
    for label in range(num_labels):
        F1, precision, recall = get_F1_from_confusion_matrix(matrix, label, return_all=True)
        F1_list.append(F1)
        pre_list.append(precision)
        rec_list.append(recall)

    if return_list:
        if return_all:
            return F1_list, pre_list, rec_list
        return F1_list
    if return_all:
        return sum(F1_list) / num_labels, sum(pre_list) / num_labels, sum(rec_list) / num_labels
    return sum(F1_list) / num_labels


def get_accuracy_from_confusion_matrix(matrix):
    return matrix.diag().sum() / matrix.sum()


def get_class_accuracy_from_confusion_matrix(matrix):
    return matrix.diag() / matrix.sum(axis=0)


def print_confusion_matrix(confusion_matrix):
    num_labels = confusion_matrix.size(0)
    title = "Confufion Matrix"
    separation_line = "=" * ((7 * num_labels + 5 - len(title) - 2) // 2)
    print(separation_line + " " + title + " " + separation_line)
    output_matrix = confusion_matrix / confusion_matrix.sum()
    print("       ", end='')
    for j in range(num_labels):
        print("%5d  " % j, end='')
    print()
    for i in range(num_labels):
        print("%5d  " % i, end='')
        for j in range(num_labels):
            print("%.3f  " % output_matrix[i][j], end='')
        print()
    print()


def evaluate(model, loader, criterion, task_type = "punc"):
    num_labels = 2 if task_type == "punc" else 8

    model.eval()
    with torch.no_grad():
        loss_sum = 0
        confusion_matrix = torch.zeros((num_labels, num_labels))

        with tqdm(loader) as pbar:
            for batch_idx, batch in enumerate(pbar):
                inputs = {k:batch[k].to(device) for k in ('input_ids', 'attention_mask', 'token_type_ids')}
                label = batch[task_type + 's'].to(device)

                logits = model(**inputs)['logits']
                loss = criterion(logits.reshape(-1, num_labels), label.reshape(-1))

                loss_sum += loss.item()
                mean_loss = loss_sum / (batch_idx + 1)

                pred = logits.argmax(dim=-1)
                metric_mask = (inputs['input_ids'] >= 3) # not 0([CLS]), 1([PAD]), or 2([SEP])
                confusion_matrix += get_confusion_matrix(pred, label, metric_mask, num_labels)
                accuracy = get_accuracy_from_confusion_matrix(confusion_matrix)

                if task_type == 'punc':
                    F1, precision, recall = get_F1_from_confusion_matrix(confusion_matrix, return_all=True)
                else:
                    F1, precision, recall = get_mF1_from_confusion_matrix(confusion_matrix, return_all=True)
                pbar.set_description("Evaluating... loss=%.4f prec=%.4f rec=%.4f acc=%.4f F1=%.4f" % \
                    (mean_loss, precision, recall, accuracy, F1))

    print_confusion_matrix(confusion_matrix)
    # F1_list, pre_list, rec_list = get_mF1_from_confusion_matrix(confusion_matrix, return_list=True, return_all=True)
    # print(f"F1_list = {F1_list}\npre_list={pre_list}\nrec_list={rec_list}")
    # print("F1_list:\n" + "\n".join(["%.1f" % (100 * F1_list[i].item()) for i in range(len(F1_list))]))
    # print("pre_list:\n" + "\n".join(["%.1f" % (100 * pre_list[i].item()) for i in range(len(pre_list))]))
    # print("rec_list:\n" + "\n".join(["%.1f" % (100 * rec_list[i].item()) for i in range(len(rec_list))]))
    # print(confusion_matrix)
    print("Validation done: loss=%.4f prec=%.4f rec=%.4f acc=%.4f F1=%.4f" % \
        (mean_loss, precision, recall, accuracy, F1))
    
    return {
        'loss': mean_loss,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'F1': F1,
        'confusion_matrix': confusion_matrix,
    }


if __name__ == '__main__':

    # task_type = "punc_type"
    # model = torch.load('training_logs/model_10_punc_type_mF1/checkpoints/best.pth')
    task_type = "punc"
    model = torch.load('./training_logs/model_11_punc_F1/checkpoints/best.pth')
    tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
    batch_size = 16

    # prepare data
    data_root = "./data/json_files/"
    raw_dataset = load_dataset('json', data_files={'val': os.path.join(data_root, 'val.json')}, field='data')
    val_dataset = PuncDataset(raw_dataset['val'], tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)

    if task_type == "punc":
        # loss_weight = 1. / torch.tensor([687889., 135750.], device=device)
        loss_weight = None
        num_labels = 2
    elif task_type == "punc_type":
        # loss_weight = 1. / torch.tensor([687889., 84318., 33464., 5900., 1868., 1259., 7350., 1591.], device=device)
        loss_weight = None
        num_labels = 8
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    evaluate(model, val_loader, criterion, task_type)