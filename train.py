# torch
import torch
from torch.utils.data import DataLoader
# huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
from datasets import load_dataset
# miscellaneous
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
# mine
from dataset import PuncDataset
from evaluate import evaluate, get_confusion_matrix, get_F1_from_confusion_matrix, get_accuracy_from_confusion_matrix, get_class_accuracy_from_confusion_matrix, get_mF1_from_confusion_matrix


def plot_curve(train_losses, val_losses, train_scores, val_scores, metric_name, save_path):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    x_range = range(1, len(train_losses) + 1)

    ax1.plot(x_range, train_losses, label='train loss', color='xkcd:light blue')
    ax2.plot(x_range, train_scores, label='train ' + metric_name, color='xkcd:light purple')

    ax1.plot(x_range, val_losses, label='val loss', color='xkcd:blue')
    ax2.plot(x_range, val_scores, label='val ' + metric_name, color='xkcd:violet')

    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax1.grid()
    ax1.set_xlabel("Epoch")
    plt.savefig(save_path)
    plt.close()


def main():

    ########## manually set these params ##########

    model_id = "10_punc_type_mF1" # NOTE! Change this avoid overwriting existing logs
    task_type = "punc_type" # "punc" or "punc_type"
    model_id = "11_punc_F1"
    task_type = "punc"

    batch_size = 4
    num_epochs = 50
    lr = 1e-5

    ###############################################

    ''' create log dirs '''
    log_dir = f"./training_logs/model_{model_id}"
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # loss_weight = torch.tensor([1., 5.]).to(device) if task_type == "punc" else None 
    if task_type == "punc":
        # loss_weight = 1. / torch.tensor([687889., 135750.], device=device)
        loss_weight = None
        num_labels = 2
    elif task_type == "punc_type":
        # loss_weight = 1. / torch.tensor([687889., 84318., 33464., 5900., 1868., 1259., 7350., 1591.], device=device)
        loss_weight = None
        num_labels = 8
    else:
        raise raiseNotImplementedError()
    
    # prepare model
    model = AutoModelForTokenClassification.from_pretrained("ethanyt/guwenbert-base", num_labels=num_labels)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")

    # prepare data
    data_root = "./data/json_files/" 
    raw_dataset = load_dataset('json', data_files={
        'train': os.path.join(data_root, 'train.json'), 
        'val': os.path.join(data_root, 'val.json')}, field='data')
    train_dataset = PuncDataset(raw_dataset['train'], tokenizer)
    val_dataset = PuncDataset(raw_dataset['val'], tokenizer)
    # val_dataset = train_dataset # TODO: delete this line!
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn)

    # prepare training
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    print("num_training_steps =", num_training_steps)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps)
    criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    
    # training loop
    model.train()

    train_losses, val_losses = [], []
    train_scores, val_scores = [], []

    for epoch in range(1, num_epochs + 1):
        epoch_loss_sum = 0
        confusion_matrix = torch.zeros((num_labels, num_labels))

        with tqdm(train_loader) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # print(batch['input_ids'], batch['puncs'])
                # exit()

                optimizer.zero_grad()
                
                inputs = {k:batch[k].to(device) for k in ('input_ids', 'attention_mask', 'token_type_ids')}
                label = batch[task_type + 's'].to(device)

                logits = model(**inputs)['logits']
                
                loss = criterion(logits.reshape(-1, num_labels), label.reshape(-1))
                # print("loss = ", loss)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss_sum += loss.item()
                mean_loss = epoch_loss_sum / (batch_idx + 1)
            
                pred = logits.argmax(dim=-1)
                metric_mask = (inputs['input_ids'] >= 3) # not 0([CLS]), 1([PAD]), or 2([SEP])
                confusion_matrix += get_confusion_matrix(pred, label, metric_mask, num_labels)
                accuracy = get_accuracy_from_confusion_matrix(confusion_matrix)

                if task_type == 'punc':
                    F1, precision, recall = get_F1_from_confusion_matrix(confusion_matrix, return_all=True)
                else:
                    F1, precision, recall = get_mF1_from_confusion_matrix(confusion_matrix, return_all=True)
                pbar.set_description("Epoch[%d/%d] lr=%.2e loss=%.4f prec=%.4f rec=%.4f acc=%.4f F1=%.4f" % \
                    (epoch, num_epochs, scheduler.get_last_lr()[0], mean_loss, precision, recall, accuracy, F1))  

            
            torch.save(model, os.path.join(checkpoint_dir, f'checkpoint{epoch}.pth'))

            val_results = evaluate(model, val_loader, criterion, task_type=task_type)
            model.train()
            
            train_losses.append(mean_loss)
            val_losses.append(val_results['loss'])
            train_scores.append(F1)
            val_scores.append(val_results['F1'])

            if max(val_scores) == val_scores[-1]:
                print(f"Best model at epoch {epoch}, saving best.pth...")
                torch.save(model, os.path.join(checkpoint_dir, f'best.pth'))

            curve_data = {
                'train_losses': train_losses, 
                'val_losses': val_losses, 
                'train_scores': train_scores,
                'val_scores': val_scores, 
            }

            torch.save(curve_data, os.path.join(log_dir, 'curve_data.pth'))
            plot_curve(**curve_data, metric_name="F1", save_path=os.path.join(log_dir, 'curve.png'))
            

if __name__ == '__main__':
    main()
