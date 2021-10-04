# Shanduan: Classical Chinese Sentence Segmentation with BERT

## Report

See [final_report.pdf](https://github.com/Rabbit-Hu/shanduan/blob/main/final_report.pdf) for details about this project.

## Demo

```bash
python demo.py
```

## Train (Fine-tune)

```bash
python utils/preprocess.py # This is to prepare data in json files. You must run this code at least once.
python train.py
```

## Evaluation
```bash
python evaluate.py
```
