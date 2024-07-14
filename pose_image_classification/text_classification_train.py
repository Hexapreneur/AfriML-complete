from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
import os
import json
model_ckpt = os.path.join("static", "distilbert-base-uncased") # or path of the model for in local storage
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_ckpt = os.path.join("static", "distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
auto_model = AutoModel.from_pretrained(model_ckpt).to(device)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = auto_model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

# encode the labels for training
# LabelEncoder() helps to track ur labels, which is helpful for accuracies


def text_train(dataset_path):

    df = pd.read_csv(dataset_path)
    # df = df[:1000]

    df = df.dropna()

    labels_data = list(df['label'])
    str_elements = [str(element) for element in labels_data]
    print(str_elements)
    with open(os.path.join('static', 'train_text_classes.json'), 'w') as f:
        json.dump(list(np.unique(str_elements)), f)

    num_classes = len(np.unique(labels_data))

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    # split into train, validate, test
    train, validate, test = \
                  np.split(df.sample(frac=1, random_state=42),
                           [int(.6*len(df)), int(.8*len(df))])

    # reset indices
    train = train.reset_index()[['text', 'label']]
    test = test.reset_index()[['text', 'label']]
    validate = validate.reset_index()[['text', 'label']]

    # dataframes to datadict
    tds = Dataset.from_pandas(train)
    testds = Dataset.from_pandas(test)
    vds = Dataset.from_pandas(validate)
    dataset = DatasetDict()
    dataset['train'] = tds
    dataset['test'] = testds
    dataset['validation'] = vds
    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)
    dataset_encoded.set_format("torch",
                                columns=["input_ids", "attention_mask", "label"])

    dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True)

    # print(np.unique(dataset_hidden['label']))
    num_labels = num_classes
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

    batch_size = 4
    logging_steps = len(dataset_hidden["train"]) // batch_size
    txt_cls_model_path = os.path.join('static', 'text_classification_model')
    os.system(f'rm -r {model_ckpt}-finetuned-emotion')
    os.system(f'rm -r {txt_cls_model_path}')

    model_name = f"{model_ckpt}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=model_name,
                                      num_train_epochs=10,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      weight_decay=0.01,
                                      evaluation_strategy="epoch",
                                      disable_tqdm=False,
                                      logging_steps=logging_steps,
                                      push_to_hub=False,
                                      log_level="error")

    trainer = Trainer(model=model, args=training_args,
                      compute_metrics=compute_metrics,
                      train_dataset=dataset_hidden["train"],
                      eval_dataset=dataset_hidden["validation"],
                      tokenizer=tokenizer)

    trainer.train()

    trainer.save_model(txt_cls_model_path)
