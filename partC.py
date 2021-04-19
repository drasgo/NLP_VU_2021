from typing import Tuple
import pandas as pd
import csv
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import time
import random
import datetime


MAX_LEN = 264
batch_size = 32
epochs = 4
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def check_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        dev = torch.device("cpu")
    return dev


def plot(loss_values):
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    plt.plot(loss_values, 'b-o')

    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labs):
    """Function to calculate the accuracy of our predictions vs labels"""
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labs.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def encode_vector(original_input, token) -> list:
    encoded = []
    for tweet in original_input:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = token.encode(
            tweet,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )
        encoded.append(encoded_sent)
    return encoded


def attention_mask(input_data) -> list:
    attention = []
    for sent in input_data:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        attention.append(att_mask)
    return attention


def prepare_dataset(inputs, labs, attentions) -> Tuple[DataLoader, DataLoader]:
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(inputs, labs,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attentions, labs,
                                                           random_state=2018, test_size=0.1)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(torch.tensor(validation_inputs), torch.tensor(validation_masks), torch.tensor(validation_labels))
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


def validation(nn, validation_dataloader, dev):
    print("")
    print("Running Validation...")
    t0 = time.time()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    nn.eval()

    for batch in validation_dataloader:
        batch = tuple(t.to(dev) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = nn(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))


def training(nn, train_dataloader, validation_dataloader, dev) -> list:
    loss_values = []
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    # torch.cuda.manual_seed_all(seed_val)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_loss = 0
        nn.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(dev)
            b_input_mask = batch[1].to(dev)
            b_labels = batch[2].to(dev)
            model.zero_grad()

            outputs = nn(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(nn.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        validation(nn, validation_dataloader, device)
        print("Training Done!")

    return loss_values


def testing(prediction_dataloader, dev):
    model.eval()
    predictions, true_labels = [], []

    for batch in prediction_dataloader:
        batch = tuple(t.to(dev) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')


def test_performance(path):
    batch = 32

    test = pd.read_csv(path, delimiter="\t", quoting=csv.QUOTE_NONE)
    test_tweets = test["Tweet text"].values
    test_labels = test["Label"].values
    test_vector = encode_vector(test_tweets, test_labels)
    test_vector = pad_sequences(test_vector, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    test_attention_mask = attention_mask(test_vector)

    prediction_inputs = torch.tensor(test_vector)
    prediction_masks = torch.tensor(test_attention_mask)
    prediction_labels = torch.tensor(test_labels)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch)

    testing(prediction_dataloader)

if __name__ == "__main__":
    device = check_device()
    df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)
    tweets = df["Tweet text"].values
    labels = df["Label"].values

    input_ids = encode_vector(tweets, tokenizer)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")

    attention_masks = attention_mask(input_ids)
    train_dataset, validation_dataset = prepare_dataset(input_ids, labels, attention_masks)
    total_steps = len(train_dataset) * epochs

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=4,  # The number of output labels--4 for multi-class classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    # pprint.pprint(labels.tolist())
    losses = training(model, train_dataset, validation_dataset, device)
    plot(losses)
