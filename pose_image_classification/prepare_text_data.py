import csv

data_dict = {}

while True:
    text = input('Please enter text... ')
    label = input('Please enter label... ')

    if text == 'done':
        break

    if label in list(data_dict.keys()):
        data_dict[label].append(text)
    else:
        data_dict[label] = [text]

fieldnames = ['text', 'label']
with open(os.path.join('static', 'train_text.csv'), 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for k, v in data_dict.items():
        for value in v:
            writer.writerow({'text': value, 'label': k})


