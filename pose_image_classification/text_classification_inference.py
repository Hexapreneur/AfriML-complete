from transformers import pipeline
import json
import os


def text_inference(text):
    try:
        with open(os.path.join('static', 'train_text_classes.json'), 'r') as f:
            classes = json.load(f)
        classifier = pipeline('text-classification', model=os.path.join('static', 'text_classification_model'))
        result = classifier(text)

        final_result = result[0]
        label = ''
        for k, v in final_result.items():
            if k == 'label':
                label = classes[int(v.split('_')[-1])]

        final_result['label'] = label

        return True, final_result, classes
    except Exception as e:
        return False, str(e)
