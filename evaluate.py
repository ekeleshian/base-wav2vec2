import pickle
import random
from transformers import Wav2Vec2ForCTC
import torch
from datasets import load_metric
import pandas as pd
import tabulate


def show_random_elements(dataset, num_examples=20):
    assert num_examples <= len(dataset), "can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    print(df)

def map_to_result(batch):
    model.to("cuda")
    input_values = processor(batch['speech'], sampling_rate=batch['sampling_rate'], return_tensors="pt").input_values.to('cuda')
    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch['pred_str'] = processor.batch_decode(pred_ids)[0]

    return batch


if __name__ == "__main__":
    with open('timit.pkl', 'rb') as f:
        timit = pickle.load(f)

    with open("processor.pkl", "rb") as f:
        processor = pickle.load(f)

    model = Wav2Vec2ForCTC.from_pretrained("/home/edgar/transformers/elizabethkeleshian-wav2vec2-base-timit-demo")
    results = timit['test'].map(map_to_result)
    wer_metric = load_metric('wer')

    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results['pred_str'], references=results['target_text'])))
    show_random_elements(results.remove_columns(['speech', 'sampling_rate']))
    
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
    

