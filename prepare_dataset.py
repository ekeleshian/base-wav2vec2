import re
import json
import pickle
from datasets import load_dataset, load_metric
from datasets import ClassLabel
import random
import pandas as pd
import soundfile as sf
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor


def remove_special_characters(batch):
    batch['text'] = re.sub(chars_to_ignore_regex, "", batch['text']).lower() + " "
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch['text'])
    vocab = list(set(all_text))
    return {"vocab": [vocab], 'all_text': [all_text]}


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read(batch['file'])
    batch['speech'] = speech_array
    batch['sampling_rate'] = sampling_rate
    batch['target_text'] = batch['text']
    return batch


def prepare_dataset(batch):
    assert (
    	len(set(batch['sampling_rate'])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    
    batch['input_values'] = processor(batch['speech'], sampling_rate=batch['sampling_rate'][0]).input_values
    
    with processor.as_target_processor():
        batch['labels'] = processor(batch['target_text']).input_ids
    return batch


if __name__ == "__main__":
	timit = load_dataset("timit_asr")

	timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

	chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
	timit = timit.map(remove_special_characters)

	vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names['train'])
	vocab_list = list(set(vocabs['train']['vocab'][0]) | set(vocabs['test']['vocab'][0]))
	vocab_dict = {v: k for k, v in enumerate(vocab_list)}

	vocab_dict["|"] = vocab_dict[" "]
	del vocab_dict[" "]

	vocab_dict['[UNK]'] = len(vocab_dict)
	vocab_dict['[PAD]'] = len(vocab_dict)

	with open('vocab.json', 'w') as vocab_file:
	    json.dump(vocab_dict, vocab_file)

	tokenizer = Wav2Vec2CTCTokenizer('./vocab.json', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)

	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

	timit = timit.map(speech_file_to_array_fn, remove_columns=timit.column_names['train'], num_proc=4)

	timit_prepared = timit.map(prepare_dataset, remove_columns=timit.column_names['train'], batch_size=8, num_proc=4, batched=True)

	with open('timit_prepared.pkl', 'wb') as f:
		pickle.dump(timit_prepared, f)

	with open('processor.pkl', 'wb') as f:
		picke.dump(processor, f)


