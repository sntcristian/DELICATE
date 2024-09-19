import argparse
from blink.main_dense import load_biencoder, _process_biencoder_dataloader
import json
from tqdm import tqdm
import torch
import numpy as np
import base64
import logging

def vector_encode(v):
    s = base64.b64encode(v).decode()
    return s

def vector_decode(s, dtype=np.float32):
    buffer = base64.b64decode(s)
    v = np.frombuffer(buffer, dtype=dtype)
    return v



def _run_biencoder_mention(biencoder, dataloader):
    biencoder.model.eval()
    encodings = []
    for batch in tqdm(dataloader):
        context_input, _, _ = batch
        with torch.no_grad():
            context_input = context_input.to(biencoder.device)
            context_encoding = biencoder.encode_context(context_input).numpy()
            context_encoding = np.ascontiguousarray(context_encoding)
        encodings.extend(context_encoding)
    return encodings




def load_models(args):
    # load biencoder model
    with open(args.biencoder_config) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)
    return biencoder, biencoder_params



def encode_mention_from_dict(doc):
    annotations = doc["annotations"]

    samples = []
    mentions = []

    for annotation in annotations:
        start = annotation["start_pos"]
        end = annotation["end_pos"]
        blink_dict = {
            'context_left': doc["text"][:start],
            'context_right': doc["text"][end],
            'mention': doc["text"][start:end],
            'label': 'unknown',
            'label_id': -1,
        }
        samples.append(blink_dict)
        mentions.append(annotation)

    dataloader = _process_biencoder_dataloader(
        samples, biencoder.tokenizer, biencoder_params
    )
    encodings = _run_biencoder_mention(biencoder, dataloader)
    if len(encodings) > 0:
        assert encodings[0].dtype == 'float32'
    encodings = [vector_encode(e) for e in encodings]

    for mention, enc in zip(mentions, encodings):
        mention['linking'] = {
            'encoding': enc,
            'source': 'blink_biencoder'
        }
    doc["annotations"]=mentions
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/blink_biencoder_base_wikipedia_ita/pytorch_model.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/blink_biencoder_base_wikipedia_ita/config.json",
        help="Path to the biencoder configuration.",
    )

    args = parser.parse_args()

    logger = logging.getLogger('biencoder_micros')

    print('Loading biencoder...')
    biencoder, biencoder_params = load_models(args)
    print('Device:', biencoder.device)
    print('Loading complete.')

doc = {
    "text":"Leonardo da Vinci fu un genio poliedrico.",
    "annotations":[
        {
            "start_pos":0,
            "end_pos":17,
            "type":"PER"
        }
    ]
}

encoded_doc = encode_mention_from_dict(doc)
print(encoded_doc)
print(vector_decode(encoded_doc["annotations"][0]["linking"]["encoding"])[:20])