import json
from tqdm import tqdm
import torch
import numpy as np
import base64
from blink.main_dense import load_biencoder, _process_biencoder_dataloader

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




def load_models(params):
    # load biencoder model
    with open(params["biencoder_config"]) as json_file:
        biencoder_params = json.load(json_file)
        biencoder_params["path_to_model"] = params["biencoder_model"]
    biencoder = load_biencoder(biencoder_params)
    return biencoder, biencoder_params



def encode_mention_from_dict(doc, biencoder, biencoder_params):
    annotations = doc["annotations"]

    samples = []
    mentions = []

    for annotation in annotations:
        start = int(annotation["start_pos"])
        end = int(annotation["end_pos"])
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


