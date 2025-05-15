from gliner import GLiNER
import json

def load_ner_model(config_file):
    with open(config_file, 'r') as file:
        params = json.load(file)
        ner_model = GLiNER.from_pretrained(params["ner_path"], load_tokenizer=True)
        ner_model.data_processor.config.max_len = 764
        return ner_model



def get_mentions_with_ner(doc, ner_model, tagset="DZ", threshold=0.9):
    text = doc["text"]
    type_mapper = {"persona": "PER", "luogo": "LOC", "opera": "WORK", "organizzazione": "ORG"}
    labels = ["persona", "luogo", "opera"] if tagset == "DZ" else ["persona", "luogo", "organizzazione"]
    entities = ner_model.predict_entities(text, labels, threshold=threshold)
    annotations = []
    for entity in entities:
        entry = {"doc_id": doc["doc_id"],
                 "surface": entity["text"],
                 "start_pos": entity["start"],
                 "end_pos": entity["end"],
                 "type": type_mapper[entity["label"]]}
        annotations.append(entry)
    doc["annotations"]=annotations
    return doc
