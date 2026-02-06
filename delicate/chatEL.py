import json
import csv
import transformers
import torch
import re
from tqdm import tqdm
import os
import argparse


def process_candidates(candidates, n_candidates):
    output = []
    candidates = sorted(candidates, key=lambda x: x['blink_score'])
    for item in candidates[:min(n_candidates, len(candidates))]:
        output.append({
            "wikipedia_page": item["title"],
            "wikidata_id": item["q_id"],
        })
    return output


def main():
    parser = argparse.ArgumentParser(description="LLM Prompting for Entity Disambiguation from Candidate List")
    parser.add_argument("--json_f", type=str, default="camera-ready/ED/DZ/candidates_top50.json", help="Path to JSON list of candidates")
    parser.add_argument("--output_dir", type=str, default="camera-ready/ED/DZ/ChatEL", help="Output directory for results")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-Small-24B-Instruct-2501", help="Huggingface repo of LLM")
    parser.add_argument("--hf_token", type=str, required=True, help="Huggingface token to access restricted repo.")
    parser.add_argument("--n_candidates", type=int, default=10, help="Number of candidates to put in prompt.")

    args = parser.parse_args()
    with open(args.json_f, "r", encoding="utf-8") as f:
        retriever_results = json.load(f)



    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=args.hf_token
    )

    system_prompt1 = """
    Sei un efficace sistema multilingue di estrazione delle informazioni specializzato nella disambiguazione di 
    entità all'interno di testi storici relativi al patrimonio culturale italiano.
    Il tuo compito è analizzare il testo fornito dall'utente e facilitare la disambiguazione del riferimento contrassegnato dai tag <ENT></ENT> fornendo una descrizione dell'entità menzionata sulla base del contesto.
    Inizia sempre la frase con: 'L'entità X nel testo rappresenta ...'
    """


    system_prompt2 = """
    Sei un efficace sistema multilingue di estrazione delle informazioni specializzato nella disambiguazione di 
    entità all'interno di testi storici relativi al patrimonio culturale italiano.
    Il tuo compito è analizzare il testo fornito dall'utente e disambiguare il riferimento contrassegnato dai tag <ENT></ENT> selezionando un'entità Wikidata da una lista di candidati fornita solo quando sei altamente sicuro, 
    classificandola con un valore NIL in caso contrario.
    Rispondi sempre restituendo una risposta in formato JSON; non generare codice Python.
    """

    output = []
    article_to_entities = dict()

    for doc in tqdm(retriever_results):
        doc_id = doc["doc_id"]
        text = doc["text"]
        for item in doc["annotations"]:
            start_pos = int(item["start_pos"])
            end_pos = int(item["end_pos"])
            processed_text = text[max(0, start_pos - 500):start_pos] + "<ENT> " + text[start_pos:end_pos] + " </ENT> "+ text[end_pos:min(len(text), end_pos + 500)]
            processed_candidates = process_candidates(item["candidates"], args.n_candidates)
            user_prompt1 = """
            Analizza attentamente il testo in input.
            Fornisci una breve descrizione di cosa rappresenta l'entità menzionata tra i tag <ENT></ENT> sulla base del contesto della frase.
            --------------------
            Testo in input:
            """ + processed_text + """
            """
            messages = [
                {"role": "system", "content": system_prompt1},
                {"role": "user", "content": user_prompt1},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=128,
            )
            augmented_context = outputs[0]["generated_text"][-1]["content"]

            user_prompt2 = """Analizza attentamente il testo in input.
            Disambigua l'entità denominata all'interno del testo tra i tag <ENT></ENT> selezionando l'entità Wikidata più 
            appropriata dalla lista di candidati. Basa la tua decisione sulla base del contesto aggiuntivo fornito nel prompt.
            Restituisci il titolo della pagina Wikipedia e l'ID Wikidata corrispondenti dell'entità selezionata in un oggetto JSON formattato come segue:

            ```json
            {"wikipedia_page":"", "wikidata_id":""}
            ```

            Assicurati di selezionare sia l'ID Wikidata che il titolo della pagina Wikipedia dalla lista di candidati fornita.
            Presta attenzione al fatto che la lista di candidati potrebbe non includere l'entità menzionata. Se nessuno dei candidati corrisponde con alta confidenza all'entità contrassegnata con [ENT], usa la stringa "NIL" come valore della chiave "wikidata_id" per indicare l'assenza di candidati corretti.
            ----------------------
            Testo in Input:
            """ + processed_text + """
            ---------------------
            Contesto aggiuntivo:

            """ + augmented_context + """
            ---------------------
            Lista JSON di Candidati:
            ```json
            """ + str(processed_candidates) + """ 
            ```."""

            messages = [
                {"role": "system", "content": system_prompt2},
                {"role": "user", "content": user_prompt2},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=128,
            )
            response = outputs[0]["generated_text"][-1]["content"]
            match = re.search(r'"wikidata_id"\s*:\s*"(Q\d+)"', response)

            if match:
                wikidata_id = match.group(1)
            else:
                wikidata_id = "NIL"


            selected_entity = [x for x in item["candidates"] if x["q_id"] == wikidata_id]
            if len(selected_entity) > 0:
                output.append({
                    "doc_id":doc_id,
                    "start_pos":start_pos,
                    "end_pos":end_pos,
                    "surface":item["surface"],
                    "type":item["type"],
                    "identifier":wikidata_id,
                    "wiki_title":selected_entity[0]["title"],
                    "answer":re.sub(r'\s+', " ", response),
                    "score":selected_entity[0]["rf_score"]
                })
            else:
                output.append({
                    "doc_id": doc_id,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "surface": item["surface"],
                    "type": item["type"],
                    "identifier": "NIL",
                    "wiki_title": item["surface"],
                    "answer": re.sub(r'\s+', " ", response),
                    "score": 0
                })

                
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "output.csv"), "w", encoding="utf-8") as out_f:
        dict_writer = csv.DictWriter(out_f, output[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(output)

if __name__ == "__main__":
    main()


