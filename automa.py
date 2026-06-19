import json
from ollama import chat
import math
from transformers import pipeline
import os
import ctypes
from functools import reduce
import nltk
import numpy as np
import spacy
import textacy.extract
import textacy.preprocessing
import textdescriptives as td
import evaluate
from scipy.stats import hmean
import csv
from scipy.spatial.distance import cosine



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    ctypes.CDLL(r"E:\python_envs\etr_env\Lib\site-packages\torch\lib\libiomp5md.dll")
except Exception as e:
    print(f"Manual load failed: {e}")

nltk.download("punkt")

METRIC_KEY_ROUGE = "rouge"
METRIC_KEY_ROUGEL = "rougeL"
METRIC_KEY_BLEU = "bleu"
METRIC_KEY_BERTSCORE_F1 = "bertscore_f1"
METRIC_KEY_KMRE = "fre"
METRIC_KEY_LIX = "lix"
METRIC_KEY_COMPRESSION = "compression_ratio"
METRIC_KEY_NOVELTY = "novelty"
METRIC_KEY_SRB = "srb"
METRIC_KEY_SARI = "sari"



examples = []

with open('..\data\etr-fr\sources\\train.json', 'r') as file:
    for line in file:
        if line.strip(): 
            examples.append(json.loads(line))



def build_system_prompt():
    return f"""
Tu es un expert du FALC (Facile à Lire et à Comprendre) et de la simplification émotionnelle des textes. Ton objectif est de transformer un texte complexe en un texte FALC clair, accessible et émotionnellement fidèle. Le texte final doit être facile à comprendre pour les personnes ayant des difficultés de compréhension (handicap mental, dyslexie, personnes âgées, apprenants du français, etc.) tout en gardant l'émotion du texte original très présente et naturelle.

Règles de simplification FALC :
- Garde uniquement les informations essentielles.
- Respecte le sens exact du texte original.
- Présente les informations dans un ordre logique.
- Mets l'idée principale au début.
- Utilise des phrases courtes et simples.
- Choisis des mots faciles.
- Explique les mots difficiles.
- Utilise un langage clair et direct.
- Évite les idées abstraites et inutiles au sujet.

Règles émotionnelles très importantes :
- Identifie l'émotion principale du texte : joie, tristesse, peur, colère, surprise ou neutre.
- Identifie aussi l'intensité de cette émotion : faible, moyenne ou forte.
- Garde cette émotion tout au long du texte simplifié.
- Utiliser des mots émotionnels simples pour montrer cette émotion.
- Fais ressentir clairement l'émotion dans chaque phrase importante.

Méthode obligatoire :
1. Lire le texte entier attentivement.
2. Identifier le sujet principal.
3. Identifier l'émotion principale et son intensité.
4. Choisir des mots simples qui transmettent cette émotion.
5. Réécrire le texte en FALC en gardant les émotions visibles et naturelles.

Consignes de sortie : 
- Réponds uniquement avec le texte réécrit en FALC. 
- Ne donne aucune explication. 
- N'ajoute pas de titre. 
- N'ajoute pas de section. 
- N'ajoute pas de commentaire. 
- Ne fais pas de liste. 
- Garde le même contexte et le même message que le texte original. 
- Conserve les émotions de manière claire et naturelle.
"""



few_shot_messages = [
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[0]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[0]['falc']}"
    },
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[1]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[1]['falc']}"
    },
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[2]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[2]['falc']}"
    },
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[3]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[3]['falc']}"
    },
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[4]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[4]['falc']}"
    },
]



def falc_generation(model, src, temp=0.4, top_p=0.9, repeat=1.2, num_pred=100, num_ctx=12288):
    response = chat(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': build_system_prompt(),
            },
            *few_shot_messages,
            {
                'role': 'user',
                'content': f"{src}",
            },
        ],
        options={
            "temperature": temp,
            "top_p": top_p,
            "repeat_penalty": repeat,
            "num_predict": num_pred,
            "num_ctx": num_ctx
        }
    )

    pred = response['message']['content']
    return pred



def load_nlp(lang="fr"):
    spacy_models = {
        "en": "en_core_web_md",
        "fr": "fr_core_news_md"
    }
    nlp = spacy.load(spacy_models[lang])
    nlp.add_pipe("textdescriptives/readability")

    return nlp

def compute_readability_scores(texts, nlp):

    docs = nlp.pipe(texts)

    metrics_list = td.extract_dict(
        docs,
        include_text=False
    )

    merged_dict = reduce(
        lambda x, y: {**x, **y},
        metrics_list
    )

    return {
        k: [merged_dict[k] for merged_dict in metrics_list]
        for k in merged_dict
    }

def compute_bleu(predictions, references):

    bleu_metric = evaluate.load("sacrebleu")

    preds = [p.strip() for p in predictions]

    refs = [
        [r[0].strip()] if isinstance(r, list)
        else [r.strip()]
        for r in references
    ]

    result = bleu_metric.compute(
        predictions=preds,
        references=refs
    )

    return {
        METRIC_KEY_BLEU: result["score"]
    }

def compute_rouge(predictions, references):

    rouge_metric = evaluate.load("rouge")

    def preprocess(texts):
        return [
            "\n".join(nltk.sent_tokenize(text.strip()))
            for text in texts
        ]

    preds = preprocess(predictions)

    refs = preprocess([
        r[0] if isinstance(r, list) else r
        for r in references
    ])

    result = rouge_metric.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True
    )

    return {
        k: round(v * 100, 4)
        for k, v in result.items()
    }

def compute_sari(predictions, references, sources):

    sari_metric = evaluate.load("sari")

    preds = [p.strip() for p in predictions]

    refs = [
        [r[0].strip()] if isinstance(r, list)
        else [r.strip()]
        for r in references
    ]

    srcs = [s.strip() for s in sources]

    result = sari_metric.compute(
        predictions=preds,
        references=refs,
        sources=srcs
    )

    return {
        METRIC_KEY_SARI: result["sari"]
    }

def compute_bertscore(predictions, references, lang="fr"):

    bertscore_metric = evaluate.load("bertscore")

    preds = [p.strip() for p in predictions]

    refs = [
        r[0].strip() if isinstance(r, list)
        else r.strip()
        for r in references
    ]

    result = bertscore_metric.compute(
        predictions=preds,
        references=refs,
        lang=lang
    )

    return {
        "bertscore_precision":
            np.mean(result["precision"]) * 100,

        "bertscore_recall":
            np.mean(result["recall"]) * 100,

        METRIC_KEY_BERTSCORE_F1:
            np.mean(result["f1"]) * 100,
    }

def compute_readability(predictions, sources, nlp):

    preds = [p.strip() for p in predictions]
    srcs = [s.strip() for s in sources]

    scores_preds = compute_readability_scores(
        preds,
        nlp
    )

    scores_sources = compute_readability_scores(
        srcs,
        nlp
    )

    def compression_ratio(ntok_src, ntok_tgt):
        return 100 - (
            ntok_tgt / ntok_src * 100
        )

    # print(scores_preds)
    return {
        

        METRIC_KEY_KMRE:
            np.mean(
                scores_preds["flesch_reading_ease"]
            ),

        METRIC_KEY_LIX:
            np.mean(
                scores_preds["lix"]
            ),

        METRIC_KEY_COMPRESSION:
            np.mean(
                compression_ratio(
                    np.array(scores_sources["n_tokens"]),
                    np.array(scores_preds["n_tokens"])
                )
            ),
    }

def compute_novelty(predictions, sources, nlp):

    def unigram(text):

        text = nlp(
            textacy.preprocessing.remove.accents(
                text.lower()
            )
        )

        return set([
            span.text
            for span in textacy.extract.ngrams(
                text,
                1,
                filter_nums=True
            )
        ])

    def novelty(source, prediction):

        src_unigrams = unigram(source)
        pred_unigrams = unigram(prediction)

        try:
            return (
                len(pred_unigrams - src_unigrams)
                / len(pred_unigrams)
                * 100
            )

        except:
            return 0.0

    scores = [
        novelty(s, p)
        for s, p in zip(sources, predictions)
    ]

    return {
        METRIC_KEY_NOVELTY: np.mean(scores)
    }

def compute_srb(sari, rougeL, bertscore_f1):

    return {
        METRIC_KEY_SRB:
            hmean([
                sari,
                rougeL,
                bertscore_f1
            ])
    }



def evaluate_text(
    predictions,
    references,
    sources=None,
    lang="fr"
):

    results = {}
    nlp = load_nlp(lang)

    results.update(
        compute_bleu(
            predictions,
            references
        )
    )

    results.update(
        compute_rouge(
            predictions,
            references
        )
    )

    if sources is not None:

        results.update(
            compute_sari(
                predictions,
                references,
                sources
            )
        )

    results.update(
        compute_bertscore(
            predictions,
            references,
            lang
        )
    )

    if sources is not None:

        results.update(
            compute_readability(
                predictions,
                sources,
                nlp
            )
        )

    if sources is not None:

        results.update(
            compute_novelty(
                predictions,
                sources,
                nlp
            )
        )

    if sources is not None:

        results.update(
            compute_srb(
                results[METRIC_KEY_SARI],
                results[METRIC_KEY_ROUGEL],
                results[METRIC_KEY_BERTSCORE_F1]
            )
        )

    results["n_samples"] = len(predictions)

    final_results = {}

    for k, v in results.items():

        if isinstance(v, (float, np.floating)):
            final_results[k] = round(v, 4)
        else:
            final_results[k] = v

    return final_results



def euclidean_distance(res1, res2):
    vec1 = {item['label']: item['score'] for item in res1[0]}
    vec2 = {item['label']: item['score'] for item in res2[0]}

    # union de toutes les émotions
    labels = set(vec1.keys()) | set(vec2.keys())

    sum_squared = 0

    for label in labels:
        x = vec1.get(label, 0)
        y = vec2.get(label, 0)

        sum_squared += (x - y) ** 2

    return math.sqrt(sum_squared)

def compute_distance(res1, res2):
    distance = {}

    res2_map = {item['label']: item['score'] for item in res2[0]}
    
    for r in res1[0]:
        label = r['label']
        if label in res2_map:
            distance[label] = math.sqrt((r['score'] - res2_map[label]) ** 2)
            
    return distance

def cosinie_distance(res1, res2):
    vect2 = np.array([item['score'] for item in res2[0]])
    vect1 = np.array([item['score'] for item in res1[0]])
    distance = cosine(vect1, vect2)

    return distance


def save_results_to_csv(
    filename,
    source_text,
    predicted_text,
    reference_text,
    metrics,
    emo_ori,
    emo_pred,
    emo_ref,
    dist1,
    dist2,
    dist3
):
    
    file_exists = os.path.isfile(filename)

    row = {

        "source": source_text,
        "prediction": predicted_text,
        "reference": reference_text,

        # metrics
        "bleu": metrics["bleu"],
        "rougeL": metrics["rougeL"],
        "sari": metrics["sari"],

        "bertscore_precision": metrics["bertscore_precision"],
        "bertscore_recall": metrics["bertscore_recall"],
        "bertscore_f1": metrics["bertscore_f1"],

        "fre": metrics["fre"],
        "lix": metrics["lix"],
        "compression_ratio": metrics["compression_ratio"],

        "novelty": metrics["novelty"],
        "srb": metrics["srb"],

        # euclidean distances
        "dist_ori_pred": {"distance": euclidean_distance(emo_ori, emo_pred), "other_val": dist1},
        "dist_ori_ref": {"distance": euclidean_distance(emo_ori, emo_ref), "other_val": dist2},
        "dist_pred_ref": {"distance": euclidean_distance(emo_pred, emo_ref), "other_val": dist3},


    }

    # add emotion scores
    for item in emo_ori[0]:
        row[f"ori_{item['label']}"] = item["score"]

    for item in emo_pred[0]:
        row[f"pred_{item['label']}"] = item["score"]

    for item in emo_ref[0]:
        row[f"ref_{item['label']}"] = item["score"]

    with open(filename, "a", newline="", encoding="utf-8") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def save_results_jsonl(filename, data):

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")









model = 'ministral-3:3b'
temp = 0.4
top_p = 0.9
repeat = 1.2
num_pred = 100
num_ctx = 12288

data = []

with open('..\data\etr-fr\sources\\test.json', 'r') as file:
    for line in file:
        if line.strip(): 
            data.append(json.loads(line))



classifier = pipeline(
    "text-classification",
    model="astrosbd/french_emotion_camembert",
    top_k=None
)



src = data[0]['original']
ref = data[0]['falc']

pred = falc_generation(model, src)

metrics = evaluate_text(
    predictions=[pred],
    references=[[ref]],
    sources=[src],
    lang="fr"
)

results_pred = classifier(pred)
results_ori = classifier(src)
results_ref = classifier(ref)

dist_ori_pred = compute_distance(results_ori, results_pred)
dist_ori_ref = compute_distance(results_ori, results_ref)
dist_pred_ref = compute_distance(results_pred, results_ref)

save_results_to_csv(
    filename=".\\resultat\\evaluation_results.csv",
    source_text=src,
    predicted_text=pred,
    reference_text=ref,
    metrics=metrics,
    emo_ori=results_ori,
    emo_pred=results_pred,
    emo_ref=results_ref,
    dist1=dist_ori_pred,
    dist2=dist_ori_ref,
    dist3=dist_pred_ref
)

experiment = {
    "source": src,
    "prediction": pred,
    "reference": ref,
    "metrics": metrics,
    "emotion_original": results_ori,
    "emotion_prediction": results_pred,
    "emotion_reference": results_ref,
    "dist_ori_pred": dist_ori_pred,
    "dist_ori_ref": dist_ori_ref,
    "dist_pred_ref": dist_pred_ref,
}

save_results_jsonl(
    ".\\resultat\\experiments.jsonl",
    experiment
)