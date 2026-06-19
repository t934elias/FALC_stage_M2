from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch


base_model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoAdapterModel.from_pretrained(base_model_name, torch_dtype=torch.float16)
adapter_name = model.load_adapter("./model/mistral_7b_etr_francois", set_active=True)

if "causal_lm" not in model.heads:
    model.add_causal_lm_head(adapter_name, overwrite_ok=True)
model.set_active_adapters(adapter_name)

model.to("cuda", dtype=torch.float16)
text = "L\u00e9a, pourquoi es-tu triste ? Ce matin, Cl\u00e9ment se r\u00e9veille tout content. Aujourd'hui, c'est l'anniversaire de sa petite s\u0153ur L\u00e9a. Encore en pyjama, il saute de son lit et se pr\u00e9cipite vers la chambre de L\u00e9a. Il est impatient de lui souhaiter son anniversaire. Arriv\u00e9 devant sa porte, il frappe doucement avant d'entrer en criant joyeusement: \u00ab Bon anniversaire, petite s\u0153ur ! \u00bb L\u00e9a, encore un peu endormie, ouvre un \u0153il. La petite fille a eu du mal \u00e0 trouver le sommeil cette nuit. C'est pourquoi, ce matin elle est vraiment tr\u00e8s fatigu\u00e9e. Elle aurait voulu se reposer encore un peu. \u00c0 la place, son fr\u00e8re n'arr\u00eate pas de sautiller devant elle en r\u00e9p\u00e9tant gaiement : \u00ab Tu es grande maintenant! \u00bb."

prompt = f"### Instruction:\nSimplifier le texte suivant en français facile à lire (ETR) :\n\n### Texte:\n{text}\n\n### Texte simplifié:\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda") # or .to("cpu")
model.to("cuda")


output_ids = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=250, 
    do_sample=True,     # Must be True to use temperature
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id # Prevents open-ended generation warnings
)

response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("--- Original Text ---")
print(text)
print("\n--- Generated ETR Text ---")
print(response)

pred = response
src = text
ref = "L\u00e9a est triste Ce matin, Cl\u00e9ment est tout content. C'est l'anniversaire de sa s\u0153ur L\u00e9a. Cl\u00e9ment entre dans la chambre de L\u00e9a. \u00ab Bon anniversaire, petite s\u0153ur ! Tu as 8 ans aujourd'hui. Tu es grande maintenant! \u00bb"

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
from transformers import pipeline
from scipy.spatial.distance import cosine
import math

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

metrics = evaluate_text(
    predictions=[pred],
    references=[[ref]],
    sources=[src],
    lang="fr"
)

print(f"BLEU Score: {metrics['bleu']:.4f}")

print(f"ROUGE-L: {metrics['rougeL']:.4f}")

print(f"SARI Score: {metrics['sari']:.4f}")

print("\nBERTScore:")
print(f"Precision: {metrics['bertscore_precision']:.4f}")
print(f"Recall:    {metrics['bertscore_recall']:.4f}")
print(f"F1:        {metrics['bertscore_f1']:.4f}")

print("\nReadability:")
print(f"flesch reading ease: {metrics['fre']:.4f}")
print(f"LIX:                 {metrics['lix']:.4f}")
print(f"Compression Ratio:   {metrics['compression_ratio']:.4f}")

print("\nNovelty:")
print(f"Novelty Score: {metrics['novelty']:.4f}")

print("\nCombined Metric:")
print(f"SRB Score: {metrics['srb']:.4f}")

print(f"\nNumber of Samples: {metrics['n_samples']}")

classifier = pipeline(
    "text-classification",
    model="astrosbd/french_emotion_camembert",
    top_k=None
)

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
    dict1 = {item['label']: item['score'] for item in res1[0]}
    dict2 = {item['label']: item['score'] for item in res2[0]}
    
    all_keys = sorted(list(set(dict1.keys()).union(dict2.keys())))
    
    vect1 = np.array([dict1.get(key, 0.0) for key in all_keys])
    vect2 = np.array([dict2.get(key, 0.0) for key in all_keys])

    distance = cosine(vect1, vect2)
    
    return distance

results_pred = classifier(pred)
results_ori = classifier(src)
results_ref = classifier(ref)

print("\nTexte original :")
for r in results_ori[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nTexte FALC :")
for r in results_pred[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nTexte ref :")
for r in results_ref[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nDistance euclidienne entre text ori et falc predit")
dist = compute_distance(results_ori, results_pred)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(cosinie_distance(results_ori, results_pred))

print("\nDistance euclidienne entre text ori et falc ref")
dist = compute_distance(results_ori, results_ref)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(cosinie_distance(results_ori, results_ref))

print("\nDistance euclidienne entre falc pred et falc ref")
dist = compute_distance(results_pred, results_ref)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(cosinie_distance(results_pred, results_ref))
