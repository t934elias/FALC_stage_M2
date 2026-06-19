# dpo
# Score=0.35⋅SARI+0.25⋅BERTScore+0.20⋅EmotionSimilarity+0.10⋅FRE+0.10⋅Compression


# grpo
# Reward=0.4⋅SARI+0.3⋅BERTScore+0.3⋅EmotionSimilarity


import json
import pandas as pd
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
from scipy.spatial.distance import cosine
from transformers import pipeline


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    ctypes.CDLL(r"E:\python_envs\etr_env\Lib\site-packages\torch\lib\libiomp5md.dll")
except Exception as e:
    print(f"Manual load failed: {e}")

nltk.download("punkt", quiet=True)

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


with open(".\\resultat\\falc_generes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

flattened_rows = []

for entry in data:
    orig_id = entry["sentence_id"]
    orig_text = entry["original_text"]
    falc_text = entry["falc"]
    
    for run in entry["runs"]:
        row = {
            "sentence_id": orig_id,
            "original_text": orig_text,
            "falc_text": falc_text,
            "temperature": run["parameters"]["temperature"],
            "num_ctx": run["parameters"]["num_ctx"],
            "falc": run["summary"]
        }
        flattened_rows.append(row)

df = pd.DataFrame(flattened_rows)

# print(df.head())


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

    results.update(compute_bleu(predictions, references))

    results.update(compute_rouge(predictions, references))

    if sources is not None:
        results.update(compute_sari(predictions, references, sources))

    results.update(compute_bertscore(predictions, references, lang))

    if sources is not None:
        results.update(compute_readability(predictions, sources, nlp))

    if sources is not None:
        results.update(compute_novelty(predictions, sources, nlp))

    if sources is not None:
        results.update(compute_srb(
                results[METRIC_KEY_SARI],
                results[METRIC_KEY_ROUGEL],
                results[METRIC_KEY_BERTSCORE_F1]))

    results["n_samples"] = len(predictions)

    final_results = {}

    for k, v in results.items():

        if isinstance(v, (float, np.floating)):
            final_results[k] = round(v, 4)
        else:
            final_results[k] = v

    return final_results



def cosine_distance(res1, res2):
    vect2 = np.array([item['score'] for item in res2[0]])
    vect1 = np.array([item['score'] for item in res1[0]])
    distance = cosine(vect1, vect2)

    return distance


classifier = pipeline(
    "text-classification",
    model="astrosbd/french_emotion_camembert",
    top_k=None
)

resultats = []

for idx, row in df.iterrows():
    pred = str(row['falc'])
    ref = str(row['falc_text'])
    src = str(row['original_text'])

    results_pred = classifier(pred)
    results_ori = classifier(src)
    results_ref = classifier(ref)

    metrics = evaluate_text(
        predictions = [pred],
        references = [[ref]],
        sources = [src],
        lang = "fr"
    )
    
    metrics['dis_ori_pred'] = 1 - cosine_distance(results_ori, results_pred)
    metrics['dis_ori_ref'] = 1 - cosine_distance(results_ori, results_ref)
    metrics['dis_pred_ref'] = 1 - cosine_distance(results_pred, results_ref)

    bert = (metrics['bertscore_precision'] + metrics['bertscore_recall'] + metrics['bertscore_f1']) / 3
    score = (0.2 * metrics['sari']) + (0.3 * (bert)) + (0.1 * (metrics['fre'])) + (0.05 * (metrics['novelty'])) + (0.05 * (metrics['compression_ratio'])) + (0.25 * metrics['dis_ori_pred'])
    
    metrics['score_dpo'] = score

    metrics['df_index'] = idx
    metrics['sentence'] = row['sentence_id']
    resultats.append(metrics)

df_res = pd.DataFrame(resultats)

df.to_excel(".\\resultat\\summaries_for_generation.xlsx", index=False)
df_res.to_excel(".\\resultat\\summaries_for_evaluation.xlsx", index=False)

# # Average summary length by temperature
# df['summary_length'] = df['summary'].apply(lambda x: len(x.split()))
# avg_lengths = df.groupby('temperature')['summary_length'].mean()

# print("Average Summary Word Count by Temperature:")
# print(avg_lengths)

