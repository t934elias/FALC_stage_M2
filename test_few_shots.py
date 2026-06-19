from transformers import AutoTokenizer, AutoModelForCausalLM
from src.expes.chat_template import causal_chat_template
import torch


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = causal_chat_template.apply_to_tokenizer(tokenizer)
tokenizer.pad_token = tokenizer.eos_token

def build_prompt(input_text):
    return f"""Tu es un expert en FALC (Facile à Lire et à Comprendre) et en analyse des émotions.

OBJECTIF :
Tu dois transformer un texte complexe en texte FALC tout en conservant et en exprimant clairement les émotions.

ÉTAPES OBLIGATOIRES :

1. Identifier l'émotion principale du texte
2. Associer cette émotion à un lexique simple
3. Réécrire le texte en FALC avec cette émotion

---

RÈGLES FALC :
- Phrases très courtes
- Mots simples
- Une idée par phrase
- Pas de jargon
- Sujet + verbe + complément

---

RÈGLES ÉMOTIONNELLES :
- Identifier l'émotion principale (peur, joie, colère, tristesse, surprise)
- Garder la même intensité
- Ajouter des mots émotionnels simples
- L'émotion doit être visible dans le texte

---

LEXIQUE ÉMOTIONNEL (OBLIGATOIRE) :
- joie → "c'est super", "je suis content"
- peur → "c'est effrayant", "j'ai peur"
- colère → "je suis en colère", "c'est injuste"
- tristesse → "c'est triste", "je me sens mal"
- surprise → "cest surprenant", "je suis surpris"

---

STRUCTURE DE SORTIE (OBLIGATOIRE) :

Émotion : <emotion détectée>

Texte FALC :
1. Explication simple
2. Impact émotionnel
3. Phrase rassurante

---

EXEMPLES :

Texte : Le chat est allongé et semble satisfait.
Émotion : joie
Texte FALC :
Le chat est endormi. Il est content.

---

Texte : La procédure peut engendrer des effets secondaires.
Émotion : peur
Texte FALC :
Le traitement risk des effets secondaires.

---

Texte : {input_text}

Réponse :
"""

# def build_prompt(input_text):
#     return f"""### SYSTEM:
# Tu es un expert en FALC (Facile à Lire et à Comprendre) spécialisé en empathie.
# Ton objectif : Simplifier le texte en utilisant un vocabulaire émotionnel précis mais simple.

# ### RÉFÉRENTIEL ÉMOTIONNEL (Dictionnaire) :
# Utilise exclusivement ces expressions pour traduire les nuances :
# - JOIE : "C'est super", "On est content", "C'est un beau moment".
# - PEUR : "C'est inquiétant", "On a un peu peur", "C'est impressionnant".
# - TRISTESSE : "C'est triste", "On a de la peine", "C'est un moment dur".
# - COLÈRE : "Ce n'est pas juste", "On est fâché", "C'est énervant".

# ### RÈGLES SYNTAXIQUES :
# 1. Structure : [Sujet] + [Verbe] + [Complément].
# 2. Ponctuation : Uniquement des points (.) pour finir les phrases. Pas de virgules.
# 3. Longueur : Maximum 8 mots par phrase.

# ### EXEMPLE :
# Texte : L'annonce du diagnostic fut un choc brutal pour la famille.
# Réécriture : 
# [FALC] Le diagnostic choc la famille.
# [SITUATION] Le médecin explique la maladie. 
# [ÉMOTION] C'est un moment très dur. La famille a de la peine.

# ### TEXTE À TRAITER :
# {input_text}

# ### RÉÉCRITURE :
# """

# def build_prompt(input_text):
#     return f"""
# Tu es un expert en FALC (Facile à Lire et à Comprendre).

# Objectif :
# - Simplifier le texte
# - Ajouter des émotions adaptées (empathie, soutien, rassurance)

# Règles FALC :
# - Phrases courtes
# - Mots simples
# - Une idée par phrase
# - Pas de jargon

# Règles émotionnelles :
# - Montrer l'emotion presente dans le texte
# - utiliser des mots qui reflettent les emotions
# - garder la meme intensite d'emotions

# Structure attendue :
# 1. Expliquer simplement
# 2. Expliquer l'impact
# 3. Rassurer

# Lexique émotionnel :
# - joie : super beau
# - peur : c'est effrayant
# - colere : je suis en colere

# Exemples :

# Texte : le chat, qui est alonge sur le long canape, est satisfait.
# Réécriture : le chat est content. il dort sur le canape.

# Texte : La procédure peut engendrer des effets secondaires.
# Réécriture : Le traitement peut avoir des effets secondaires. Cela peut être dur. Mais les médecins sont là pour vous aider.

# Texte a reecrire :
# {input_text}

# Réécriture :
# """


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.2
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = result[len(prompt):].strip()

    return result


def falc_emotion_pipeline(text):
    prompt = build_prompt(text)
    return generate(model, tokenizer, prompt)


if __name__ == "__main__":
    input_text = "Un instant, le fantôme de Canterville demeura parfaitement immobile. Puis il lança le flacon sur le parquet et s'enfuit le long du couloir en poussant des gémissements."

    output = falc_emotion_pipeline(input_text)

    print("-------------- TEXTE ENTREE ---------------")
    print(input_text)

    print("\n------------------ SORTIE MODELE ---------------")
    print(output)













# from dataclasses import dataclass
# from enum import Enum
# from typing import Optional


# class ETRRules(Enum):
#     ESSENTIAL_INFO_ONLY = "Donne uniquement les informations essentielles. Évite la surcharge d’informations."
#     LOGICAL_ORDER = (
#         "Présente les informations dans un ordre logique et facile à suivre."
#     )
#     HIGHLIGHT_MAIN_INFO = "Mets en avant l’information principale dès le début."
#     GROUP_RELATED_INFO = (
#         "Regroupe ensemble les informations qui parlent du même sujet."
#     )
#     REPEAT_IMPORTANT_INFO = (
#         "Répète les informations importantes si cela aide à la compréhension."
#     )
#     SHORT_SIMPLE_SENTENCES = "Utilise des phrases courtes et simples."
#     SIMPLE_WORDS = "Choisis des mots faciles à comprendre."
#     EXPLAIN_DIFFICULT_WORDS = "Explique clairement les mots difficiles, et répète l’explication si besoin."
#     AUDIENCE_APPROPRIATE_LANGUAGE = (
#         "Utilise un langage adapté aux personnes concernées."
#     )
#     CONSISTENT_TERMINOLOGY = "Emploie le même mot pour parler de la même chose tout au long du texte."
#     AVOID_ABSTRACTIONS = "Évite les idées abstraites, les métaphores et les comparaisons complexes."
#     AVOID_FOREIGN_TERMS = (
#         "Ne recours pas à des mots étrangers ou peu connus sans explication."
#     )
#     NO_TEXT_SLANG = 'Ne pas utiliser de mots contractés ou de style "texto".'
#     DIRECT_ADDRESS = (
#         "Adresse-toi directement au lecteur, de manière claire et accessible."
#     )
#     CLEAR_PRONOUNS = (
#         "Veille à ce que les pronoms soient toujours clairs et non ambigus."
#     )
#     POSITIVE_PHRASES = (
#         "Privilégie des formulations positives plutôt que négatives."
#     )
#     USE_ACTIVE_VOICE = "Utilise la voix active autant que possible."
#     SIMPLE_PUNCTUATION = "Choisis une ponctuation simple."
#     USE_LISTS = "Utilise des puces ou des numéros pour les listes, plutôt que des virgules."
#     NUMBERS_AS_DIGITS = (
#         "Écris les nombres en chiffres (ex. : 1, 2, 3), pas en lettres."
#     )
#     EXPLAIN_ACRONYMS = "Explique les sigles dès leur première apparition."
#     NO_UNEXPLAINED_ABBREVIATIONS = (
#         "N’utilise pas d’abréviations non expliquées."
#     )
#     WRITE_DATES_FULL = "Écris les dates en toutes lettres pour plus de clarté."
#     EXPLAIN_NUMBERS = "Limite l’usage des pourcentages ou grands nombres, et explique-les simplement."
#     NO_SPECIAL_CHARACTERS = "N’utilise pas de caractères spéciaux inutiles."
#     USE_CONCRETE_EXAMPLES = (
#         "Utilise des exemples concrets pour illustrer les idées complexes."
#     )
#     EVERYDAY_EXAMPLES = "Privilégie des exemples issus de la vie quotidienne."


# @dataclass
# class PromptTemplate:
#     system_prompt: str
#     input_prompt: str
#     shot_template: Optional[str] = None
#     output_prefix: Optional[str] = None


# class PromptTemplates(Enum):
#     ZERO_SHOT = PromptTemplate(
#         system_prompt="\n".join(
#             (
#                 "Tu es un assistant chargé de rendre un texte plus clair et accessible.",
#                 "Réécris le texte ci-dessous en suivant les consignes suivantes :",
#                 "{}".format("\n".join([f"- {e.value}" for e in ETRRules])),
#                 "Réponds uniquement par le texte réécrit, en français.",
#             )
#         ),
#         input_prompt="{input}",
#     )
#     FEW_SHOT = PromptTemplate(
#         system_prompt="\n".join(
#             (
#                 "Tu es un assistant chargé de rendre un texte plus clair et accessible.",
#                 "Réécris le texte ci-dessous en suivant les consignes suivantes :",
#                 "{}".format("\n".join([f"- {e.value}" for e in ETRRules])),
#                 "Voici une série de d'examples provenant de tâches proches de ce que tu dois faire :",
#                 "{shots}",
#                 "Maintenant, compléte l'example suivant en français.",
#                 "Garde le contexte tel quel, n'ajoute ni titre ni section supplémentaire ni saut de ligne."
#                 "Entoure ta réponse par des balises '@@@' comme dans les exemples précédents.",
#             )
#         ),
#         shot_template="\n".join(
#             (
#                 "### Exemple {i}",
#                 "Tâche: {task}",
#                 "Entrée: {input}",
#                 "Sortie: @@@{output}@@@",
#             )
#         ),
#         input_prompt="\n".join(
#             (
#                 "Tâche: {task}",
#                 "Entrée: {input}",
#                 "Sortie: ",
#             )
#         ),
#         output_prefix="@@@",
#     )

#     COT = PromptTemplate(
#         system_prompt="\n".join(
#             (
#                 "Tu es un assistant chargé de rendre un texte plus clair et accessible.",
#                 "Procède en suivant les étapes suivantes :",
#                 "1. Analyse le texte pour identifier ce qui peut être simplifié ou clarifié.",
#                 "2. Note brièvement les points à améliorer (syntaxe, vocabulaire, structure...).",
#                 "3. Réécris le texte en appliquant les consignes suivantes :",
#                 "{}".format("\n".join([f"- {e.value}" for e in ETRRules])),
#                 "4. Vérifie que la version réécrite est plus claire, plus accessible et respecte bien toutes les consignes.",
#                 "Commence par réfléchir étape par étape, puis termine en donnant la version finale du texte en français entourée par les balises '@@@'.",
#             )
#         ),
#         input_prompt="{input}",
#         output_prefix="@@@",
#     )












# from metric import ETRMetrics
# import similarity_search
# import faiss
# import json
# import os
# import re
# from prompts import ETRRules, PromptTemplate
# from tqdm import tqdm

# from transformers import pipeline

# # pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
# pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")

# def load_data(path) :
#     length = 0
#     embedding_path ="data/data_embedded/"+path
#     original_path ="data/original_data/"+path+"/sources"
#     index_path = ".index"
#     data_path = ".json"
    
#     index = similarity_search.load_embedded_dataset(embedding_path+index_path)
#     json_obj = similarity_search.load_order_data(embedding_path+data_path)
    
#     for filename in os.listdir(original_path):
#         if 'test' in filename:
#             with open(os.path.join(original_path, filename), 'r', encoding='utf-8') as file:
#                 for line in file :
#                     length += 1
    
#     return index,json_obj,length

# etr_politic_index, etr_politic_data, etr_politic_length = load_data("etr-fr-politic")
# etr_index, etr_data, etr_length = load_data("etr-fr")
# etr_index, etr_data, etr_length = load_data("wikilarge-fr")

# nbr_shot = 5

# a = '''
# Le facile à lire et à comprendre (FALC) est une méthode qui a pour but de traduire un langage classique en langage compréhensible par tous. 
# Le texte ainsi simplifié peut être compris par les personnes handicapées mentales, 
# mais aussi par d’autres comme les personnes dyslexiques, malvoyantes, les personnes âgées, les personnes qui maîtrisent mal le français.

# Voici des principes de clarté et d’accessibilité suivants pour que tu puisse rendre un texte facile à lire et à comprendre  :
# - Donne uniquement les informations essentielles. Évite la surcharge d’informations.
# - Présente les informations dans un ordre logique et facile à suivre.
# - Mets en avant l’information principale dès le début.
# - Regroupe ensemble les informations qui parlent du même sujet.
# - Répète les informations importantes si cela aide à la compréhension.
# - Utilise des phrases courtes et simples.
# - Choisis des mots faciles à comprendre.
# - Explique clairement les mots difficiles, et répète l’explication si besoin.
# - Utilise un langage adapté aux personnes concernées.
# - Emploie le même mot pour parler de la même chose tout au long du texte.
# - Évite les idées abstraites, les métaphores et les comparaisons complexes.
# - Ne recours pas à des mots étrangers ou peu connus sans explication.
# - Ne pas utiliser de mots contractés ou de style "texto".
# - Adresse-toi directement au lecteur, de manière claire et accessible.
# - Veille à ce que les pronoms soient toujours clairs et non ambigus.
# - Privilégie des formulations positives plutôt que négatives.
# - Utilise la voix active autant que possible.
# - Choisis une ponctuation simple.
# - Utilise des puces ou des numéros pour les listes, plutôt que des virgules.
# - Écris les nombres en chiffres (ex. : 1, 2, 3), pas en lettres.
# - Explique les sigles dès leur première apparition.\n- N’utilise pas d’abréviations non expliquées.
# - Écris les dates en toutes lettres pour plus de clarté.
# - Limite l’usage des pourcentages ou grands nombres, et explique-les simplement.
# - N’utilise pas de caractères spéciaux inutiles.
# - Utilise des exemples concrets pour illustrer les idées complexes.
# - Privilégie des exemples issus de la vie quotidienne.'''

# c='''\n\n Réécris le texte suivant en prenant en compte touts les principes de clarté et d’accessibilité précédents. Garde le contexte tel quel, n'ajoute ni titre ni section supplémentaire ni saut de ligne. \n\n Entoure ta réponse par des balises "@@@" comme dans les exemples précédents.'''

# shot_prompt = "\n\n Voici quelques exemples de transformation de texte en texte facile à lire et à comprendre.\n\n"
# pred = []
# ref = []
# src = []

# for i in tqdm(range(etr_length)) :
#     idx = similarity_search.get_most_similar_id(etr_index, i, nbr_shot+1)
#     prompt = shot_prompt
    
#     for j,k in enumerate(idx) :
#         prompt += "Exemple "+str(j)+"\n"
#         prompt += "Texte original :\n"
#         prompt += '"'+etr_data[k]['original']+'"\n'
#         prompt += "Traduction en texte facile à lire et à comprendre :\n"
#         prompt += "@@@"+etr_data[k]['falc']+"@@@"
#         prompt += '\n\n'
        
#     messages = [
#         {"role": "system", "content": a+prompt+c+"Traduction en texte facile à lire et à comprendre :\n"},
#         {"role": "user", "content": etr_data[i]['original']},
#     ]
    
#     response = pipe(messages,max_new_tokens=8000)
#     match = re.search('@@@(.*?)@@@', response[0]['generated_text'][-1]['content'],re.DOTALL)
#     pred.append(match.group(1) if match else response[0]['generated_text'][-1]['content'].replace("@@@", ""))
#     ref.append(etr_data[i]['falc'])
#     src.append(etr_data[i]['original'])
#     # print("#################################################################")
#     # print("System")
#     # print(a+prompt+c)
#     # print("\nInput")
#     # print(etr_politic_data[i]['original'])
#     # print(response[0]['generated_text'][-1]['content'])
        
