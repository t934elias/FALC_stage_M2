from ollama import chat
import json
# from not_used import dictionnaire

data = []

with open('..\data\etr-fr\sources\\test.json', 'r') as file:
    for line in file:
        if line.strip(): 
            data.append(json.loads(line))

# Ekman’s Basic Emotions: Anger, disgust, fear, joy, sadness, surprise.
# Plutchik's Wheel of Emotion: Adds trust and anticipation


# voici le dictionnaire des emotions associer a chaque mot, pour savoir lequel choisir:
# {dictionnaire.get_feel()}

def build_system_prompt():
    return f"""
Le facile à lire et à comprendre (FALC) est une méthode qui a pour but de traduire un langage classique en langage compréhensible par tous. 
Le texte ainsi simplifié peut être compris par les personnes handicapées mentales, mais aussi par d'autres comme les personnes dyslexiques, malvoyantes, les personnes âgées, les personnes qui maîtrisent mal le français.
Les emotions jouent aussi un role tres important dans la comprehension donc c'est essentiel de conserver les emotions presentes dasn le texte le plus que possible.

Principes de clarté et d'accessibilité :
- Donne uniquement les informations essentielles.
- Présente les informations dans un ordre logique.
- Mets l'information principale au début.
- Utilise des phrases courtes et simples.
- Choisis des mots faciles.
- Explique les mots difficiles.
- Utilise un langage clair et direct.
- Évite les idées abstraites.
- Utilise des exemples simples si nécessaire.
- Choisis une ponctuation simple.

Règles émotionnelles :
- Identifier l'émotion principale (peur, joie, colère, tristesse, surprise)
- Garder la même emotion dans le texte FALC
- Utiliser des mots émotionnels simples pour montrer cette émotion
- Garder la même intensité émotionnelle que dans le texte original

Étapes obligatoires :
1. Identifier l'émotion principale du texte
2. Associer cette émotion à un lexique simple
3. Réécrire le texte en FALC avec cette émotion

Réponds uniquement avec le texte réécrit en FALC tout en conservant et exprimant clairement les émotions. Pas d'explication. Garde le contexte tel quel, n'ajoute ni titre ni section supplémentaire ni saut de ligne
"""

few_shot_messages = [
    {
        "role": "user",
        "content": "Réécris ce texte en FALC : \"J'ai fini la bouteille qui etait au frais, il faut aller en chercher une nouvelle dans le garage\", les informe le pere de Jules. Sa mere se leve et part dans le garage. A la grande surprise de Jules, son pere lui propose une grande balade en velo tous ensemble aujourd'hui avec un pique-nique pres du lac que le petit garçon aime bien. Jules est ravi, mais egalement embete pour Bou qu'il doit normalement retrouver dans le jardin. Au moment ou Jules ouvre la bouche pour repondre, le pere et le fils entendent un cri strident. \"Cela vient du garage ! C'est maman ! \" Les garçons se precipitent pour voir ce qu'il se passe. \" Une souris! \" La mere de Jules est perchee sur un escabeau et continue a hurler. Son pere s'empare d'un balai et tente de la chasser avec. Sans succes."  
    },
    {
        "role": "assistant",
        "content": " Maman dit : \"Il n'y a plus de lait sur la table. Je vais chercher du lait dans le garage.\" Soudain, maman crie tres fort. \" Au secours, au secours ! \" Jules et son papa courent dans le garage. Maman a peur. \" Une souris ! Une souris !\" Papa attrape un balai et dit : \"Je vais attraper la souris !\" "
    },
    {
        "role": "user",
        "content": "Réécris ce texte en FALC : Et puis, d'un coup, tout se deroule tres vite. Bou surgit et, en trois bonds, attrape la souris. La mere de Jules retrouve son calme et son pere lache son balai. \" Bou ! s'exclame Jules admiratif. - Bou ? lui lancent ses parents interrogatifs. - Tu l'aimes vraiment ce chat, soupire sa mere. - Oui, lui repond-il, je te l'ai dit, c'est mon ami ! Regarde, il t'a debarrasse de cette souris! Avec Bou a la maison, tu n'en auras plus une qui traine ! - Il n'a pas tort, fait remarquer son pere. - Vous etes donc tous les deux contre moi, s'exclame la mere de Jules. Et il n'a pas de famille, ce Bou ? - Non, il a ete abandonne par son ancienne famille et maintenant il est tout seul. Maman, est-ce que Bou peut rester ? la supplie Jules. Sa mere regarde son mari qui hausse les epaules l'air de dire pourquoi pas puis repond : - D'accord, Bou peut rester mais c'est bien parce qu'il nous a debarrasse de cette souris! Et je te previens, c'est toi qui passeras l'aspirateur pour enlever ses poils qui trainent et a la premiere griffure sur mes rideaux il prend la porte. On est bien d'accord ! - Merci ! s'exclame Jules en serrant sa mere dans ses bras."
    },
    {
        "role": "assistant",
        "content": "Soudain, Bou arrive. Bou attrape la souris. \" J'ai eu tres peur, merci Bou \". \" Maman, Bou n'a pas de famille. Est-ce que Bou peut rester a la maison ? \"  \" Oui, mon cheri. Si tu me promets de t'occuper de Bou. \" \" Oh ! Oui, promis ! Merci maman ! Super ! \""
    },
    {
        "role": "user",
        "content": "Réécris ce texte en FALC : \"Me voici revenue au village. Je m'installe sur mon lit pour ecrire mon journal. Je n'arrete pas de penser a ce trafic d'elephant. Je voudrais en parler a mes parents, mais ils sont occupes a discuter avec le guide. Ah, si Lucas et Flairtout etaient la pour m'aider a en savoir plus! Tant pis, je vais mener mon enquete toute seule. Les enfants m'ont parle d'un sorcier, je vais lui demander s'il connait les braconniers. Le journal de Manon s'interrompt sur ces mots. Lucas referme doucement le cahier et le repose a sa place, sous l'oreiller. Il dit a Flairtout qu'ils n'ont pas le choix : ils doivent suivre les traces de leur amie, quitte a prendre des risques."
    },
    {
        "role": "assistant",
        "content": "\" C'est trop triste. Je dois vite retrouver ces mechants chasseurs. Les enfants du village m'ont parle du sorcier qui sait tout. Je pars chercher ce sorcier. Ah ! Si Lucas et Flairtout etaient la pour m'aider ! \" L'histoire de Manon est finie. Lucas ferme le cahier de Manon."
    },
    {
        "role": "user",
        "content": "Réécris ce texte en FALC : Un instant, le fantôme de Canterville demeura parfaitement immobile. Puis il lança le flacon sur le parquet et s'enfuit le long du couloir en poussant des gémissements."
    },
    {
        "role": "assistant",
        "content": "Le fantôme est vexé. Il lance le flacon huile sur le sol. Il fuit dans le couloir."
    },
]

src = data[0]['original']
ref = data[0]['falc']

def pred_falc(src):
    response = chat(
        model='ministral-3:3b',
        messages=[
            {
                'role': 'system',
                'content': build_system_prompt(),
            },
            # *few_shot_messages,
            {
                'role': 'user',
                # 'content': f"Réécris ce texte en FALC : {src}",
                'content': f"{src}"
            },
        ],
        options={
            "temperature": 0.4, # 0 : Deterministic output (greedy)
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "num_predict": 100,
            # "stop": ["\n"], # stop sequence
            "num_ctx": 12288 #8192#32768#4096 # increase context
        }
    )

    pred = response['message']['content']
    return pred

pred = pred_falc(src)
print(pred)