# from ollama import chat


# def build_prompt(input_text):
#     return f"""
# Le facile à lire et à comprendre (FALC) est une méthode qui a pour but de traduire un langage classique en langage compréhensible par tous. 
# Le texte ainsi simplifié peut être compris par les personnes handicapées mentales, 
# mais aussi par d'autres comme les personnes dyslexiques, malvoyantes, les personnes âgées, les personnes qui maîtrisent mal le français.

# Voici des principes de clarté et d'accessibilité suivants pour que tu puisse rendre un texte facile à lire et à comprendre  :
# - Donne uniquement les informations essentielles. Évite la surcharge d'informations.
# - Présente les informations dans un ordre logique et facile à suivre.
# - Mets en avant l'information principale dès le début.
# - Regroupe ensemble les informations qui parlent du même sujet.
# - Répète les informations importantes si cela aide à la compréhension.
# - Utilise des phrases courtes et simples.
# - Choisis des mots faciles à comprendre.
# - Explique clairement les mots difficiles, et répète l'explication si besoin.
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
# - Explique les sigles dès leur première apparition.\n- N'utilise pas d'abréviations non expliquées.
# - Écris les dates en toutes lettres pour plus de clarté.
# - Limite l'usage des pourcentages ou grands nombres, et explique-les simplement.
# - N'utilise pas de caractères spéciaux inutiles.
# - Utilise des exemples concrets pour illustrer les idées complexes.
# - Privilégie des exemples issus de la vie quotidienne.
#  \n\n Réécris le texte suivant en prenant en compte touts les principes de clarté et d'accessibilité précédents. Garde le contexte tel quel, n'ajoute ni titre ni section supplémentaire ni saut de ligne.

# Texte : {input_text}

# Réponse :
# """

# # texte = "Un instant, le fantôme de Canterville demeura parfaitement immobile. Puis il lança le flacon sur le parquet et s'enfuit le long du couloir en poussant des gémissements."
# texte = "La cloche sonne. C'est la fin de l'annee, tant attendue par tous les eleves. Une ruee de jeunes franchit la grille du college avec des hurlements de joie. Lucas traverse la cour d'un pas leger. \" Salut, bonnes vacances! \", lui crient ses amis en lui faisant des signes de la main."

# prompt = build_prompt(texte)

# prompt2 = f"""
# Réécris le texte suivant en prenant en compte touts les principes de clarté et d'accessibilité précédents du FALC. Garde le contexte tel quel, n'ajoute ni titre ni section supplémentaire ni saut de ligne. 
# Texte : {texte}
# Réponse :
# """

# # phi3
# # ministral-3
# # gemma2:2b
# # phi4-mini-reasoning:latest
# # phi4-mini:latest

# response = chat(model='ministral-3:3b', messages=[
#   {
#     'role': 'system',
#     'content': prompt,
#   },
#   {
#     'role': 'user',
#     'content': texte,
#   },
# ])
# print(response.message.content)





from ollama import chat

def build_system_prompt():
    return """
Le facile à lire et à comprendre (FALC) est une méthode qui a pour but de traduire un langage classique en langage compréhensible par tous. 
Le texte ainsi simplifié peut être compris par les personnes handicapées mentales, mais aussi par d'autres comme les personnes dyslexiques, malvoyantes, les personnes âgées, les personnes qui maîtrisent mal le français.

Voici des principes de clarté et d'accessibilité :
- Donne uniquement les informations essentielles.
- Présente les informations dans un ordre logique.
- Mets l'information principale au début.
- Utilise des phrases courtes et simples.
- Choisis des mots faciles.
- Explique les mots difficiles.
- Utilise un langage clair et direct.
- Évite les idées abstraites.
- Utilise des exemples simples si nécessaire.

Réponds uniquement avec le texte réécrit en FALC. Pas d'explication.
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

# texte = "La cloche sonne. C'est la fin de l'annee, tant attendue par tous les eleves. Une ruee de jeunes franchit la grille du college avec des hurlements de joie. Lucas traverse la cour d'un pas leger. \" Salut, bonnes vacances! \", lui crient ses amis en lui faisant des signes de la main."
texte = "C'est un des plus beaux jours de sa vie ! D'autant plus que maintenant, le p\u00e8re de Jules a pr\u00e9par\u00e9 les v\u00e9los et sa m\u00e8re le pique-nique. Ils sont tous partis. Et devinez o\u00f9 est Bou\u2026 dans le panier \u00e0 v\u00e9lo de maman ! Il est amusant avec ses babines au vent. C'est la meilleure promenade que Jules a faite. Ils ont pass\u00e9 une tr\u00e8s bonne journ\u00e9e tous ensemble, en famille, papa, maman, Bou et Jules !"


response = chat(
    model='ministral-3:3b',
    messages=[
        {
            'role': 'system',
            'content': build_system_prompt(),
        },
        *few_shot_messages,
        {
            'role': 'user',
            'content': f"Réécris ce texte en FALC : {texte}",
        },
    ],
    options={
        "temperature": 0.4, # 0 : Deterministic output (greedy)
        "top_p": 0.9,
        "repeat_penalty": 1.2,
        "num_predict": 100,
        # "stop": ["\n"], # stop sequence
        "num_ctx": 4096 # increase context
    }
)

print(response['message']['content'])