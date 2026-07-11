"""
Shared slop lexicons.

Single source of truth for the phrase lists used by the calibrated scorer
(quirky.detector.calibrate), the span scorer (quirky.detector.spans), and the
active fixer (quirky.fix.text). Keep entries lowercase; matching is
case-insensitive on word boundaries.
"""

# Boilerplate transitions and LLM tropes -- the strongest lexical AI tells.
# Superset of the cliché list in formulas.compute_text_ai_score and the plain
# forms of TextHumanizer.AI_REPLACEMENTS keys.
AI_CLICHES: tuple[str, ...] = (
    "furthermore",
    "moreover",
    "in conclusion",
    "in summary",
    "it is important to note",
    "it is worth noting",
    "it should be noted",
    "as an ai",
    "as an ai language model",
    "firstly",
    "secondly",
    "lastly",
    "consequently",
    "additionally",
    "delve into",
    "delve",
    "leverage",
    "utilize",
    "facilitate",
    "robust",
    "seamless",
    "seamlessly",
    "cutting-edge",
    "state-of-the-art",
    "game-changer",
    "in today's fast-paced world",
    "in the ever-evolving landscape",
    "navigate the complexities",
    "unlock the potential",
    "a testament to",
    "underscores the importance",
    "plays a crucial role",
    "plays a pivotal role",
    "it is essential to",
    "in order to",
    "at the end of the day",
    "needless to say",
    "when it comes to",
    "harness the power",
    "embark on a journey",
    "dive deep",
    "elevate",
    "streamline",
    "holistic",
    "synergy",
    "paradigm",
)

# Hedging phrases -- confidence-diluting qualifiers.
HEDGES: tuple[str, ...] = (
    "arguably",
    "it could be argued",
    "one might say",
    "in some ways",
    "to some extent",
    "somewhat",
    "generally speaking",
    "broadly speaking",
    "more or less",
    "may potentially",
    "could potentially",
    "potentially",
    "perhaps",
    "possibly",
    "it seems that",
    "it appears that",
    "it is likely that",
    "tend to",
    "tends to",
    "relatively",
    "fairly",
    "quite possibly",
    "in most cases",
    "for the most part",
)

# Filler and over-explanation padding -- removable without meaning loss.
FILLERS: tuple[str, ...] = (
    "it is important to note that",
    "it is worth noting that",
    "it should be noted that",
    "as previously mentioned",
    "as mentioned earlier",
    "as mentioned above",
    "needless to say",
    "in other words",
    "to put it simply",
    "simply put",
    "basically",
    "essentially",
    "actually",
    "really",
    "very",
    "just",
    "in fact",
    "as a matter of fact",
    "at this point in time",
    "in the process of",
    "due to the fact that",
    "in light of the fact that",
    "for all intents and purposes",
    "each and every",
    "first and foremost",
    "last but not least",
    "the fact of the matter is",
    "when all is said and done",
)

# Stopwords for the lexical-overlap (Jaccard) drift guard. Function words only;
# content words and numbers are what the guard must preserve.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the this that these those there here
    i you he she it we they me him her us them my your his its our their
    is are was were be been being am
    do does did doing done have has had having
    will would shall should can could may might must
    and or but nor so yet for if then else when while because since although
    though whereas whether
    to of in on at by with from as into onto upon about above below over under
    between among through during before after against without within along
    across behind beyond off out up down
    not no what which who whom whose where why how all any both each few more
    most other some such
    also just very really quite rather somewhat
    """.split()
)
