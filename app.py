"""
Health Chatbot — app.py
========================
Key feature: Smart Differential Diagnosis Engine
  - Collects initial symptoms from user
  - Picks the MOST DISTINGUISHING symptom to ask about next
    (the symptom that best separates the top candidate diseases)
  - Keeps asking until confidence >= 70% OR max questions reached
  - Only names a disease when confident; otherwise shows honest top-3
  - Wikipedia NLP answers work at any point in conversation

Install:
  pip install flask flask-session pandas numpy scikit-learn wikipedia spacy fuzzywuzzy python-Levenshtein
  python -m spacy download en_core_web_sm
"""

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import pandas as pd
import numpy as np
import os
import re
import traceback
import warnings

import wikipedia
import spacy
from fuzzywuzzy import fuzz, process

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

warnings.filterwarnings("ignore")

# ── spaCy ─────────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# ── Load & Train ──────────────────────────────────────────────────────────────
training = pd.read_csv("Data/Training.csv")
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]

cols  = training.columns[:-1]       # symptom columns
X     = training[cols]
y     = training["prognosis"]

le    = preprocessing.LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.33, random_state=42
)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

symptoms_dict  = {sym: i for i, sym in enumerate(cols)}
symptom_phrases = [s.replace("_", " ") for s in symptoms_dict]
phrase_to_key   = {s.replace("_", " "): s for s in symptoms_dict}

# Pre-build: for each disease, which symptoms does it have?
disease_symptom_map = {}
for disease in training["prognosis"].unique():
    row = training[training["prognosis"] == disease].iloc[0]
    disease_symptom_map[disease] = set(c for c in cols if row[c] == 1)

CONFIDENCE_THRESHOLD  = 70   # only confirm a disease above this %
MAX_CONFIRM_QUESTIONS = 12   # safety cap on questions asked

# ── Precautions Dictionary ────────────────────────────────────────────────────
PRECAUTIONS = {
    "Fungal infection":         ["Keep skin clean and dry", "Use antifungal cream", "Avoid sharing personal items", "Wear breathable clothing"],
    "Allergy":                  ["Avoid known allergens", "Carry antihistamines", "Keep windows closed during high pollen", "Wear a mask outdoors if needed"],
    "GERD":                     ["Avoid spicy and fatty foods", "Don't lie down after eating", "Eat smaller meals", "Elevate head while sleeping"],
    "Chronic cholestasis":      ["Avoid alcohol completely", "Follow low-fat diet", "Take prescribed vitamins", "Regular liver check-ups"],
    "Drug Reaction":            ["Stop the triggering drug immediately", "Consult a doctor before any new medication", "Carry a list of drug allergies", "Seek emergency care if severe"],
    "Peptic ulcer diseae":      ["Avoid NSAIDs and aspirin", "Eat small frequent meals", "Avoid spicy food and alcohol", "Take prescribed antacids"],
    "AIDS":                     ["Use protection during sex", "Never share needles", "Take antiretroviral therapy regularly", "Regular medical monitoring"],
    "Diabetes":                 ["Monitor blood sugar daily", "Follow a low-sugar diet", "Exercise regularly", "Take medication as prescribed"],
    "Gastroenteritis":          ["Drink plenty of fluids (ORS)", "Wash hands frequently", "Avoid contaminated food and water", "Rest and avoid solid food initially"],
    "Bronchial Asthma":         ["Avoid smoke and dust", "Carry inhaler at all times", "Avoid cold air exposure", "Identify and avoid triggers"],
    "Hypertension":             ["Reduce salt intake", "Exercise regularly", "Manage stress", "Take antihypertensive medication as prescribed"],
    "Migraine":                 ["Avoid bright lights and loud noise", "Stay hydrated", "Maintain regular sleep schedule", "Identify and avoid personal triggers"],
    "Cervical spondylosis":     ["Maintain good posture", "Do neck exercises", "Avoid heavy lifting", "Use ergonomic chair and desk"],
    "Paralysis (brain hemorrhage)": ["Seek emergency care immediately", "Physiotherapy after stabilization", "Control blood pressure", "Avoid stress and heavy exertion"],
    "Jaundice":                 ["Drink clean water only", "Avoid alcohol", "Rest and avoid fatty foods", "Consult a doctor for liver tests"],
    "Malaria":                  ["Use mosquito nets and repellents", "Wear full-sleeve clothing", "Take antimalarial medication if traveling", "Eliminate stagnant water near home"],
    "Chicken pox":              ["Isolate from others", "Avoid scratching blisters", "Keep skin clean", "Take prescribed antivirals if severe"],
    "Dengue":                   ["Use mosquito repellents", "Wear protective clothing", "Eliminate standing water", "Monitor platelet count and stay hydrated"],
    "Typhoid":                  ["Drink boiled or bottled water", "Eat freshly cooked food", "Get vaccinated if traveling", "Wash hands before meals"],
    "hepatitis A":              ["Get vaccinated", "Drink clean water", "Wash hands thoroughly", "Avoid raw shellfish"],
    "Hepatitis B":              ["Get vaccinated", "Avoid sharing needles or razors", "Practice safe sex", "Regular liver monitoring"],
    "Hepatitis C":              ["Avoid sharing needles", "Practice safe sex", "Don't share personal hygiene items", "Regular liver function tests"],
    "Hepatitis D":              ["Vaccinate against Hepatitis B (prevents D)", "Avoid sharing needles", "Practice safe sex", "Regular medical check-ups"],
    "Hepatitis E":              ["Drink clean water", "Avoid raw/undercooked meat", "Maintain good hygiene", "Especially important in pregnancy"],
    "Alcoholic hepatitis":      ["Stop alcohol consumption immediately", "Follow high-calorie nutritious diet", "Regular liver monitoring", "Join support program if needed"],
    "Tuberculosis":             ["Complete full course of antibiotics", "Cover mouth while coughing", "Ensure good ventilation", "Avoid close contact with others until non-infectious"],
    "Common Cold":              ["Rest and stay hydrated", "Wash hands frequently", "Avoid contact with infected people", "Use tissues and dispose properly"],
    "Pneumonia":                ["Complete full antibiotic course", "Rest and stay warm", "Stay hydrated", "Get pneumococcal vaccine"],
    "Dimorphic hemmorhoids(piles)": ["Eat high-fiber diet", "Drink plenty of water", "Avoid straining during bowel movements", "Take sitz baths"],
    "Heart attack":             ["Call emergency services immediately", "Chew aspirin if not allergic", "Keep calm and rest", "Do not eat or drink"],
    "Varicose veins":           ["Elevate legs when resting", "Wear compression stockings", "Exercise regularly", "Avoid prolonged standing or sitting"],
    "Hypothyroidism":           ["Take thyroid medication as prescribed", "Regular thyroid function tests", "Eat iodine-rich foods", "Avoid raw cruciferous vegetables in excess"],
    "Hyperthyroidism":          ["Avoid caffeine and stimulants", "Take prescribed antithyroid medication", "Regular thyroid monitoring", "Manage stress"],
    "Hypoglycemia":             ["Carry glucose tablets or juice", "Eat regular small meals", "Monitor blood sugar frequently", "Avoid skipping meals"],
    "Osteoarthritis":           ["Exercise regularly (low impact)", "Maintain healthy weight", "Use hot/cold therapy for pain", "Take prescribed anti-inflammatory medication"],
    "Arthritis":                ["Exercise gently and regularly", "Maintain healthy weight", "Apply hot/cold packs", "Take prescribed medication"],
    "Vertigo Paroymsal  Positional Vertigo": ["Avoid sudden head movements", "Do Epley maneuver exercises", "Sleep with head elevated", "Avoid caffeine and alcohol"],
    "Acne":                     ["Wash face twice daily with mild cleanser", "Avoid touching face", "Use non-comedogenic products", "Avoid popping pimples"],
    "Urinary tract infection":  ["Drink plenty of water", "Urinate frequently, don't hold", "Wipe front to back", "Complete full antibiotic course"],
    "Psoriasis":                ["Moisturize skin daily", "Avoid skin injuries", "Manage stress", "Follow prescribed treatment"],
    "Impetigo":                 ["Keep affected area clean", "Don't scratch lesions", "Wash hands frequently", "Complete antibiotic course"],
}

def get_precautions(disease: str) -> list:
    """Return precautions for a disease, with fuzzy fallback."""
    # Exact match first
    if disease in PRECAUTIONS:
        return PRECAUTIONS[disease]
    # Case-insensitive match
    for key in PRECAUTIONS:
        if key.lower() == disease.lower():
            return PRECAUTIONS[key]
    # Fuzzy match
    keys = list(PRECAUTIONS.keys())
    match, score = process.extractOne(disease, keys, scorer=fuzz.token_sort_ratio)
    if score >= 70:
        return PRECAUTIONS[match]
    return ["Consult a doctor for specific precautions",
            "Rest and stay hydrated",
            "Avoid self-medication",
            "Seek professional medical advice"]


# ── Local disease descriptions (shown when Wikipedia is unavailable) ──────────
DISEASE_INFO = {
    "Fungal infection":       "A fungal infection occurs when a fungus invades body tissue and causes disease. Common types include athlete's foot, ringworm, and candidiasis.",
    "Allergy":                "An allergy is an immune system response to a foreign substance that is not typically harmful. Symptoms range from mild sneezing to severe anaphylaxis.",
    "GERD":                   "Gastroesophageal reflux disease (GERD) is a chronic condition where stomach acid flows back into the esophagus, causing heartburn and irritation.",
    "Chronic cholestasis":    "Chronic cholestasis is reduced or absent bile flow from the liver, leading to accumulation of bile acids and damage to liver cells.",
    "Drug Reaction":          "A drug reaction is an unwanted or harmful response to a medication. Reactions range from mild rashes to life-threatening anaphylaxis.",
    "Peptic ulcer diseae":    "A peptic ulcer is a sore on the lining of the stomach or duodenum, usually caused by H. pylori bacteria or overuse of NSAIDs.",
    "AIDS":                   "AIDS (Acquired Immunodeficiency Syndrome) is the advanced stage of HIV infection, severely weakening the immune system.",
    "Diabetes":               "Diabetes is a chronic disease that affects how the body processes blood sugar. Type 1 and Type 2 are the most common forms.",
    "Gastroenteritis":        "Gastroenteritis is inflammation of the stomach and intestines, typically caused by a viral or bacterial infection, causing vomiting and diarrhea.",
    "Bronchial Asthma":       "Bronchial asthma is a chronic lung disease causing inflammation and narrowing of airways, leading to wheezing, shortness of breath, and coughing.",
    "Hypertension":           "Hypertension (high blood pressure) is a common condition where the force of blood against artery walls is consistently too high.",
    "Migraine":               "A migraine is a severe recurring headache, often accompanied by nausea, vomiting, and sensitivity to light and sound.",
    "Cervical spondylosis":   "Cervical spondylosis is age-related wear of the spinal discs in the neck, causing pain and stiffness.",
    "Jaundice":               "Jaundice is a yellowing of the skin and eyes caused by excess bilirubin in the blood, often indicating liver disease.",
    "Malaria":                "Malaria is a life-threatening disease caused by parasites transmitted through infected mosquito bites, causing fever and chills.",
    "Chicken pox":            "Chickenpox is a highly contagious viral infection causing an itchy blister-like rash, fever, and fatigue.",
    "Dengue":                 "Dengue fever is a mosquito-borne viral infection causing high fever, severe headache, joint pain, and rash.",
    "Typhoid":                "Typhoid fever is a bacterial infection caused by Salmonella typhi, spread through contaminated food and water.",
    "hepatitis A":            "Hepatitis A is a viral liver infection spread through contaminated food and water. It is usually self-limiting.",
    "Hepatitis B":            "Hepatitis B is a viral infection that attacks the liver and can cause chronic disease and liver cancer.",
    "Hepatitis C":            "Hepatitis C is a blood-borne viral infection that causes liver inflammation, sometimes leading to cirrhosis.",
    "Alcoholic hepatitis":    "Alcoholic hepatitis is inflammation of the liver caused by heavy alcohol consumption over time.",
    "Tuberculosis":           "Tuberculosis (TB) is a bacterial infection primarily affecting the lungs, spread through airborne droplets.",
    "Common Cold":            "The common cold is a viral infection of the upper respiratory tract, causing runny nose, sore throat, and cough.",
    "Pneumonia":              "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid.",
    "Heart attack":           "A heart attack occurs when blood flow to the heart is blocked, causing muscle damage. It requires immediate emergency care.",
    "Hypothyroidism":         "Hypothyroidism occurs when the thyroid gland does not produce enough hormones, causing fatigue, weight gain, and depression.",
    "Hyperthyroidism":        "Hyperthyroidism is when the thyroid gland produces too much hormone, causing rapid heartbeat, weight loss, and anxiety.",
    "Hypoglycemia":           "Hypoglycemia is abnormally low blood sugar, causing shakiness, sweating, and confusion. Common in diabetic patients.",
    "Osteoarthritis":         "Osteoarthritis is a degenerative joint disease causing cartilage breakdown, leading to pain and stiffness.",
    "Arthritis":              "Arthritis is inflammation of joints causing pain, swelling, and reduced range of motion.",
    "Acne":                   "Acne is a skin condition caused by clogged hair follicles with oil and dead skin cells, causing pimples and blackheads.",
    "Urinary tract infection":"A urinary tract infection (UTI) is a bacterial infection in any part of the urinary system, causing burning during urination.",
    "Psoriasis":              "Psoriasis is a chronic autoimmune skin condition causing rapid skin cell buildup, resulting in scales and red patches.",
    "Impetigo":               "Impetigo is a highly contagious bacterial skin infection causing red sores that rupture and form honey-colored crusts.",
    "Dengue":                 "Dengue fever is a mosquito-borne tropical disease causing high fever, headaches, and joint pain.",
    "Varicose veins":         "Varicose veins are enlarged, twisted veins, most commonly in the legs, caused by weakened vein valves.",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids are swollen veins in the rectum or anus causing discomfort, bleeding, and itching.",
}

# Diseases unlikely in young patients (penalise if age < 35)
ELDERLY_DISEASES = {
    "Heart attack", "Osteoarthritis", "Cervical spondylosis",
    "Varicose veins", "Hypertension", "Hypothyroidism",
    "Paralysis (brain hemorrhage)"
}
# Diseases unlikely in older patients
YOUNG_DISEASES = {"Acne", "Chicken pox", "Impetigo"}
# Diseases more common in females
FEMALE_DISEASES = {"Urinary tract infection", "Migraine"}
# Diseases more common in males
MALE_DISEASES   = {"Alcoholic hepatitis"}


def _rerank_by_demographics(top3: list, age: int, gender: str) -> list:
    """
    Adjust confidence scores based on patient age and gender.
    Penalises statistically unlikely diseases for this patient profile.
    """
    adjusted = []
    for disease, conf in top3:
        penalty = 0
        if age < 35 and disease in ELDERLY_DISEASES:
            penalty += 20
        if age > 60 and disease in YOUNG_DISEASES:
            penalty += 15
        if gender in ("female", "f") and disease in MALE_DISEASES:
            penalty += 15
        if gender in ("male", "m") and disease in FEMALE_DISEASES:
            penalty += 10
        adjusted.append((disease, max(1.0, round(conf - penalty, 2))))

    # Re-sort by adjusted confidence
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


def get_disease_description(disease: str) -> str:
    """Try Wikipedia with medical filter first; fall back to local descriptions."""
    try:
        results = wikipedia.search(f"{disease} disease medical", results=4)
        for result in results:
            try:
                summary = wikipedia.summary(result, sentences=2, auto_suggest=False)
                if _is_medical_result(summary):
                    return summary
            except wikipedia.DisambiguationError as e:
                try:
                    s = wikipedia.summary(e.options[0], sentences=2, auto_suggest=False)
                    if _is_medical_result(s):
                        return s
                except Exception:
                    continue
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: local dictionary
    if disease in DISEASE_INFO:
        return DISEASE_INFO[disease]
    keys = list(DISEASE_INFO.keys())
    try:
        match, score = process.extractOne(disease, keys, scorer=fuzz.token_sort_ratio)
        if score >= 70:
            return DISEASE_INFO[match]
    except Exception:
        pass
    return f"{disease} is a medical condition. Please consult a doctor for detailed information."


# ── Synonym Map ───────────────────────────────────────────────────────────────
SYNONYM_MAP = {
    "high fever":           "high_fever",
    "mild fever":           "mild_fever",
    "low fever":            "mild_fever",
    "slight fever":         "mild_fever",
    "stomach pain":         "abdominal_pain",
    "belly pain":           "abdominal_pain",
    "tummy pain":           "abdominal_pain",
    "stomach ache":         "abdominal_pain",
    "gut pain":             "abdominal_pain",
    "dry cough":            "cough",
    "sore throat":          "throat_irritation",
    "scratchy throat":      "throat_irritation",
    "throwing up":          "vomiting",
    "throw up":             "vomiting",
    "threw up":             "vomiting",
    "can't sleep":          "lack_of_sleep",
    "cant sleep":           "lack_of_sleep",
    "no sleep":             "lack_of_sleep",
    "trouble sleeping":     "lack_of_sleep",
    "tired":                "fatigue",
    "exhausted":            "fatigue",
    "very tired":           "fatigue",
    "no energy":            "fatigue",
    "weakness":             "weakness_in_limbs",
    "weak":                 "weakness_in_limbs",
    "skin rash":            "skin_rash",
    "rashes":               "skin_rash",
    "itchy skin":           "itching",
    "itchy":                "itching",
    "weight loss":          "weight_loss",
    "losing weight":        "weight_loss",
    "chest pain":           "chest_pain",
    "chest ache":           "chest_pain",
    "back pain":            "back_pain",
    "back ache":            "back_pain",
    "joint pain":           "joint_pain",
    "painful joints":       "joint_pain",
    "muscle pain":          "muscle_pain",
    "muscle ache":          "muscle_pain",
    "body ache":            "muscle_pain",
    "body pain":            "muscle_pain",
    "night sweats":         "sweating",
    "shivering":            "chills",
    "feeling sick":         "nausea",
    "upset stomach":        "nausea",
    "loss of appetite":     "loss_of_appetite",
    "not eating":           "loss_of_appetite",
    "no appetite":          "loss_of_appetite",
    "diarrhea":             "diarrhoea",
    "loose stool":          "diarrhoea",
    "loose motion":         "diarrhoea",
    "watery stool":         "diarrhoea",
    "dark urine":           "dark_urine",
    "yellow eyes":          "yellowing_of_eyes",
    "yellow skin":          "yellowish_skin",
    "jaundice":             "yellowish_skin",
    "shortness of breath":  "breathlessness",
    "difficulty breathing": "breathlessness",
    "hard to breathe":      "breathlessness",
    "blurry vision":        "blurred_and_distorted_vision",
    "blurred vision":       "blurred_and_distorted_vision",
    "swollen legs":         "swollen_legs",
    "swollen feet":         "swollen_legs",
    "mood swings":          "mood_swings",
    "memory loss":          "loss_of_memory",
    "memory problem":       "loss_of_memory",
    "memory issue":         "loss_of_memory",
    "forgetting things":    "loss_of_memory",
    "can't remember":       "loss_of_memory",
    "cant remember":        "loss_of_memory",
    "poor memory":          "loss_of_memory",
    "forgetful":            "loss_of_memory",
    "stiff neck":           "stiff_neck",
    "neck pain":            "stiff_neck",
    "skin peeling":         "skin_peeling",
    "peeling skin":         "skin_peeling",
    "blisters":             "skin_rash",
    "fast heartbeat":       "fast_heart_rate",
    "rapid heartbeat":      "fast_heart_rate",
    "palpitations":         "palpitations",
    "heartburn":            "acidity",
    "acid reflux":          "acidity",
    "frequent urination":   "polyuria",
    "urinating a lot":      "polyuria",
    "excessive thirst":     "excessive_hunger",
    "always thirsty":       "excessive_hunger",
    "pus":                  "dischromic_patches",
    "sunken eyes":          "sunken_eyes",
    "bloating":             "abdominal_pain",
    "gas":                  "abdominal_pain",
}

WIKI_STOP = {
    "what","is","are","tell","me","about","explain","define","describe",
    "information","info","meaning","know","learn","show","give","please",
    "can","you","the","a","an","how","does","do","cause","causes",
    "symptom","symptoms","treatment","treat","cure","help","hey","hi",
}
WIKI_TRIGGERS = [
    "what is", "what are", "tell me about", "explain", "define",
    "describe", "how does", "what causes", "how to treat",
    "treatment for", "cure for", "information about", "info on",
    "info about", "what do you know about",
]


# ── NLP Helpers ───────────────────────────────────────────────────────────────
def is_wiki_question(text: str) -> bool:
    t = text.lower().strip()
    return any(t.startswith(tr) or tr in t for tr in WIKI_TRIGGERS)


def extract_wiki_query(text: str) -> str:
    t = text.lower().strip()
    for phrase in ["what is","what are","tell me about","explain","define",
                   "describe","give me info on","information about",
                   "how does","what causes","how to treat",
                   "treatment for","cure for"]:
        t = t.replace(phrase, " ")
    doc = nlp(t)
    keywords = [
        token.text for token in doc
        if token.pos_ in {"NOUN","PROPN","ADJ"}
        and token.text not in WIKI_STOP
        and not token.is_stop
        and len(token.text) > 2
    ]
    if not keywords:
        keywords = [w for w in t.split() if w not in WIKI_STOP and len(w) > 2]
    return " ".join(keywords).strip() or t.strip()



# Keywords that indicate a Wikipedia result is medically relevant
MEDICAL_KEYWORDS = {
    "disease", "disorder", "condition", "syndrome", "infection", "symptom",
    "treatment", "therapy", "diagnosis", "medical", "health", "patient",
    "clinical", "chronic", "acute", "virus", "bacteria", "inflammation",
    "immune", "organ", "tissue", "cell", "nerve", "muscle", "blood",
    "skin", "lung", "liver", "kidney", "heart", "stomach", "intestin",
    "cancer", "tumor", "pain", "fever", "rash", "cough", "nausea",
    "vomit", "diarrhea", "fatigue", "headache", "allergy", "asthma",
    "diabetes", "hypertension", "medicine", "drug", "antibiotic",
    "vaccine", "surgery", "hospital", "doctor", "physician", "pathogen",
    "contagious", "hereditary", "genetic", "autoimmune", "hormone",
    "deficiency", "toxin", "overdose", "wound", "injury", "fracture",
}

def _is_medical_result(text: str) -> bool:
    """Return True if the Wikipedia summary is medically relevant."""
    text_lower = text.lower()
    matches = sum(1 for kw in MEDICAL_KEYWORDS if kw in text_lower)
    return matches >= 2   # at least 2 medical keywords = relevant


def wiki_answer(query: str) -> str:
    """
    Only answers health/disease/symptom questions.
    Returns 1-2 sentences max. Rejects non-medical topics.
    """
    try:
        medical_query = f"{query} disease medical condition"
        results = wikipedia.search(medical_query, results=6)
        if not results:
            results = wikipedia.search(query, results=4)
        if not results:
            return _local_description_fallback(query)

        for result in results:
            try:
                # 2 sentences max — short and accurate
                summary = wikipedia.summary(result, sentences=2, auto_suggest=False)
                if not _is_medical_result(summary):
                    continue
                # Strip to first 2 sentences cleanly
                sentences = summary.split(". ")
                short = ". ".join(sentences[:2]).strip()
                if not short.endswith("."):
                    short += "."
                return short
            except wikipedia.DisambiguationError as e:
                for option in e.options[:3]:
                    try:
                        s = wikipedia.summary(option, sentences=2, auto_suggest=False)
                        if _is_medical_result(s):
                            sentences = s.split(". ")
                            short = ". ".join(sentences[:2]).strip()
                            if not short.endswith("."):
                                short += "."
                            return short
                    except Exception:
                        continue
            except Exception:
                continue

        return _local_description_fallback(query)

    except Exception:
        return _local_description_fallback(query)


def _local_description_fallback(query: str) -> str:
    """
    Try to match the query to a known disease in DISEASE_INFO.
    If nothing matches, return a polite refusal instead of wrong info.
    """
    query_lower = query.lower().strip()

    # Direct match
    for disease, desc in DISEASE_INFO.items():
        if disease.lower() in query_lower or query_lower in disease.lower():
            return f"**{disease}**\n\n{desc}"

    # Fuzzy match
    keys = list(DISEASE_INFO.keys())
    try:
        match, score = process.extractOne(query, keys, scorer=fuzz.token_sort_ratio)
        if score >= 65:
            return f"**{match}**\n\n{DISEASE_INFO[match]}"
    except Exception:
        pass

    return (
        f"I can only answer health and disease related questions.\n"
        f"I couldn't find medical information about '{query}'.\n\n"
        f"Try asking about a specific disease or symptom, e.g.:\n"
        f"  • 'What is diabetes?'\n"
        f"  • 'What is fever?'\n"
        f"  • 'Explain asthma'"
    )



def extract_symptoms(text: str) -> list:
    """4-layer NLP symptom extraction."""
    text_lower = text.lower().strip()
    found = set()

    # Layer 1: spaCy noun chunks + lemmas → direct + fuzzy
    doc = nlp(text_lower)
    candidates = (
        [chunk.text for chunk in doc.noun_chunks] +
        [token.lemma_ for token in doc if not token.is_stop and not token.is_punct] +
        [text_lower]
    )
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        for sym in symptoms_dict:
            if sym.replace("_", " ") in candidate:
                found.add(sym)
        match, score = process.extractOne(candidate, symptom_phrases, scorer=fuzz.token_sort_ratio)
        if score >= 82:
            found.add(phrase_to_key[match])

    # Layer 2: Synonym map (longest phrases first)
    for phrase in sorted(SYNONYM_MAP, key=len, reverse=True):
        if phrase in text_lower:
            key = SYNONYM_MAP[phrase]
            if key in symptoms_dict:
                found.add(key)

    # Layer 3: Direct substring
    for sym in symptoms_dict:
        if sym.replace("_", " ") in text_lower:
            found.add(sym)

    # Layer 4: Single-word fuzzy (typo tolerance)
    for word in text_lower.split():
        if len(word) < 4:
            continue
        match, score = process.extractOne(word, symptom_phrases, scorer=fuzz.ratio)
        if score >= 85:
            found.add(phrase_to_key[match])

    return list(found)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_disease(symptoms: list):
    """Returns (best_disease, confidence_pct, top3_list)."""
    try:
        vector = np.zeros(len(symptoms_dict))
        for s in symptoms:
            if s in symptoms_dict:
                vector[symptoms_dict[s]] = 1
        probs   = model.predict_proba([vector])[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top3    = [(le.inverse_transform([i])[0], round(probs[i] * 100, 2)) for i in top_idx]
        return top3[0][0], top3[0][1], top3
    except Exception:
        return "System Error", 0, []


# ── Differential Diagnosis Engine ────────────────────────────────────────────
def pick_best_question(current_symptoms: list, asked_symptoms: set) -> str | None:
    """
    Finds the single symptom that BEST DISTINGUISHES between the top candidate
    diseases given what we know so far.

    Strategy:
      1. Get top-5 candidate diseases from model
      2. For each symptom not yet asked and not already confirmed:
           score = how differently it splits the top diseases
           (present in some but not all → high score)
      3. Return the highest-scoring symptom

    This means we ask the question most likely to eliminate wrong diseases.
    """
    _, _, top5_raw = predict_disease(current_symptoms)
    # Use top 5 for better coverage
    vector = np.zeros(len(symptoms_dict))
    for s in current_symptoms:
        if s in symptoms_dict:
            vector[symptoms_dict[s]] = 1
    probs   = model.predict_proba([vector])[0]
    top_idx = np.argsort(probs)[::-1][:5]
    top5_diseases = [le.inverse_transform([i])[0] for i in top_idx]

    best_sym   = None
    best_score = -1

    already_confirmed = set(current_symptoms)

    for sym in symptoms_dict:
        if sym in asked_symptoms or sym in already_confirmed:
            continue

        # How many of the top-5 diseases have this symptom?
        present_in = sum(
            1 for d in top5_diseases
            if sym in disease_symptom_map.get(d, set())
        )

        # Ideal: present in ~half the candidates → maximum discrimination
        # Score peaks when present_in == len(top5)/2
        n = len(top5_diseases)
        # information-gain-like score: maximised at 50/50 split
        if present_in == 0 or present_in == n:
            score = 0   # useless — either all have it or none do
        else:
            # proportion-based balance score
            p = present_in / n
            score = 1 - abs(p - 0.5) * 2   # 1.0 at 50%, 0.0 at 0% or 100%

            # Boost symptoms that appear in the TOP disease (more relevant)
            if sym in disease_symptom_map.get(top5_diseases[0], set()):
                score += 0.3

        if score > best_score:
            best_score = score
            best_sym   = sym

    return best_sym


@app.route("/restart", methods=["POST"])
def restart():
    session.clear()
    session["step"] = "welcome"
    return jsonify(ok=True)


# ── Flask Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    session.clear()
    session["step"] = "welcome"
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        msg = request.json.get("message", "").strip()

        step = session.get("step", "welcome")

        # On very first load, trigger welcome regardless of input
        if step == "welcome":
            session["step"] = "name"
            return jsonify(reply=(
                "👋 Hi there! I'm your personal Health Assistant 😊\n\n"
                "I'll chat with you about how you're feeling, ask a few simple questions, "
                "and help figure out what condition you might have — based on your symptoms.\n\n"
                "I won't guess until I'm confident, so the more honestly you answer, "
                "the better I can help you! 💙\n\n"
                "Let's start — what's your name?"
            ))

        if not msg:
            return jsonify(reply="⚠️ Please type something.")

        # ── Wikipedia intercept — any step ───────────────────────────────────
        if is_wiki_question(msg):
            query = extract_wiki_query(msg)
            if query:
                info = wiki_answer(query)
                cont = "\n\n_(Type anything to continue.)_"
                return jsonify(reply=f"📖 {info}{cont}")

        # ── Conversation ─────────────────────────────────────────────────────

        elif step == "name":
            # Strip digits/symbols — name should be letters only
            name_clean = re.sub(r"[^a-zA-Z\s]", "", msg).strip()
            if not name_clean:
                return jsonify(reply="⚠️ Please enter a valid name (letters only).")
            session["name"] = name_clean.title()
            session["step"] = "age"
            return jsonify(reply=f"Nice to meet you, {session['name']}! 👉 How old are you?")

        elif step == "age":
            nums = re.findall(r'\b(\d+)\b', msg)
            if not nums or not (1 <= int(nums[0]) <= 120):
                return jsonify(reply="⚠️ Please enter a valid age between 1 and 120.")
            session["age"] = int(nums[0])
            session["step"] = "gender"
            return jsonify(reply="👉 What is your gender? (Male / Female / Other)")

        elif step == "gender":
            g = msg.strip().lower()
            if g in ("m", "male", "man", "boy"):
                session["gender"] = "male"
            elif g in ("f", "female", "woman", "girl"):
                session["gender"] = "female"
            elif g in ("other", "o", "prefer not to say", "none"):
                session["gender"] = "other"
            else:
                return jsonify(reply=(
                    "⚠️ Please enter a valid option:\n"
                    "  • Male\n  • Female\n  • Other"
                ))
            session["step"] = "symptoms"
            return jsonify(reply=(
                f"Got it! 😊 Now, tell me how you're feeling.\n\n"
                f"Describe your symptoms in simple words — like:\n"
                f"'I have fever, headache and stomach pain'"
            ))

        elif step == "symptoms":
            symptoms = extract_symptoms(msg)
            if not symptoms:
                return jsonify(reply=(
                    "❌ I couldn't detect any recognizable symptoms.\n\n"
                    "Try describing them more clearly, e.g.:\n"
                    "'I have fever, cough and fatigue'\n\n"
                    "Or ask 'What is fever?' to learn about a symptom."
                ))

            session["symptoms"]        = symptoms
            session["asked_symptoms"]  = list(symptoms)
            session["questions_asked"] = 0
            session["current_confirm_sym"] = None   # no answer to record yet
            session["step"]            = "confirm"

            disease, conf, top3 = predict_disease(symptoms)
            sym_text = ", ".join(s.replace("_", " ") for s in symptoms)

            if conf >= CONFIDENCE_THRESHOLD:
                return _start_confirm_loop(symptoms, disease, conf)

            # Jump straight into first question — don't wait for "ok"
            next_sym = pick_best_question(symptoms, set(symptoms))
            if not next_sym:
                return _start_confirm_loop(symptoms, top3[0][0], top3[0][1])

            session["current_confirm_sym"] = next_sym
            session["asked_symptoms"].append(next_sym)
            sym_display = next_sym.replace("_", " ")
            return jsonify(reply=(
                f"🩺 Detected: {sym_text}\n\n"
                f"I'll ask a few questions to confirm your condition accurately.\n\n"
                f"👉 Do you also have {sym_display}? (yes / no)"
            ))

        elif step == "confirm":
            return _handle_confirm_answer(msg)

        elif step == "days":
            nums = re.findall(r'\b(\d+)\b', msg)
            if not nums or not (1 <= int(nums[0]) <= 365):
                return jsonify(reply="⚠️ Please enter the number of days (e.g. 2, 5, 10).")
            session["days"] = int(nums[0])
            session["step"] = "severity"
            return jsonify(reply="👉 On a scale of 1–10, how severe are your symptoms?\n(1 = very mild, 10 = unbearable)")

        elif step == "severity":
            nums = re.findall(r'\b(\d+)\b', msg)
            if not nums or not (1 <= int(nums[0]) <= 10):
                return jsonify(reply="⚠️ Please enter a number between 1 and 10.")
            session["severity"] = int(nums[0])
            session["step"]     = "sym_confirm"
            # Show collected symptoms and ask for confirmation
            symptoms = session.get("symptoms", [])
            sym_text = "\n".join(f"   ✔️ {s.replace('_', ' ').title()}" for s in symptoms)
            return jsonify(reply=(
                f"Great! Before I generate your report, let me confirm the symptoms I recorded:\n\n"
                f"{sym_text}\n\n"
                f"Is this correct? (yes / no)\n"
                f"If no, I'll let you correct them."
            ))

        elif step == "sym_confirm":
            ans = msg.strip().lower()
            if ans in ("yes", "y", "yeah", "yep", "correct", "right"):
                session["step"] = "final"
                return final_result()
            elif ans in ("no", "n", "nope", "wrong", "incorrect"):
                # Reset symptoms and go back to symptom entry
                session["symptoms"]       = []
                session["asked_symptoms"] = []
                session["questions_asked"]= 0
                session["current_confirm_sym"] = None
                session["step"] = "symptoms"
                return jsonify(reply=(
                    "No problem! Let's redo it. 😊\n\n"
                    "Please describe your symptoms again clearly:\n"
                    "Example: 'I have fever, headache and cough'"
                ))
            else:
                symptoms = session.get("symptoms", [])
                sym_text = "\n".join(f"   ✔️ {s.replace('_', ' ').title()}" for s in symptoms)
                return jsonify(reply=(
                    f"Please answer yes or no.\n\n"
                    f"Symptoms recorded:\n{sym_text}\n\n"
                    f"Is this correct? (yes / no)"
                ))

        elif step == "final":
            return final_result()

        return jsonify(reply="⚠️ Something went wrong. Please refresh and restart.")

    except Exception:
        print(traceback.format_exc())
        return jsonify(reply="⚠️ An error occurred. Please try again.")


# ── Confirm Loop Helpers ──────────────────────────────────────────────────────
def _start_confirm_loop(symptoms, disease, conf):
    """
    Called when confidence already >= threshold after initial symptoms.
    """
    session["pred_disease"] = disease
    session["pred_conf"]    = conf
    session["step"]         = "days"
    sym_text = ", ".join(s.replace("_", " ") for s in symptoms)
    return jsonify(reply=(
        f"I think I have a good idea of what's going on! 🩺\n\n"
        f"Based on: {sym_text}\n"
        f"I'm leaning towards: {disease} ({conf}% confident)\n\n"
        f"A couple more quick questions to complete your report.\n"
        f"👉 How many days have you had these symptoms?"
    ))


def _handle_confirm_answer(msg: str):
    """
    Processes yes/no answer, re-ranks by age/gender, asks next best question or concludes.
    """
    answer          = msg.lower().strip()
    current_sym     = session.get("current_confirm_sym")
    symptoms        = session.get("symptoms", [])
    asked           = set(session.get("asked_symptoms", []))
    questions_asked = session.get("questions_asked", 0)

    YES = {"yes", "y", "yeah", "yep", "yup", "sure", "correct", "true", "yea"}
    NO  = {"no",  "n", "nope", "nah", "not", "never", "false"}

    # Record answer
    if current_sym:
        if answer in YES:
            if current_sym not in symptoms:
                symptoms.append(current_sym)
            session["symptoms"] = symptoms
        elif answer in NO:
            pass   # confirmed absence — useful info, don't add
        else:
            sym_display = current_sym.replace("_", " ")
            return jsonify(reply=f"⚠️ Please reply yes or no — do you have {sym_display}?")

        asked.add(current_sym)
        session["asked_symptoms"]  = list(asked)
        session["questions_asked"] = questions_asked + 1

    # Re-predict with updated symptoms
    disease, conf, top3 = predict_disease(symptoms)

    # Age/gender re-ranking: penalise unlikely diseases for this patient
    age    = session.get("age", 30)
    gender = session.get("gender", "").lower()
    top3   = _rerank_by_demographics(top3, age, gender)
    disease, conf = top3[0]

    # ── Confident enough? ────────────────────────────────────────────────────
    if conf >= CONFIDENCE_THRESHOLD:
        session["pred_disease"] = disease
        session["pred_conf"]    = conf
        session["step"]         = "days"
        return jsonify(reply=(
            f"✅ I'm now confident about your condition.\n\n"
            f"📌 Condition identified: {disease} ({conf}% confidence)\n\n"
            f"💡 Type 'What is {disease}?' to learn more.\n\n"
            f"👉 How many days have you had these symptoms?"
        ))

    # ── Max questions reached? ───────────────────────────────────────────────
    if session.get("questions_asked", 0) >= MAX_CONFIRM_QUESTIONS:
        session["pred_disease"] = top3[0][0]
        session["pred_conf"]    = top3[0][1]
        session["step"]         = "days"
        top3_text = "\n".join(f"   {i+1}. {d} ({c}%)" for i, (d, c) in enumerate(top3))
        return jsonify(reply=(
            f"I've gathered enough information. Here are the most likely conditions:\n\n"
            f"📊 Top possibilities:\n{top3_text}\n\n"
            f"💡 Ask me 'What is {top3[0][0]}?' to learn more.\n\n"
            f"👉 How many days have you had these symptoms?"
        ))

    # ── Ask next best question ───────────────────────────────────────────────
    next_sym = pick_best_question(symptoms, asked)
    if not next_sym:
        session["pred_disease"] = top3[0][0]
        session["pred_conf"]    = top3[0][1]
        session["step"]         = "days"
        return jsonify(reply=(
            f"📌 Best match: {top3[0][0]} ({top3[0][1]}% confidence)\n\n"
            f"👉 How many days have you had these symptoms?"
        ))

    session["current_confirm_sym"] = next_sym
    asked.add(next_sym)
    session["asked_symptoms"] = list(asked)
    sym_display = next_sym.replace("_", " ")

    progress = f"_(Question {session['questions_asked']+1})_\n\n" \
               if session.get("questions_asked", 0) % 4 == 0 and session.get("questions_asked", 0) > 0 \
               else ""

    return jsonify(reply=f"{progress}👉 Do you also have {sym_display}? (yes / no)")



# ── Final Report ──────────────────────────────────────────────────────────────
def final_result():
    try:
        name     = session.get("name", "User")
        symptoms = session.get("symptoms", [])
        age      = session.get("age", "N/A")
        severity = session.get("severity", 0)
        days     = session.get("days", "N/A")

        if not symptoms:
            return jsonify(reply="⚠️ No symptoms recorded. Please restart.")

        disease, conf, top3 = predict_disease(symptoms)

        try:
            sev_int = int(severity)
        except (ValueError, TypeError):
            sev_int = 0

        urgency = (
            "🔴 High severity — please see a doctor immediately."
            if sev_int >= 7 else
            "🟡 Moderate — monitor your condition and rest well."
            if sev_int >= 4 else
            "🟢 Mild — rest, stay hydrated, and observe."
        )

        sym_text = ", ".join(s.replace("_", " ") for s in symptoms)

        if conf >= CONFIDENCE_THRESHOLD:
            diagnosis_line = f"📌 Diagnosed Condition: {disease}\n🔎 Confidence: {conf}%"
            description    = get_disease_description(disease)
            desc_section   = f"\n📖 About {disease}:\n   {description}\n" if description else ""
            precautions    = get_precautions(disease)
            prec_lines     = "\n".join(f"   ✅ {p}" for p in precautions)
            prec_section   = f"\n🛡️ Precautions:\n{prec_lines}\n"
        else:
            top3_text      = "\n".join(f"   {i+1}. {d} ({c}%)" for i, (d, c) in enumerate(top3))
            diagnosis_line = f"📊 Possible Conditions:\n{top3_text}"
            desc_section   = ""
            precautions    = get_precautions(top3[0][0])
            prec_lines     = "\n".join(f"   ✅ {p}" for p in precautions)
            prec_section   = f"\n🛡️ General Precautions (most likely condition):\n{prec_lines}\n"

        top_disease = disease if conf >= CONFIDENCE_THRESHOLD else top3[0][0]

        return jsonify(reply=(
            f"╔══════════════════════════════════╗\n"
            f" 🌸HEALTH REPORT — {name.upper()}\n"
            f"╚══════════════════════════════════╝\n\n"
            f"👤 Age: {age}  |  Severity: {sev_int}/10  |  Duration: {days} day(s)\n\n"
            f"🩺 Symptoms confirmed:\n   {sym_text}\n\n"
            f"{diagnosis_line}\n"
            f"{desc_section}"
            f"{prec_section}\n"
            f"{urgency}\n\n"
            f"💡 Ask me anything:\n"
            f"   • 'What is {top_disease}?'\n"
            f"   • 'What causes {top_disease}?'\n"
            f"   • 'How to treat {top_disease}?'\n\n"
            f"⚠️  This is NOT a medical diagnosis.\n"
            f"   Please consult a qualified doctor."
        ))

    except Exception:
        print(traceback.format_exc())
        return jsonify(reply="⚠️ Error generating the report. Please try again.")


if __name__ == "__main__":
    app.run(debug=True)