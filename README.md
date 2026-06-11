# 🎓 Exam Success Predictor: A Data Science Journey

Welkom bij de **Exam Success Predictor**! In dit project neem ik je mee in mijn reis van een basis 'schoolvoorbeeld' naar een productie-waardig Machine Learning model. Het project is opgedeeld in drie opeenvolgende stappen om de leercurve, data-simulatie en de logaritmische wiskunde achter classificatie te illustreren.

Doel van het model: Voorspellen of een student slaagt (1) of zakt (0) voor een examen op basis van het aantal gestudeerde uren, gebruikmakend van **Logistieke Regressie**.

---

## 🚀 De Reis (Stapsgewijze Opbouw)

De repository is opgedeeld in drie Python-scripts die de evolutie van het model laten zien:

### Stap 1: Het Basis Model (`ML Sigmoïde kleine dataset`)
* **Doel:** Begrijpen hoe de logistische S-curve (sigmoïde) reageert op een perfecte, kleine dataset van 10 studenten.
* **Inzicht:** Het model trekt een perfecte scheidslijn, maar een kleine dataset is extreem gevoelig voor uitschieters en representeert de grillige werkelijkheid niet.

### Stap 2: Schaalvergroting & Ruis (`ML Sigmoïde grote dataset`)
* **Doel:** De dataset via `numpy` uitbreiden naar 200 fictieve studenten en natuurlijke 'ruis' (toeval) introduceren rondom het kantelpunt van 5.5 uur.
* **Inzicht:** De werkelijkheid is niet perfect zwart-wit. Door de ruis ontstaat er een realistischere 'wolk' van datapunten waarbij studenten rond het kantelpunt een subtielere kansverdeling krijgen. De gegenereerde grafiek toont de paarse S-curve die strak door de gekrompen datapunten snijdt.

### Stap 3: Professionele Evaluatie (`ML Sigmoïde grote dataset + train_test + accuracy`)
* **Doel:** Het model betrouwbaar testen en valideren zonder 'vals te spelen' (overfitting voorkomen).
* **Techniek:** De dataset vergroten naar 500 studenten. Toepassing van een **Train/Test Split (80/20)**. Het model traint op 400 studenten en wordt kritisch beoordeeld op 100 'geheime' test-studenten.

---

## 📊 Geavanceerde Evaluatiemetrieken

Om de prestaties van het model op de testset te beoordelen, kijken we verder dan alleen de standaard nauwkeurigheid:

### 1. Accuracy (Nauwkeurigheid)
Geeft het totale percentage correcte voorspellingen weer (meestal rond de **95%**).

### 2. Confusion Matrix (Verwarringsmatrix)
Breekt de voorspellingen op in vier kwadranten om te zien *waar* het model de fout in gaat:
* **True Negatives (TN) & True Positives (TP):** Correct voorspelde zakkers en slaagders.
* **False Positives (FP):** Model dacht dat de student zou slagen, maar deze zakte.
* **False Negatives (FN):** Model dacht dat de student zou zakken, maar deze slaagde.

### 3. Log-Loss (Logaritmisch Verlies): ~0.1273
De ultieme wiskundige test voor een classificatiemodel. Log-Loss straft fouten niet lineair, maar **logaritmisch** af op basis van de voorspelde kans (`predict_proba`). 

$$\text{Log-Loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i)]$$

* Als het model twijfelt (bijv. 55% kans voorspelt) en het heeft het fout, is de boete laag.
* Als het model super zeker is (bijv. 99% kans voorspelt) en het heeft het fout, schiet de logaritmische boete exponentieel omhoog.
* Een score van **0.1273** is uitstekend en bewijst dat het model niet alleen nauwkeurig is, maar ook een gezonde, wiskundige zelfverzekerdheid bezit zonder 'arrogante' fouten te maken.
