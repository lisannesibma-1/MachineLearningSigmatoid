# In plaats van een handmatige kleine dataset gereneren we een grotere dataset met behulp van de numpy random bibliotheek

from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Om ervoor te zorgen dat het resultaat elke keer hetzelfde is
np.random.seed(42)

# X: Uren gestudeerd
# We genereren eerst 200 willekeurige studenten die tussen de 1 en 10 uur heeft gestudeerd
X_groot = np.random.uniform(1, 10, size=(200,1))

# We berekenen de wiskundige kans met wat ruis om de dataset realistischer te maken
wiskundige_kans = 1/(1+np.exp(-(X_groot.flatten()-5.5)))
ruis = np.random.normal(0, 0.15, 200)

# y: Examenresultaat (0 = gezakt, 1 = geslaagd)
# Als kans + ruis boven de 0.5 is, is de student geslaagd, anders gezakt
y_groot = ((wiskundige_kans + ruis) > 0.5).astype(int)

# We maken een model aan met Logistic Regression
model_slagen_groot = LogisticRegression()

# We trainen het model met onze data
model_slagen_groot.fit(X_groot, y_groot)

# We geven een nieuwe input aan het model: 4.5 uur studeren
# We berekenen de kans op falen/slagen met de method .predict_proba()
uur_gewerkt = 4.5
# De output van .predict_proba() is een array met 1 rij met 2 kolommen, met de kans op zakken in de linker en kans op slagen in de rechterkolom
kansen = model_slagen_groot.predict_proba([[uur_gewerkt]])

# De kans op slagen is (in %, dus *100) de tweede kolom van de eerste rij van kansen
kans_op_slagen = kansen[0][1] * 100

# We zien dat de kans op slagen met een grotere groep studenten is gedaald ten opzichte van onze kleine handmatige dataset
print(f"De kans om te slagen na {uur_gewerkt} uur studeren is {kans_op_slagen:.1f}%")

# We maken een plot om de nieuwe situatie te zien voor een grotere groep studenten
# Eerst een dichte reeks van 0 tot 10 uur voor een vloeiende lijn
X_dicht = np.arange(0, 10, 0.1).reshape(-1, 1)

# Alle kansen voor zowel zakken/slagen van deze reeks met model_slagen.predict_proba()
alle_kansen = model_slagen_groot.predict_proba(X_dicht)
# Voor slagingskansen pakken we alleen de tweede kolom van alle rijen
y_kansen = alle_kansen[:, 1]

# Eerst plotten we onze data in een scatter plot
y_kansen_input = model_slagen_groot.predict_proba(X_groot)
y_kansen_input_slagen = y_kansen_input[:,1]
plt.scatter(X_groot, y_kansen_input_slagen, color='red', s=10, zorder=2, label='Studenten')

# Daarna trekken we daar een vloeiende lijn doorheen op basis van X_dicht en y_kansen met plt.plot()
plt.plot(X_dicht, y_kansen, color='purple', zorder=3, label='Logistische S-curve')

# Labels voor de x-as en y-as
plt.xlabel('Uren gestudeerd', fontsize=12)
plt.ylabel('Kans op slagen', fontsize=12)

# We laten de plot zien
plt.legend()
plt.show()