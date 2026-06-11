# Doel: Het bepalen van de slagingskans van een student op basis van het aantal gestudeerde uren
# Input: Aantal gestudeerde uren, Output: 0=gezakt, 1=geslaagd

from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# X: Uren gestudeerd
X_uren = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# y: Examenresultaat (0 = gezakt, 1 = geslaagd)
y_slaagt = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

# We maken een model aan met Logistic Regression
model_slagen = LogisticRegression()

# We trainen het model met onze data
model_slagen.fit(X_uren, y_slaagt)

# We geven een nieuwe input aan het model: 4.5 uur studeren
uur_gewerkt = 4.5
# We berekenen de kans op falen/slagen met de method .predict_proba()
# De output van .predict_proba() is een array met 1 rij met 2 kolommen, met de kans op zakken in de linker en kans op slagen in de rechterkolom
kansen = model_slagen.predict_proba([[uur_gewerkt]])
print(kansen)

# De kans op slagen is (in %, dus *100) de tweede kolom van de eerste rij van kansen
kans_op_slagen = kansen[0][1] * 100

print(f"De kans om te slagen na {uur_gewerkt} uur studeren is {kans_op_slagen:.1f}%")

# De volgende stap is een plot maken om voor alle mogelijke uren een beeld te krijgen van de slagingskans
# Eerst een dichte reeks van 0 tot 10 uur voor een vloeiende lijn
X_dicht = np.arange(0, 10, 0.1).reshape(-1, 1)

# Alle kansen voor zowel zakken/slagen van deze reeks met model_slagen.predict_proba()
alle_kansen = model_slagen.predict_proba(X_dicht)
# Voor slagingskansen pakken we alleen de tweede kolom van alle rijen
y_kansen = alle_kansen[:, 1]

# Eerst plotten we onze data
y_kansen_input = model_slagen.predict_proba(X_uren)
y_kansen_input_slagen = y_kansen_input[:,1]
plt.scatter(X_uren, y_kansen_input_slagen, color='red')

# Daarna trekken we daar een vloeiende lijn doorheen op basis van X_dicht en y_kansen met plt.plot()
plt.plot(X_dicht, y_kansen, color='purple')

# Labels voor de x-as en y-as
plt.xlabel('Uren gestudeerd', fontsize=12)
plt.ylabel('Kans op slagen', fontsize=12)

# We laten de plot zien
plt.show()