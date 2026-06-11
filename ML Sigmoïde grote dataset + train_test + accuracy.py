# In plaats van al onze data aan het model te geven, splitsen we onze data op in train/test subsets
# Hierdoor kunnen we de nauwkeurigheid van ons model bepalen, met accuracy_score en log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt

# Om ervoor te zorgen dat het resultaat elke keer hetzelfde is
np.random.seed(42)

# X: Uren gestudeerd
# We genereren eerst 200 willekeurige studenten die tussen de 1 en 10 uur heeft gestudeerd
X_groot = np.random.uniform(1, 10, size=(500,1))

# We berekenen de wiskundige kans met wat ruis om de dataset realistischer te maken
wiskundige_kans = 1/(1+np.exp(-(X_groot.flatten()-5.5)))
ruis = np.random.normal(0, 0.15, 500)

# y: Examenresultaat (0 = gezakt, 1 = geslaagd)
# Als kans + ruis boven de 0.5 is, is de student geslaagd, anders gezakt
y_groot = ((wiskundige_kans + ruis) > 0.5).astype(int)

# We voegen een train/test split toe om overfitting te voorkomen, we kiezen voor test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_groot, y_groot, test_size=0.2, random_state=42)

# We maken een model aan met Logistic Regression
model_slagen_groot = LogisticRegression()

# We trainen het model met onze train data
model_slagen_groot.fit(X_train, y_train)

# We testen de nauwkeurigheid van ons getrainde model door te vergelijken met y_test
y_pred = model_slagen_groot.predict(X_test)
nauwkeurigheid = accuracy_score(y_test, y_pred)

# We kijken welke false positives/false negatives er zijn
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print(f"Model Nauwkeurigheid op de testset: {nauwkeurigheid:.2f}")

# We berekenen de logarithmic loss om te bepalen hoe fout de verkeerde voorspellingen van het model zijn
y_pred_proba = model_slagen_groot.predict_proba(X_test)
score_logloss = log_loss(y_test, y_pred_proba)
print(f"Log-Loss score van het model: {score_logloss:.4f}")