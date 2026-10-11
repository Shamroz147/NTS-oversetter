import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

mendeley_df = pd.read_csv("mendeley.csv")
egen_df = pd.read_csv("data.csv")

mendeley_df["bokstav"] = mendeley_df["bokstav"].replace({"AE": "Æ", "OE": "Ø"})




def normaliser(rad, speilvend=False, forhold=1.0):
    punkter = np.array(rad, dtype=float).reshape(21, 3) #Gjør om 63 tall til tabell med 21 rader (punkter) og 3 kolonner (x, y, z)
    punkter = punkter - punkter[0]

    punkter[:, 0] = punkter[:, 0] * forhold
    punkter[:, 2] = punkter[:, 2] * forhold

    if speilvend:
        punkter[:, 0] = -punkter[:, 0]


    skala = np.linalg.norm(punkter[9])
    punkter = punkter / skala

    return punkter.flatten()



X_mendeley = np.array([normaliser(rad) for rad in mendeley_df.drop(columns="bokstav").values])
y_mendeley = mendeley_df["bokstav"].values


X_egen = np.array([normaliser(rad, forhold=640 / 480) for rad in egen_df.drop(columns="bokstav").values])
y_egen = egen_df["bokstav"].values


X_train, X_test, y_train, y_test = train_test_split(
    X_mendeley, y_mendeley, test_size=0.2, stratify=y_mendeley, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

gjetninger = model.predict(X_test)


print("Treffsikkerhet på Mendeley-test:", accuracy_score(y_test, gjetninger))

maske = np.isin(y_egen, model.classes_)
X_egen_test = X_egen[maske]
y_egen_test = y_egen[maske]

gjetninger_egen = model.predict(X_egen_test)
print("Treffsikkerhet på egen data:", accuracy_score(y_egen_test, gjetninger_egen))

print()
for bokstav in sorted(set(y_egen_test)):
    m = y_egen_test == bokstav
    riktig = (gjetninger_egen[m] == bokstav).mean()
    vanligst = pd.Series(gjetninger_egen[m]).value_counts().index[0]
    print(f"{bokstav}: {riktig:.0%} riktig, gjettet oftest {vanligst}")



X_egen_speil = np.array([normaliser(rad, speilvend=True, forhold=640 / 480) for rad in egen_df.drop(columns="bokstav").values])
X_egen_speil_test = X_egen_speil[maske]

gjetninger_speil = model.predict(X_egen_speil_test)
print()
print("Treffsikkerhet på egne data, speilvendt:", accuracy_score(y_egen_test, gjetninger_speil))


print()
print("--- Kombinert Modell ---")


train_idx = []
test_idx = []
for bokstav in sorted(set(y_egen)):
    indekser = np.where(y_egen == bokstav)[0]
    halv = len(indekser) // 2
    train_idx.extend(indekser[:halv])
    test_idx.extend(indekser[halv:])


X_kombinert = np.vstack([X_train, X_egen[train_idx]])
y_kombinert = np.concatenate([y_train, y_egen[train_idx]])


model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_kombinert, y_kombinert)

X_egen_halv = X_egen[test_idx]
y_egen_halv = y_egen[test_idx]
maske_halv = np.isin(y_egen_halv, model.classes_)


print("Gammel modell, egne testdata:",
      accuracy_score(y_egen_halv[maske_halv], model.predict(X_egen_halv[maske_halv])))
print("Ny modell, egne testdata:", 
      accuracy_score(y_egen_halv, model2.predict(X_egen_halv)))
print("Ny modell, Mendeley-data:",
      accuracy_score(y_test, model2.predict(X_test)))




print()
print("--- Eksperiment A: vekt egne data høyere ---")

vekter = np.concatenate([np.ones(len(y_train)), np.full(len(train_idx), 15.0)])

model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model3.fit(X_kombinert, y_kombinert, sample_weight=vekter)

print("Vektet modell, egne testdata:", accuracy_score(y_egen_halv, model3.predict(X_egen_halv)))
print("Vektet modell, Mendeley-test:", accuracy_score(y_test, model3.predict(X_test)))


print()
print("--- Eksperiment B: bare egne data ---")


model4 = RandomForestClassifier(n_estimators=100, random_state=42)
model4.fit(X_egen[train_idx], y_egen[train_idx])

print("Bare egne data, egne testdata:", accuracy_score(y_egen_halv, model4.predict(X_egen_halv)))

print()
print("--- Endelig modell ---")

X_alt = np.vstack([X_mendeley, X_egen])
y_alt = np.concatenate([y_mendeley, y_egen])
vekter_alt = np.concatenate([np.ones(len(y_mendeley)), np.full(len(y_egen), 15.0)])

model_finale = RandomForestClassifier(n_estimators=100, random_state=42)
model_finale.fit(X_alt, y_alt, sample_weight=vekter_alt)

joblib.dump(model_finale, "modell.joblib")
print("Modellen er lagret i modell.joblib")
