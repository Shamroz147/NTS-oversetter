# NTS-oversetter 
## Sanntidsgjenkjenning av norsk håndalfabet med MediaPipe og Random Forest.

Et fritidsprosjekt der jeg prøver å gjenkjenne det norske håndalfabetet med webkamera og maskinlæring.

Programmet finner hånden din med MediaPipe, henter ut 21 punkter på den (fingertupper, knoker og håndledd), og en Random Forest-modell gjetter hvilken bokstav du viser.

## Hvordan det virker

1. MediaPipe finner 21 punkter på hånden
2. Punktene normaliseres, så det ikke spiller noen rolle hvor i bildet hånden er eller hvor nær kameraet du står
3. En Random Forest gjetter bokstaven og viser hvor sikker den er

Jeg bruker punktene i stedet for selve bildet, fordi da bryr ikke modellen seg om bakgrunn eller lys, og den trenger mye mindre data.

## Filer

- `samle_data.py` – samler inn egne treningsdata med webkameraet
- `mendeley_landemerker.py` – henter ut punkter fra bildene i Mendeley-datasettet
- `tren.py` – trener og tester modellen
- `demo.py` – sanntidsdemo med webkameraet

## Kom i gang

```
pip install -r requirements.txt
```

Du trenger også:
- [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) fra MediaPipe, i samme mappe som skriptene
- [Mendeley-datasettet](https://data.mendeley.com/datasets/3cfrj4vd4m/1), pakket ut i `mendeley/raadata_BS_H_N/`

Kjør så skriptene i denne rekkefølgen:

```
python mendeley_landemerker.py
python samle_data.py
python tren.py
python demo.py
```

I `samle_data.py` starter du opptak med mellomrom, bytter bokstav med N og P, og avslutter med Q.

## Resultater

Modellen fikk 99,8 % på testdelen av Mendeley-datasettet, men bare rundt 20–35 % på min egen hånd. Grunnen er at Mendeley bare har to personer, så modellen lærte seg hendene deres og ikke bokstavene generelt.

Det jeg prøvde:

| Modell | Min hånd |
|---|---|
| Bare Mendeley | 20 % |
| Mendeley + mine data | 43 % |
| Mendeley + mine data (vektet høyere) | 52 % |
| Bare mine data | 75 % |

Jeg testet også om speilvending av webkameraet eller bildeformatet (4:3 mot kvadratisk) var problemet, men det hjalp nesten ingenting. Det som hjalp mest var mer og bedre data.

## Begrensninger

- Blander bokstaver som ligner, særlig A/E/S og M/N
- Bare testet på meg selv
- Bare bokstaver uten bevegelse, og H og Å er ikke med i Mendeley-datasettet
- Gjenkjenner bare håndalfabetet, ikke ekte tegnspråk med bevegelser og ansiktsuttrykk

## Kanskje senere

- Samle data fra flere personer
- Bygge ord ved å holde en bokstav stille en stund
- Støtte tegn med bevegelse

## Kilder

- Kadry, S. & Svendsen, B. (2023). Norwegian Sign Language. Mendeley Data. https://doi.org/10.17632/3cfrj4vd4m.1
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
