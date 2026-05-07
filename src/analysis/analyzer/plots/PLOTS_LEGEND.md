# Legenda wykresow i diagramow

Przewodnik interpretacji wizualizacji generowanych przez `ExperimentAnalyzer`.
Metodologia statystyczna opiera sie na Demsarze (2006, JMLR) i Arcuri & Briand
(2014, STVR).

---

## 1. Critical Difference Diagram (CD Diagram)

**Zrodlo:** Demsar J. (2006) "Statistical Comparisons of Classifiers over
Multiple Data Sets", JMLR 7, Sec. 3.2.

### Co przedstawia

CD diagram wizualizuje wynik testu Friedmana z testem post-hoc Nemenyi'ego.
Sluzy do odpowiedzi na pytanie: **ktore algorytmy sa statystycznie
nierozroznialnie od siebie, a ktore roznia sie istotnie?**

### Elementy diagramu

```
                CD = 1.23
              |-----------|
  4     3     2     1          <-- os rang (nizszy = lepszy)
  |     |     |     |
  |     |     |     +--  AlgA (1.25)    <-- najlepszy (ranga najnizsza)
  |     |     +--------  AlgB (2.00)
  |     +----------------  AlgC (3.10)
  +-----------------------  AlgD (3.65)  <-- najgorszy

  ==============================         <-- belka kliki (clique bar)
```

- **Os rang (rank axis):** Pozioma os z liczbami calkowitymi. **Ranga 1 =
  najlepszy** algorytm, ranga k = najgorszy (k = liczba algorytmow). Algorytmy
  sa umieszczone po obu stronach osi: najlepsze po prawej, najgorsze po lewej.

- **Srednia ranga (avg rank):** Liczba w nawiasie przy nazwie algorytmu.
  Obliczana testem Friedmana: w kazdym datasecie (kombinacja srodowisko x seed)
  algorytmy sa rangowane od najlepszego (1) do najgorszego (k), a nastepnie
  rangi sa usredniane.

- **CD (Critical Difference):** Odcinek z oznaczeniem "CD = X.XX" nad osia.
  Okreslony testem Nemenyi'ego przy alpha = 0.05. Jesli roznica srednich rang
  dwoch algorytmow jest **mniejsza** niz CD, to nie mozna odrzucic hipotezy
  zerowej — algorytmy sa **statystycznie nierozroznialne**.

- **Belki klik (clique bars):** Grube poziome linie pod etykietami algorytmow.
  Lacza grupy algorytmow, ktorych srednie rangi roznia sie o mniej niz CD.
  Algorytmy polaczone ta sama belka sa **statystycznie nierozroznialne** (brak
  istotnej roznicy przy p < 0.05). Przerywane pionowe linie prowadza od
  pozycji rang na osi do belek klik, ulatwiajac identyfikacje polaczonych
  algorytmow.

### Jak czytac

1. Znajdz algorytm z najnizsza srednia ranga (po prawej) — to najlepszy.
2. Sprawdz, czy belka kliki laczy go z innymi algorytmami. Jesli tak —
   roznica nie jest statystycznie istotna.
3. Algorytmy **nie** polaczone belka kliki roznia sie istotnie (roznica
   rang > CD).
4. Jesli nie ma zadnych belek klik — wszystkie pary algorytmow roznia sie
   istotnie.

### Przyklad interpretacji

> Na diagramie CD dla hypervolume algorytm NSGA-3 ma range 1.50, MSFFOA 2.00,
> OOA 3.00, SSA 3.50. CD = 1.23. Belka kliki laczy NSGA-3 i MSFFOA
> (roznica 0.50 < 1.23). OOA i SSA rowniez laczone belka (0.50 < 1.23).
> Brak belki miedzy MSFFOA a OOA (1.00 < 1.23 — tu akurat byloby tez polaczenie).
>
> Interpretacja: nie mozna stwierdzic istotnej roznicy pomiedzy NSGA-3 a MSFFOA
> w metryce hypervolume.

---

## 2. Boxplot (wykres pudelkowy)

**Zrodlo:** Demsar (2006) Sec. 3.4; Tukey J.W. (1977) "Exploratory Data
Analysis".

### Co przedstawia

Rozklad wartosci metryki (np. hypervolume, IGD+) pogrupowany po algorytmach
w danym srodowisku. Kazdy boxplot odpowiada jednemu algorytmowi.

### Elementy

```
         |          <-- whisker gorny (max bez outlierow)
    +---------+
    |         |
    |----o----|     <-- mediana (linia wewnatrz pudla)
    |         |
    +---------+     <-- Q1 (25. percentyl) i Q3 (75. percentyl)
         |          <-- whisker dolny (min bez outlierow)
         *          <-- outlier (wartosc > 1.5 * IQR od Q1 lub Q3)
```

- **Pudlo (box):** Obejmuje od 25. do 75. percentyla (IQR = interquartile
  range). Srodkowe 50% obserwacji.
- **Mediana:** Linia wewnatrz pudla. Wartosc srodkowa rozkladu.
- **Whiskery:** Rozciagaja sie do najdalszej obserwacji w granicach
  1.5 * IQR od krawedzi pudla.
- **Outliery (gwiazdki/kolka):** Obserwacje poza whiskerami.

### Jak czytac

1. Porownaj mediany — wyzsza mediana HV lub nizsza mediana IGD+ oznacza
   lepszy algorytm.
2. Wezsze pudlo = mniejsza zmiennosc wynikow (bardziej stabilny algorytm).
3. Outliery wskazuja na sporadyczne awarie lub wyjatkowo dobre/zle runy.
4. Jesli pudla dwoch algorytmow zachodzą na siebie — roznica moze nie byc
   istotna (zweryfikuj testem Wilcoxona w tabelach raportu).

---

## 3. Krzywe zbieznosci (convergence curves)

**Zrodlo:** Engelbrecht A.P. (2007) "Computational Intelligence", Sec. 16;
Beyer & Schwefel (2002) "Evolution Strategies".

### Co przedstawia

Jak wartosc metryki (np. hypervolume, best_so_far) zmienia sie w kolejnych
generacjach (iteracjach) algorytmu ewolucyjnego. Kazdy algorytm ma oddzielna
krzywa ze wstega odchylenia standardowego.

### Elementy

- **Os X:** Numer generacji (iteracji) algorytmu.
- **Os Y:** Wartosc metryki (np. hypervolume).
- **Linia ciagla:** Srednia wartosc metryki (usredniona po seedach/runach).
- **Wstega (band):** Mean +/- 1 odchylenie standardowe. Szerokosc wstegi
  odzwierciedla zmiennosc miedzy runami.

### Jak czytac

1. Algorytm, ktory szybciej osiaga plateau = szybsza zbieznosc.
2. Wyzsza koncowa wartosc HV (lub nizsza IGD+) = lepsza jakosc rozwiazania.
3. Szeroka wstega = duza zmiennosc, algorytm jest niestabilny.
4. Jesli krzywa wciaz rosnie na koncu — algorytm mogl jeszcze nie zbiec,
   warto zwiekszyc liczbe generacji.
5. Gwaltowne skoki w krzywej moga oznaczac znalezienie nowego dominujacego
   rozwiazania.

---

## 4. Wykresy slupkowe (bar charts)

**Zrodlo:** Engelbrecht (2007), standard w empirycznym porownaniu.

### Rodzaje

#### 4.1 Success Rate (wskaznik sukcesu)

- **Os X:** Algorytmy.
- **Os Y:** Odsetek runow zakonczonych sukcesem (0-1 lub 0-100%).
- Sukces = brak kolizji i dotarcie do celu w fazie online.

#### 4.2 Mean Collision Count (srednia liczba kolizji)

- **Os X:** Algorytmy.
- **Os Y:** Srednia liczba kolizji per run.
- Nizszy slupek = bezpieczniejszy algorytm.

#### 4.3 Failure Rate (wskaznik awarii)

- **Os X:** Kombinacja (srodowisko, algorytm).
- **Os Y:** Odsetek runow z awaria.
- Offline: awaria = HV = 0 lub pusty front Pareto.
- Online: awaria = collision_count > 0.

### Jak czytac

1. Porownaj wysokosci slupkow miedzy algorytmami.
2. Dla success rate — wyzszy = lepszy.
3. Dla collision count i failure rate — nizszy = lepszy.
4. Slupki pogrupowane per srodowisko pozwalaja ocenic, ktore srodowisko
   jest najtrudniejsze.

---

## 5. Ranking Heatmap (mapa cieplna rang)

**Zrodlo:** Demsar (2006) Sec. 5.

### Co przedstawia

Macierz (srodowisko x algorytm) gdzie kolor komorki odpowiada sredniej randze
algorytmu w danym srodowisku dla wybranej metryki.

### Elementy

- **Wiersze:** Srodowiska.
- **Kolumny:** Algorytmy.
- **Kolor:** Im ciemniejszy/cieplejszy, tym lepsza (nizsza) ranga.
  Skala kolorow jest podana w legendzie obok wykresu.
- **Liczba w komorce:** Konkretna srednia ranga.

### Jak czytac

1. Szukaj algorytmu z najciemniejszymi komorkami — dominuje w wiekszosci
   srodowisk.
2. Jesli algorytm jest ciemny w jednym srodowisku ale jasny w innym —
   jego skutecznosc zalezy od srodowiska.
3. Heatmap daje szybki obraz, ale nie zastepuje testow statystycznych
   (patrz: CD diagram, test Friedmana).

---

## 6. Scatter plot (wykres punktowy)

**Zrodlo:** Hansen et al. (2009) "Real-Parameter Black-Box Optimization
Benchmarking".

### Co przedstawia

Zaleznosc miedzy jakoscia rozwiazania (hypervolume) a kosztem obliczeniowym
(czas lub liczba ewaluacji). Kazdy punkt = jeden run.

### Elementy

- **Os X:** Czas trwania (elapsed_s) lub laczna liczba ewaluacji.
- **Os Y:** Hypervolume.
- **Kolor/ksztalt:** Algorytm.

### Jak czytac

1. Punkty w prawym gornym rogu = wysoki HV, ale drogi obliczeniowo.
2. Punkty w lewym gornym rogu = wysoki HV, tanio — idealny algorytm.
3. Jesli algorytm ma duze rozproszenie w osi X — jego czas jest niestabilny.
4. Trade-off: algorytm moze byc lepszy jakosciowo, ale zbyt wolny dla
   zastosowan realtime.

---

## 7. Projekcje frontu Pareto (Pareto front projections)

**Zrodlo:** Riquelme et al. (2015) Sec. 3.5; Deb et al. (2002) NSGA-II.

### Co przedstawia

Rozwiazania z ostatniej generacji (front Pareto) w przestrzeni celow
(objectives). Dla n_obj > 2, generowane sa wszystkie pary celow (i, j).

### Elementy

- **Os X, Os Y:** Wartosci dwoch wybranych celow (objectives).
- **Punkty:** Rozwiazania z frontu Pareto.
- **Kolor:** Algorytm.

### Jak czytac

1. Front blizszy poczatkowi ukladu wspolrzednych (dla minimalizacji) jest
   lepszy.
2. Szerokosc frontu = roznorodnosc rozwiazan (wiecej kompromisow).
3. Jesli punkty jednego algorytmu dominuja punkty drugiego (sa blizej
   optimum we wszystkich celach jednoczesnie) — ten algorytm jest lepszy.
4. Przerwy w froncie = brak rozwiazan w pewnych regionach przestrzeni celow.

---

## Glossary (slowniczek)

| Termin | Opis |
|:---|:---|
| **HV (Hypervolume)** | Objetosc zdominowanej przestrzeni celow. Wyzszy = lepszy front Pareto. |
| **IGD+ (Inverted Generational Distance Plus)** | Odleglosc od referencyjnego frontu. Nizszy = lepszy. |
| **GD (Generational Distance)** | Odleglosc punktow frontu od referencyjnego. Nizszy = lepszy. |
| **Friedman test** | Nieparametryczny test rang dla porownania wiecej niz dwoch algorytmow na wielu datasetach. |
| **Nemenyi post-hoc** | Test porownujacy pary algorytmow po odrzuceniu H0 w tescie Friedmana (alpha=0.05). |
| **Wilcoxon signed-rank** | Nieparametryczny test parowy dwoch algorytmow. |
| **Holm-Bonferroni** | Korekcja wielokrotnych porownani — kontroluje family-wise error rate. |
| **A12 (Vargha-Delaney)** | Miara wielkosci efektu. A12=0.5: brak roznicy, >0.71: duzy efekt. |
| **CD (Critical Difference)** | Minimalna roznica srednich rang wymagana do stwierdzenia istotnej roznicy (Nemenyi). |
| **Clique (klika)** | Grupa algorytmow nierozroznialnych statystycznie (roznica rang < CD). |
| **IQR** | Interquartile range = Q3 - Q1. Rozstep miedzykwartylowy. |
| **Front Pareto** | Zbior rozwiazan niezdominowanych — zadne inne rozwiazanie nie jest lepsze we wszystkich celach jednoczesnie. |

---

*Opracowano na podstawie: Demsar (2006, JMLR), Arcuri & Briand (2014, STVR),
Engelbrecht (2007), Deb et al. (2002), Riquelme et al. (2015).*
