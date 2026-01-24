# LiDAR Arheološki Preglednik v0.99

## Korisnička dokumentacija

---

> ⚠️ **NAPOMENA O PRIJEVODU**
> 
> Ovaj dokument je strojno preveden s engleskog jezika. Iako je prijevod pregledan, moguće su manje netočnosti ili nespretan izričaj. Za tehničke nejasnoće, molimo konzultirajte izvornu englesku verziju dokumentacije.

---

## Sadržaj

1. [Pregled](#pregled)
2. [Sistemski zahtjevi](#sistemski-zahtjevi)
3. [Instalacija](#instalacija)
4. [Brzi početak](#brzi-početak)
5. [Kako radi](#kako-radi)
6. [Korištenje preglednika](#korištenje-preglednika)
7. [Koordinatni sustavi](#koordinatni-sustavi)
8. [Referenca naredbenog retka](#referenca-naredbenog-retka)
9. [Rješavanje problema](#rješavanje-problema)

---

## Pregled

LiDAR Arheološki Preglednik je alat temeljen na Pythonu koji pretvara datoteke digitalnog modela reljefa (DMR) u interaktivni web preglednik optimiziran za arheološku prospekciju. Obrađuje GeoTIFF datoteke koje sadrže LiDAR podatke o terenu i generira:

- Visokokvalitetne vizualizacije sjenčanog reljefa koristeći višesmjerno osvjetljenje
- Besprijekorno sučelje za pomicanje i zumiranje u stilu Google Mapsa
- Pretvorbu koordinata između projiciranog (HTRS96/TM) i geografskog (WGS84) sustava
- Sloj satelitskih snimaka za usporedbu terena
- Funkcionalnost pretraživanja adresa i koordinata

Preglednik je dizajniran za arheologe, istraživače i entuzijaste koji žele identificirati potencijalne arheološke značajke (zemljane radove, grobne humke, antičke ceste, sustave polja itd.) u LiDAR podacima o terenu.

---

## Sistemski zahtjevi

### Softverske ovisnosti

- **Python 3.8+**
- **Potrebni Python paketi:**
  - `numpy` - numerička obrada
  - `Pillow` (PIL) - obrada slika
  - `rasterio` - čitanje GeoTIFF datoteka
  - `tqdm` - trake napretka (opcionalno, ali preporučeno)

### Hardverske preporuke

- **RAM:** minimum 8GB, preporučeno 16GB+ za velike skupove podataka
- **Pohrana:** preporučen SSD; veličina izlaza je približno 10-30% ulaznih podataka
- **Zaslon:** preporučena minimalna rezolucija 1920×1080

### Ulazni podaci

- GeoTIFF datoteke (.tif) koje sadrže podatke o nadmorskoj visini
- Podržani koordinatni sustavi:
  - EPSG:3765 (HTRS96/TM - Hrvatska)
  - EPSG:3794 (D96/TM - Slovenija)
  - EPSG:32633 (UTM Zona 33N)
  - EPSG:32634 (UTM Zona 34N)
  - Lokalni koordinatni sustavi (automatski detektirani na temelju vrijednosti koordinata)

---

## Instalacija

### Korak 1: Instalirajte Python ovisnosti

```bash
pip install numpy pillow rasterio tqdm
```

### Korak 2: Preuzmite skriptu

Spremite `convert_dmr_to_images.py` u svoj radni direktorij.

### Korak 3: Provjerite instalaciju

```bash
python convert_dmr_to_images.py --help
```

---

## Brzi početak

### Osnovna uporaba

```bash
python convert_dmr_to_images.py -i /putanja/do/tif/datoteka -o /putanja/do/izlaza
```

### Način samo-reference (preporučeno za ponovna pokretanja)

Ako ste već obradili slike i samo želite regenerirati preglednik:

```bash
python convert_dmr_to_images.py -i /putanja/do/tif/datoteka -o /putanja/do/izlaza --reference-only
```

### Pregledajte rezultate

Otvorite `viewer.html` u izlaznom direktoriju s bilo kojim modernim web preglednikom.

---

## Kako radi

### Cjevovod obrade

Alat izvršava sljedeće korake:

#### 1. Otkrivanje pločica i ekstrakcija metapodataka

Skripta skenira ulazni direktorij za GeoTIFF datoteke i ekstrahira:
- Geografske granice (lijevo, desno, gore, dolje)
- Koordinatni referentni sustav (CRS)
- Dimenzije piksela
- Središnje koordinate

#### 2. Generiranje sjenčanog reljefa

Za svaku pločicu izračunava se višesmjerni sjenčani reljef:

```
Konačni sjenčani reljef = ponderirani prosjek sjenčanih reljefa iz više kutova sunca
```

Zadani kutovi sunca: 315°, 270°, 225°, 360° (SZ, Z, JZ, S)

Algoritam sjenčanog reljefa koristi Hornovu metodu za izračun nagiba i aspekta iz mreže nadmorskih visina, zatim izračunava osvjetljenje na temelju položaja sunca.

#### 3. Izvoz slika

Obrađene pločice spremaju se kao JPEG (zadano, 85% kvalitete) ili PNG slike, čuvajući originalno imenovanje pločica.

#### 4. Generiranje pregledne karte

Stvara se pregledna karta niske rezolucije koja prikazuje sve pločice, korištena za navigaciju u pregledniku.

#### 5. Generiranje HTML preglednika

Generira se interaktivna HTML datoteka koja sadrži:
- Sve metapodatke pločica (granice, pozicije)
- Funkcije transformacije koordinata
- Sučelje za pomicanje/zumiranje
- Funkcionalnost pretraživanja
- Sustav satelitskog sloja

### Struktura izlaza

```
izlazni_direktorij/
├── viewer.html          # Glavni interaktivni preglednik
├── overview_map.png     # Navigacijska pregledna slika
├── tiles.csv            # Proračunska tablica metapodataka pločica
├── plocica_001.jpg      # Obrađene slike pločica
├── plocica_002.jpg
├── ...
└── plocica_NNN.jpg
```

---

## Korištenje preglednika

### Raspored sučelja

```
┌─────────────────────────────────────────────────────────────┐
│  Bočna traka                │  Glavno područje karte        │
│  ──────────                 │                               │
│  🔍 Arheološki pregled      │   [Zum: 25% 50% 100% 200%]   │
│                             │   [Prikaz koordinata]         │
│  PRETRAŽI PLOČICE           │                               │
│  [____________]             │                               │
│                             │      ┌─────────────────┐      │
│  PRETRAŽI ADRESU            │      │                 │      │
│  [____________]             │      │  LiDAR pločice  │      │
│                             │      │                 │      │
│  IDI NA KOORDINATE          │      │ (pomicanje i    │      │
│  [X/Lon] [Y/Lat]            │      │     zumiranje)  │      │
│  [Idi] [Zalijepi]           │      └─────────────────┘      │
│                             │                               │
│  SLOJ KARTE                 ├───────────────────────────────┤
│  Satelit [────────] 0%      │ Povuci za pomicanje │ Kotačić │
│  ☐ Prikaži nazive mjesta    │ C kopiraj WGS84 │ M HTRS96   │
│                             │ L ime pločice │ [?]          │
│  PRIKAZ                     └───────────────────────────────┘
│  Svjetlina [────────]       
│  Kontrast  [────────]       
│  [Invertiraj][Reset][Oznake]
│                             
│  PREGLED                    
│  ┌─────────────────┐        
│  │ [pregledna      │        
│  │     karta]      │        
│  └─────────────────┘        
│                             
│  INFO O PLOČICI             
│  [info pri prelasku mišem]  
└─────────────────────────────┘
```

### Navigacija

| Radnja | Metoda |
|--------|--------|
| Pomicanje | Kliknite i povucite na karti |
| Zumiranje | Kotačić miša (zumira prema kursoru) |
| Zumiranje (gumbi) | Kliknite 25%, 50%, 100%, 200% ili Prikaži sve |
| Skok na lokaciju | Kliknite na preglednu kartu |

### Tipkovnički prečaci

| Tipka | Radnja |
|-------|--------|
| **C** | Kopiraj WGS84 koordinate (geografska širina, dužina) |
| **M** | Kopiraj HTRS96/TM koordinate (Easting, Northing) |
| **L** | Kopiraj naziv trenutne pločice |
| **I** | Uključi/isključi inverziju boja |
| **+** / **=** | Povećaj zum |
| **-** | Smanji zum |
| **0** | Prikaži sve pločice u pogledu |
| **Esc** | Zatvori info modal |

### Značajke pretraživanja

#### Pretraživanje pločica
- Upišite dio naziva pločice za filtriranje
- Koristite `*` kao zamjenski znak (npr. `DMR*103*`)
- Kliknite rezultat ili koristite strelice + Enter za navigaciju
- Padajući izbornik prikazuje naziv pločice i WGS84 koordinate

#### Pretraživanje adresa
- Upišite bilo koji naziv mjesta, ulice ili adresu
- Pretražuje Hrvatsku i Sloveniju putem Photon/Nominatim API-ja
- Djelomični nazivi rade (npr. "Zagr" pronalazi "Zagreb")
- Kliknite rezultat za navigaciju do te lokacije
- Radi čak i za lokacije izvan vaše LiDAR pokrivenosti

#### Pretraživanje koordinata
- Unesite koordinate u bilo kojem formatu:
  - **WGS84:** `46.137, 15.778` (decimalni stupnjevi)
  - **HTRS96:** `444265, 5111092` (metri)
- Sustav automatski detektira format na temelju veličine vrijednosti
- Kliknite "Idi" ili pritisnite Enter za navigaciju
- Gumb "Zalijepi" parsira koordinate iz međuspremnika

### Satelitski sloj

Klizač satelita kontrolira vidljivost zračnih snimaka ispod vaših LiDAR podataka:

| Položaj klizača | Rezultat |
|-----------------|----------|
| 0% | Samo LiDAR (puna neprozirnost) |
| 50% | LiDAR poluproziran preko satelita |
| 100% | Samo satelit (LiDAR nevidljiv) |

Ovo vam omogućuje:
- Verificirati LiDAR značajke u odnosu na moderni teren
- Identificirati zgrade, ceste i vegetaciju
- Korelirati arheološke značajke s trenutnom uporabom zemljišta

**Potvrdni okvir Nazivi mjesta:** Prekriva nazive naselja, ceste i geografske oznake iz OpenStreetMapa preko vašeg prikaza.

### Prilagodbe prikaza

| Kontrola | Učinak |
|----------|--------|
| **Svjetlina** | Posvijetli (>100%) ili potamni (<100%) sliku |
| **Kontrast** | Povećaj (>100%) ili smanji (<100%) kontrast |
| **Invertiraj** | Zamijeni crno/bijelo; korisno za uočavanje suptilnih značajki |
| **Reset** | Vrati na zadane postavke prikaza |
| **Oznake** | Uključi/isključi oznake granica pločica |

### Panel informacija o pločici

Pri prelasku mišem preko pločice, bočna traka prikazuje:
- **Naziv pločice** (naziv datoteke bez ekstenzije)
- **Veličina** u pikselima
- **Raspon Eastinga** (projicirane X koordinate)
- **Raspon Northinga** (projicirane Y koordinate)
- **Središte** u WGS84 (klikni za kopiranje)

---

## Koordinatni sustavi

### Razumijevanje dvaju sustava

Preglednik istovremeno prikazuje koordinate u dva formata:

#### HTRS96/TM (EPSG:3765) - Projicirane koordinate

**Primjer:** `E: 444.264,9  N: 5.111.092,4`

Ovo je službeni koordinatni sustav Hrvatske, poprečna Mercatorova projekcija.

| Parametar | Vrijednost |
|-----------|------------|
| Elipsoid | GRS80 |
| Središnji meridijan | 16,5°E |
| Lažni easting | 500.000 m |
| Lažni northing | 0 m |
| Faktor mjerila | 0,9999 |

**Čitanje koordinata:**
- **Easting (E):** Metri istočno/zapadno od središnjeg meridijana (16,5°E), plus 500.000m pomak
  - E < 500.000 → zapadno od 16,5°E
  - E > 500.000 → istočno od 16,5°E
- **Northing (N):** Metri sjeverno od ekvatora

**Prednosti:** 
- Udaljenosti u metrima su intuitivne
- Lako izračunavanje pravocrtnih udaljenosti
- Nema negativnih brojeva

#### WGS84 (EPSG:4326) - Geografske koordinate

**Primjer:** `46,137375, 15,778623`

Ovo je globalni standard koji koriste GPS, Google Maps i većina web kartiranja.

| Komponenta | Značenje |
|------------|----------|
| Geografska širina (prvi broj) | Stupnjevi sjeverno od ekvatora |
| Geografska dužina (drugi broj) | Stupnjevi istočno od nultog meridijana |

**Prednosti:**
- Univerzalno - radi bilo gdje na Zemlji
- Direktno upotrebljivo u Google Mapsu, GPS uređajima
- Standard za dijeljenje lokacija online

### Primjer pretvorbe

```
HTRS96: E: 444.264,9, N: 5.111.092,4
        ↓
Easting: 444.265 - 500.000 = -55.735m (zapadno od 16,5°E)
Na 46°N: 1° geografske dužine ≈ 77,8 km
Geografska dužina: 16,5° - (55,735 / 77,8) ≈ 15,78°E

Northing: 5.111.092m od ekvatora
Koristeći geometriju elipsoida → 46,137°N
        ↓
WGS84: 46,137°N, 15,778°E
```

### Info gumb

Kliknite gumb **?** u traci pomoći za prikaz detaljnog objašnjenja oba koordinatna sustava unutar preglednika.

---

## Referenca naredbenog retka

### Osnovna sintaksa

```bash
python convert_dmr_to_images.py -i ULAZ -o IZLAZ [opcije]
```

### Obavezni argumenti

| Argument | Opis |
|----------|------|
| `-i`, `--input` | Ulazni direktorij koji sadrži GeoTIFF datoteke |
| `-o`, `--output` | Izlazni direktorij za obrađene datoteke |

### Opcionalni argumenti

| Argument | Zadano | Opis |
|----------|--------|------|
| `--format` | `jpg` | Izlazni format: `jpg` ili `png` |
| `--quality` | `85` | JPEG kvaliteta (1-100) |
| `--reference-only` | isključeno | Preskoči obradu slika; regeneriraj samo preglednik |
| `--sun-elevation` | `45` | Kut elevacije sunca u stupnjevima |
| `--sun-azimuths` | `315,270,225,360` | Kutovi azimuta sunca odvojeni zarezom |
| `--workers` | broj CPU-a | Broj paralelnih radnih procesa |

### Primjeri

**Visokokvalitetni PNG izlaz:**
```bash
python convert_dmr_to_images.py -i ./dem_plocice -o ./izlaz --format png
```

**Prilagođeni kutovi sunca za poboljšane detalje sjena:**
```bash
python convert_dmr_to_images.py -i ./dem_plocice -o ./izlaz --sun-azimuths 315,45,135,225
```

**Regeneriraj preglednik nakon ažuriranja koda:**
```bash
python convert_dmr_to_images.py -i ./dem_plocice -o ./izlaz --reference-only
```

---

## Rješavanje problema

### Česti problemi

#### "Nema pronađenih valjanih pločica"

**Uzrok:** Ulazni direktorij ne sadrži čitljive GeoTIFF datoteke.

**Rješenja:**
- Provjerite imaju li datoteke ekstenziju `.tif`
- Provjerite jesu li datoteke valjani GeoTIFF-ovi s `gdalinfo naziv_datoteke.tif`
- Osigurajte dozvole za čitanje datoteka

#### Koordinate izgledaju pomaknuto od Google Mapsa

**Uzrok:** Netočna detekcija koordinatnog sustava.

**Rješenja:**
- Provjerite konzolni izlaz tijekom generiranja za "Detected CRS" i "Using projection"
- Ako automatska detekcija ne uspije, provjerite stvarni CRS vaših podataka
- Hrvatski podaci trebaju koristiti HTRS96 (središnji meridijan 16,5°E)
- Slovenski podaci trebaju koristiti D96TM (središnji meridijan 15,0°E)

#### Satelitski sloj nije poravnat s LiDAR-om

**Uzrok:** Nepodudaranje transformacije koordinata.

**Rješenja:**
- Provjerite odgovaraju li parametri projekcije vašim podacima
- Provjerite odgovaraju li WGS84 koordinate u pregledniku očekivanim lokacijama
- Usporedite poznatu oznaku u LiDAR i satelitskom prikazu

#### Preglednik prikazuje praznu stranicu ili greške

**Uzrok:** Sigurnosna ograničenja preglednika za lokalne datoteke.

**Rješenja:**
- Koristite moderan preglednik (Chrome, Firefox, Edge)
- Posluživanje datoteka kroz lokalni web poslužitelj:
  ```bash
  cd izlazni_direktorij
  python -m http.server 8000
  ```
  Zatim otvorite `http://localhost:8000/viewer.html`

#### Pretraživanje adresa ne vraća rezultate

**Uzrok:** Mrežna povezanost ili ograničenja API-ja.

**Rješenja:**
- Provjerite internetsku vezu
- Pokušajte s konkretnijim pojmovima za pretraživanje
- Provjerite konzolu preglednika za CORS greške
- Photon API može imati ograničenja brzine; pričekajte i pokušajte ponovno

### Savjeti za performanse

1. **Koristite JPEG format** za brže učitavanje (manje datoteke od PNG-a)
2. **Obradite podskupove** za testiranje prije pokretanja cijelih skupova podataka
3. **Koristite `--reference-only`** kada su potrebne samo promjene preglednika
4. **SSD pohrana** dramatično poboljšava brzinu učitavanja pločica
5. **Smanjite zum preglednika** (Ctrl+0) ako se pomicanje čini sporim s mnogo pločica

---

## Zasluge i izvori podataka

### Satelitske snimke
- ESRI World Imagery (ArcGIS)

### Oznake karte
- CartoDB/CARTO sloj oznaka
- OpenStreetMap suradnici

### Pretraživanje adresa
- Photon by Komoot (primarni)
- OpenStreetMap Nominatim (rezervni)

### LiDAR podaci
Ovaj alat je dizajniran za korištenje s LiDAR podacima nacionalnih kartografskih agencija kao što su:
- DGU - Državna geodetska uprava (Hrvatska)
- GURS - Geodetska uprava Republike Slovenije (Slovenija)

---

## Povijest verzija

### v0.99 (Trenutna)
- Sučelje karte s kontinuiranim pomicanjem (stil Google Mapsa)
- Dvostruki prikaz koordinata (HTRS96 + WGS84)
- Sloj satelitskih snimaka s kontrolom prozirnosti
- Sloj oznaka naziva mjesta
- Pretraživanje adresa s automatskim dovršavanjem
- Pretraživanje pločica s padajućim izbornikom
- Pretraživanje koordinata (oba formata)
- Tipkovnički prečaci za kopiranje koordinata
- Modalni prozor s informacijama o koordinatnim sustavima
- Automatska detekcija hrvatskih/slovenskih projekcija

---

## Licenca

Ovaj alat je namijenjen za arheološka istraživanja i obrazovne svrhe.

LiDAR podaci mogu biti podložni uvjetima licenciranja od strane izvorne nacionalne kartografske agencije. Korisnici su odgovorni za usklađenost s primjenjivim uvjetima korištenja podataka.

---

> 📝 **Izvorna verzija:** Ova dokumentacija je izvorno napisana na engleskom jeziku. 
> Verzija prijevoda: 0.99 | Datum prijevoda: siječanj 2025.
