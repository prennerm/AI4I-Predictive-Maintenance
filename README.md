# AI4I Predictive Maintenance 🔧

> Vorhersage von Maschinenausfällen mit künstlicher Intelligenz

## Was macht dieses Projekt?

Dieses Tool analysiert Sensordaten von Industriemaschinen und sagt vorher, wann sie wahrscheinlich ausfallen werden. So können Sie:

- **Ausfälle vermeiden** bevor sie passieren
- **Wartungskosten senken** durch bessere Planung  
- **Produktion optimieren** mit weniger ungeplanten Stopps
- **Geld sparen** durch präventive statt reaktive Wartung

## Quick Start 🚀

### 1. Installation

```bash
# Repository herunterladen
git clone https://github.com/prennerm/AI4I-Predictive-Maintenance.git
cd AI4I-Predictive-Maintenance

# Python-Pakete installieren
pip install -r requirements.txt
```

### 2. Erstes Modell trainieren

```bash
# Vollständiges Training mit Standardeinstellungen
python scripts/train_model.py

# Schneller Test (nur für Ausprobieren)
python scripts/train_model.py --quick-run
```

### 3. Vorhersagen machen

```bash
# Einzelne Vorhersage (Beispielwerte)
python scripts/predict.py --model models/best_model/ --input "300,310,1500,40,120"

# CSV-Datei mit mehreren Maschinen
python scripts/predict.py --model models/best_model/ --input your_data.csv --batch
```

### 4. Report erstellen

```bash
# Business-Report für Management
python scripts/generate_report.py --type executive --models models/

# Technischer Report
python scripts/generate_report.py --type technical --models models/
```

## Wie verwende ich eigene Daten? 📊

### Datenformat

Ihre CSV-Datei sollte diese Spalten enthalten:

```csv
Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure
300.1,309.2,1551,42.8,0,0
298.2,308.6,1408,46.3,3,0
...
```

- **Machine failure**: 0 = Normal, 1 = Ausfall (nur für Training nötig)
- **Andere Spalten**: Sensormesswerte Ihrer Maschinen

### Eigene Daten verwenden

```bash
# 1. Ihre Daten ins data/raw/ Verzeichnis kopieren
cp ihre_daten.csv data/raw/

# 2. Modell mit Ihren Daten trainieren
python scripts/train_model.py --data data/raw/ihre_daten.csv

# 3. Vorhersagen für neue Daten
python scripts/predict.py --model models/best_model/ --input neue_daten.csv
```

## Verstehe die Ergebnisse 🎯

### Vorhersage-Output

```json
{
  "prediction": 1,              // 0 = OK, 1 = Ausfall erwartet
  "probability": 0.85,          // 85% Ausfallwahrscheinlichkeit  
  "risk_level": "HIGH",         // LOW/MEDIUM/HIGH/CRITICAL
  "recommended_action": "Schedule maintenance within 48 hours",
  "confidence": 0.92            // Wie sicher ist das Modell
}
```

### Risk Levels erklärt

- 🟢 **LOW (0-30%)**: Weiterlaufen lassen
- 🟡 **MEDIUM (30-60%)**: Wartung innerhalb 1 Woche planen
- 🟠 **HIGH (60-80%)**: Wartung innerhalb 48h einplanen
- 🔴 **CRITICAL (80%+)**: SOFORT warten - Maschine stoppen!

## Häufige Anwendungsfälle 💼

### Tägliche Überwachung

```bash
# Alle Maschinen täglich prüfen
python scripts/predict.py --model models/best_model/ --input daily_readings.csv --business-report
```

### Wöchentliche Management-Reports

```bash
# Automatischer Wochenreport
python scripts/generate_report.py --type executive --models models/ --format html
```

### API für andere Systeme

```bash
# REST API starten (Port 8080)
python scripts/predict.py --model models/best_model/ --api-mode --port 8080

# Dann HTTP POST an: http://localhost:8080/predict
# mit JSON: {"input": [300, 310, 1500, 40, 120]}
```

## Testen & Debugging 🔍

### Funktioniert alles?

```bash
# Schneller Funktionstest
python scripts/train_model.py --quick-run --no-hyperopt

# Sollte ohne Fehler durchlaufen und Modelle erstellen
```

### Häufige Probleme & Lösungen

#### Problem: "ModuleNotFoundError"
```bash
# Lösung: Pakete installieren
pip install -r requirements.txt

# Oder spezifisches Paket
pip install pandas scikit-learn xgboost
```

#### Problem: "FileNotFoundError"
```bash
# Daten existieren?
ls data/raw/

# Modell existiert?
ls models/

# Lösung: Erst trainieren, dann vorhersagen
python scripts/train_model.py
```

#### Problem: Schlechte Vorhersagen
```bash
# Mehr Daten verwenden
python scripts/train_model.py --data data/raw/mehr_daten.csv

# Andere Modelle probieren
python scripts/train_model.py --models random_forest xgboost svm

# Performance prüfen
python scripts/evaluate_model.py --model-dir models/ --compare-all
```

### Debug-Modus aktivieren

```bash
# Ausführliche Logs für Problemdiagnose
python scripts/train_model.py --log-level DEBUG

# Logs finden Sie in: logs/
```

### Performance prüfen

```bash
# Detaillierte Modell-Evaluation
python scripts/evaluate_model.py --model-dir models/ --business-analysis

# Ergebnisse in: reports/evaluation/
```

## Erweiterte Nutzung ⚙️

### Konfigurationsdatei verwenden

Erstellen Sie `config.yaml`:

```yaml
data:
  test_size: 0.2
  preprocessing:
    scale_features: true

training:
  hyperparameter_optimization:
    enabled: true
    n_trials: 50

models:
  traditional:
    random_forest:
      n_estimators: [100, 200]
```

Dann verwenden:
```bash
python scripts/train_model.py --config config.yaml
```

### Batch-Verarbeitung

```bash
# Viele Dateien auf einmal
for file in data/raw/*.csv; do
    python scripts/predict.py --model models/best_model/ --input "$file" --output "predictions/$(basename $file .csv)_predictions.json"
done
```

### Automatisierung (Cron)

```bash
# Täglich um 6 Uhr neue Vorhersagen
0 6 * * * cd /pfad/zum/projekt && python scripts/predict.py --model models/best_model/ --input data/daily.csv --output reports/daily_predictions.json

# Wöchentlich Modell neu trainieren  
0 2 * * 1 cd /pfad/zum/projekt && python scripts/train_model.py --data data/raw/weekly_data.csv
```

## Tipps für bessere Ergebnisse 💡

### Datenqualität verbessern

1. **Mehr Daten sammeln** - Je mehr, desto besser
2. **Ausgewogene Daten** - Sowohl normale als auch Ausfall-Beispiele
3. **Aktuelle Daten** - Regelmäßig neue Daten hinzufügen
4. **Vollständige Daten** - Wenige fehlende Werte

### Modell optimieren

```bash
# Verschiedene Feature-Sets probieren
python scripts/train_model.py --feature-set basic      # Weniger Features
python scripts/train_model.py --feature-set extended   # Standard
python scripts/train_model.py --feature-set comprehensive  # Alle Features

# Hyperparameter-Optimierung
python scripts/train_model.py --models xgboost --config hyperopt_config.yaml
```

### Business-Integration

1. **Schrittweise einführen** - Erst parallel zu bestehenden Prozessen
2. **Teams schulen** - Wartungsteams über neue Prozesse informieren
3. **KPIs messen** - Erfolg quantifizieren (Ausfälle, Kosten, Uptime)
4. **Feedback sammeln** - Kontinuierliche Verbesserung

## Projektstruktur verstehen 📁

```
AI4I/
├── scripts/           # Die 4 Hauptprogramme
│   ├── train_model.py       # Modelle trainieren
│   ├── evaluate_model.py    # Performance bewerten  
│   ├── predict.py           # Vorhersagen machen
│   └── generate_report.py   # Reports erstellen
├── src/               # Interner Code (nicht direkt verwenden)
├── data/              # Ihre Daten
│   ├── raw/                 # Original CSV-Dateien
│   └── processed/           # Verarbeitete Daten
├── models/            # Trainierte Modelle
├── reports/           # Generierte Reports
└── logs/              # Debug-Informationen
```

**Faustregel**: Sie arbeiten hauptsächlich mit den 4 Scripts in `scripts/`!

## Support & Hilfe 🆘

### Dokumentation

- **Technische Details**: Siehe `architecture.md`
- **API-Referenz**: Docstrings in den Python-Dateien
- **Beispiele**: Siehe Kommentare in den Scripts

### Fehlerbehebung

1. **Logs prüfen**: `logs/` Verzeichnis
2. **Debug-Modus**: `--log-level DEBUG` verwenden
3. **Step-by-step**: Ein Script nach dem anderen testen
4. **Clean start**: `models/` und `reports/` löschen, neu anfangen

### Typische Workflows

#### Erstmaliger Nutzer
```bash
1. pip install -r requirements.txt
2. python scripts/train_model.py --quick-run
3. python scripts/predict.py --model models/best_model/ --input "300,310,1500,40,120"
4. python scripts/generate_report.py --type executive --models models/
```

#### Produktive Nutzung
```bash
1. python scripts/train_model.py --data data/raw/production_data.csv
2. python scripts/evaluate_model.py --model-dir models/ --business-analysis
3. python scripts/predict.py --model models/best_model/ --input daily_sensors.csv --business-report
4. python scripts/generate_report.py --type operational --predictions predictions.json
```

## Was als nächstes? 🎯

1. **Experimentieren**: Probieren Sie verschiedene Einstellungen aus
2. **Integrieren**: Verbinden Sie mit Ihren bestehenden Systemen
3. **Skalieren**: Erweitern Sie auf mehr Maschinen/Standorte
4. **Optimieren**: Nutzen Sie die Reports zur kontinuierlichen Verbesserung

---

🚀 **Viel Erfolg bei der Vorhersage von Maschinenausfällen!**

*Bei Fragen oder Problemen: Schauen Sie in die Logs (`logs/`) oder aktivieren Sie den Debug-Modus mit `--log-level DEBUG`*
