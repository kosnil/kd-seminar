# KD-Seminar

## Finance Daten:
- finance_data_pipeline.py lässt den Download und Aggregation ausführen
- Daten werden in Ordner finance_data/data/ gespeichert

## ER Daten:
- Feature Liste werden in feature_generation.py für einen definierten Zeitraum (startDate,endDate) erzeugt
- Daten je Zeitraum werden in Ordner er_data/data gespeichert
- features_aggregation.py ausführen um alle Zeiträume zusammenzuführen und in einem DF zu speichern
- #TODO: Gleiches Vorgehen bei IBM Daten

# data_aggregation.py:
- Holt sich die Daten von ER und Finance und führt beide zusammen in df
- Speichert ab in final_data/
- #TODO: IBM Daten dann ebenfalls hier einfügen.