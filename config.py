default_features = ["Differenz_Abisolierposition", "Differenz_Abisolierlaenge_max", "Abisolierungs-Einzeldefektflaeche_max",
                    "Abisolierungs-Gesamtdefektflaeche"]

features_with_threshold = {"Differenz_Abisolierposition": ['Differenz_Abisolierposition', 'Min_Differenz_Abisolierposition', 'Max_Differenz_Abisolierposition'],
                           "Differenz_Abisolierlaenge_max": ['Differenz_Abisolierlaenge_max', 'Min_Differenz_Abisolierlaenge_max',
                                                             'Max_Differenz_Abisolierlaenge_max'],
                           "Differenz_Abisoliermitte_max": ['Differenz_Abisoliermitte_max', 'Min_Differenz_Abisoliermitte_max',
                                                            'Max_Differenz_Abisoliermitte_max'],
                           "Isolations-Einzeldefektflaeche_max": ['Isolations-Einzeldefektflaeche_max', 'Min_Isolations-Einzeldefektflaeche_max', 'Max_Isolations-Einzeldefektflaeche_max'],
                           "Isolations-Gesamtdefektflaeche": ['Isolations-Gesamtdefektflaeche', 'Min_Isolations-Gesamtdefektflaeche', 'Max_Isolations-Gesamtdefektflaeche'],
                           "Abisolierungs-Einzeldefektflaeche_max": ['Abisolierungs-Einzeldefektflaeche_max', 'Min_Abisolierungs-Einzeldefektflaeche_max',
                                                                     'Max_Abisolierungs-Einzeldefektflaeche_max'],
                           "Abisolierungs-Gesamtdefektflaeche": ['Abisolierungs-Gesamtdefektflaeche', 'Min_Abisolierungs-Gesamtdefektflaeche', 'Max_Abisolierungs-Gesamtdefektflaeche']}

thresholds = {
    "Differenz_Abisolierposition_norm": {"min": 0, "max": 1, "target": 0.5},
    "Differenz_Abisolierlaenge_max_norm": {"min": 0, "max": 1, "target": 0.5},
    "Differenz_Abisoliermitte_max_norm": {"min": 0, "max": 1, "target": 0.5},
    "Isolations-Einzeldefektflaeche_max_norm": {"min": 0, "max": 1, "target": 0},
    "Isolations-Gesamtdefektflaeche_norm": {"min": 0, "max": 1, "target": 0},
    "Abisolierungs-Einzeldefektflaeche_max_norm": {"min": 0, "max": 1, "target": 0},
    "Abisolierungs-Gesamtdefektflaeche_norm": {"min": 0, "max": 1, "target": 0},
}
