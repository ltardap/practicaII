import pandas as pd
import pyreadstat

# Carga el fichero de estudiantes
df_estudiantes, meta_estudiantes = pyreadstat.read_sav('C:/Users/laura/Downloads/pisa-2022-esp/PISA2022_Estudiantes_Esp.sav')

# Carga el fichero de centros educativos
df_centros, meta_centros = pyreadstat.read_sav('C:/Users/laura/Downloads/pisa-2022-esp/PISA2022_CentrosEducativos_Esp.sav')

# Muestra las primeras filas
print(df_estudiantes.head())
print(df_centros.head())

# Muestra las columnas disponibles
print(df_estudiantes.columns.tolist())
print(df_centros.columns.tolist())
