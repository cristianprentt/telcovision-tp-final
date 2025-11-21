import pandas as pd
import yaml
import os

def main():
    # Leer parámetros
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["prepare"]

    input_path = params["input"]
    output_path = params["output"]

    print(f"Cargando dataset desde: {input_path}")
    df = pd.read_csv(input_path)

    # Limpieza básica mínima (puedes ampliarla luego)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Reemplazar errores típicos en total_charges
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # Quitar filas con valores faltantes
    df = df.dropna()

    # Crear carpeta output si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Guardando dataset limpio en: {output_path}")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
