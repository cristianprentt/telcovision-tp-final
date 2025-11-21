# TelcoVision - TP Final

**Predicción de churn en Telecomunicaciones con MLOps**

Proyecto final de la materia **Laboratorio de Minería de Datos**  
Docente: **Diego Mosquera**  
Alumno: **Cristian Prentt**

**Descripción del proyecto**

La empresa ficticia **TelcoVision** busca reducir la rotación de clientes (**churn**).  
Este proyecto implementa un pipeline **MLOps** completo que permite:

* Versionado de datos y modelos con **DVC**  
* Registro y comparación de experimentos con **MLflow** (DagsHub)  
* Ejecución reproducible del pipeline de ML  
* Evaluación y selección del mejor modelo  
* CI/CD automatizado con **GitHub Actions**

---

## Notebook principal

El flujo completo del proyecto (clonado del repo, instalación de dependencias, conexión a DagsHub, `dvc pull`, ejecución de experimentos y evaluación final) se puede reproducir automáticamente ejecutando el siguiente notebook:

**[TP_FINAL.ipynb — Ejecutar en Google Colab](https://colab.research.google.com/drive/1KZ0-FgKhhT5uMCN86jTb-i2-4zGbiyLp#scrollTo=g_VPoTRr1AoF)**

Este notebook realiza:

1. Clonado de este repositorio desde GitHub  
2. Instalación de dependencias (`requirements.txt`)  
3. Configuración de MLflow con tracking remoto en DagsHub  
4. `dvc pull`: Descarga dataset y outputs versionados  
5. Ejecución de los experimentos (`src/experiments.py`)  
6. Comparación de métricas en MLflow  
7. Evaluación final del modelo con curva ROC y matriz de confusión  
8. Registro automático de métricas y artefactos

Ideal para cualquier persona que quiera **reproducir el proyecto en un entorno nuevo** 

---

## Estructura del repositorio

telcovision-tp-final/
├── data/
│ ├── raw/ ← Dataset original (bajo control DVC)
│ └── processed/ ← Limpieza y features (DVC)
├── src/
│ ├── data_prep.py ← ETL / Feature Engineering
│ ├── experiments.py ← Ejecución múltiples modelos
│ └── evaluate.py ← Model champion + métricas finales
├── models/ ← Artefactos de entrenamiento (DVC)
├── dvc.yaml ← Pipeline definido en DVC
├── params.yaml ← Hiperparámetros
├── requirements.txt ← Dependencias del proyecto
└── .github/workflows/ ← CI/CD con GitHub Actions

---

## Mejor modelo encontrado

| Modelo             | Accuracy | Precision | Recall | F1-score |
|--------------------|:--------:|:---------:|:------:|:--------:|
| `exp_balance_lowC` |  0.6484  |   0.50    | 0.6897 | **0.5808** |

Modelo almacenado y trazable en DagsHub.

---

## Reproducción del Pipeline (local)

```bash
`git clone https://github.com/cristianprentt/telcovision-tp-final.git
cd telcovision-tp-final
pip install -r requirements.txt
dvc pull
dvc repro

```
--

Links útiles
Recurso	URL
* GitHub Repo - https://github.com/cristianprentt/telcovision-tp-final

* Tracking MLflow + DVC en DagsHub -https://dagshub.com/cristianprentt/telcovision-tp-final.mlflow

* Colab ejecutable del proyecto - https://colab.research.google.com/drive/1KZ0-FgKhhT5uMCN86jTb-i2-4zGbiyLp

Agradecimientos

 * Nadia Casá
 * Karla Silva Vargas
