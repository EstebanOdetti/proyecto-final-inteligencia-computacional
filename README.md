# Detección de posibles afecciones cardiovasculares empleando técnicas de machine learning

![Generic badge](https://img.shields.io/badge/made%20with-Python-blue.svg) 

Este es el proyecto final que realize junto con otros colaboradores para la materia Inteligencia Computacional. Consta de entrenar distintos clasificadores para detectar posibles enfermedades cardiacas. En el informe se encuentran todos los detalles del proytecto. 
Extracurricularmente realizamos una interfaz grafica que utiliza el RFC para predecir nuevas ocurrencias.

## Podes descargar el programa desde [onedrive](https://1drv.ms/u/s!Ak1hSkZeE5KjhMIxkJuEYNYDA90Pzg?e=VMF2Ha)

## Resumen 
Las enfermedades cardiovasculares son la primera causa de muerte a nivel global con un estimado de 18 millones de muertes al año. Estas representan casi un de 31% de todas las muertes en todo el mundo1. Las fallas cardíacas son comúnmente causadas por enfermedades preexistentes, pudiendo ser prevenidas incorporando hábitos saludables con el fin de mitigar los factores de riesgo que predisponen a este grupo de enfermedades. En el presente trabajo se propone analizar el desempeño de distintos clasificadores binarios basados en aprendizaje automático aplicados a un conjunto de datos que contiene características relevantes para la detección de posibles enfermedades cardiovasculares. Dichas predicciones están realizadas sobre datos que pueden ser obtenidos de forma sencilla, rápida y en forma rutinaria en cualquier clínica u hospital. Los pasos básicos para la preparación del sistema son: preprocesar el dataset; entrenar los distintos modelos propuestos; realizar una búsqueda de parámetros óptimos y finalmente comparar los distintos resultados.

## Programacion 
- PyCharm 
- Python 3.9
- Octave 6.2.0 (para normalizar el dataset)

## Bibliotecas 
- Numpy 
- Pandas 
- Sklearn 
- Seaborn (para la representacion de datos)

## Dataset 
El dataset está conformado por un conjunto de datos reales reunidos por la Cleveland Clinic Foundation. Puede acceder al mismo aqui: 
[Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci) o [UCI Machine learning repository](https://archive-beta.ics.uci.edu/ml/datasets/heart+disease)


## Clasificadores utilizados
- Random forest
- Suport Vector Machine
- Perceptron Multicapa
- Voting Ensembled
- K neighbors 

## Algunos resultados

| Metodo | Precisión |
| ------ | ------ |
| Random forest  | 0.84 |
| Suport vector Machine | 0.77 |
| Voting Hard | 0.77 |
| Voting Soft | 0.80 |
| Perceptron multicapa | 0.80 |
|  K neighbors | 0.77 |

