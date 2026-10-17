# 🏦 FinanceGuard - Predicción de Churn

> Proyecto Integrador M4 - Módulo de Machine Learning (Henry)
> Predicción de fuga de clientes (churn) bancario con Regresión Logística

---

## 📋 Descripción del Proyecto

**FinanceGuard** es un banco digital que ha crecido exponencialmente alcanzando **10,000 clientes activos**. Sin embargo, la dirección ha detectado un problema crítico:

- 🔴 **Tasa de abandono (churn)**: 20% anual (20.37% real según el análisis)
- 💰 **Impacto**: Pérdidas millonarias por clientes que abandonan el banco
- 🎯 **Meta**: Reducir el churn del 20% al 15%

### ¿Qué es el Churn Bancario?

El **churn** (o tasa de abandono) es el porcentaje de clientes que dejan de usar los servicios del banco en un período determinado. Es uno de los KPIs más críticos en la industria financiera porque:

1. **Costo de adquisición**: Conseguir un nuevo cliente es 5-7 veces más caro que retener uno existente
2. **Valor del tiempo de vida (LTV)**: Clientes de largo plazo generan más ingresos
3. **Reputación**: Alto churn puede indicar problemas de servicio
4. **Competencia**: En la era digital, cambiar de banco es cada vez más fácil

---

## 🎯 Objetivo

Desarrollar un **modelo predictivo de Regresión Logística** (baseline) que:

- Prediga qué clientes tienen mayor probabilidad de abandonar el banco
- Identifique los factores más influyentes en la decisión de irse
- Permita al equipo de retención actuar proactivamente
- Sirva como línea base para modelos más avanzados (Gradient Boosting, XGBoost, etc.)

---

## 📂 Estructura del Proyecto

```
M4PI/
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── ProyectoM4_LUIS_CARLOS_PACHECO_L/
    ├── Data/
    │   └── Churn_Modelling.csv          # Dataset con 10,000 clientes y 14 variables
    ├── Documentación/
    │   └── INFORME.pdf                   # Informe detallado del proyecto
    └── Notebooks/
        ├── 1_EDA_RegresionLogistica.ipynb   # Avance 1: EDA + Regresión Logística (baseline)
        ├── 2_GradientBoosting_Optimizacion.ipynb  # Avance 2: Gradient Boosting y optimización
        └── 3_AprendizajeNoSupervisado.ipynb     # Avance 3: Clustering y segmentación
```

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnología | Uso |
|-----------|------------|-----|
| Lenguaje | Python 3.10 | Desarrollo del modelo y análisis |
| Manipulación de datos | Pandas 2.2.2 | Carga, limpieza y transformación de datos |
| Cálculo numérico | NumPy 2.0.2 | Operaciones matemáticas y arrays |
| Machine Learning | Scikit-learn | Regresión Logística, train/test split, StandardScaler, LabelEncoder, métricas |
| Visualización | Matplotlib | Gráficos básicos y curvas ROC |
| Visualización | Seaborn | Mapas de calor, boxplots, gráficos estadísticos |
| Entorno | Jupyter Notebook / Google Colab | Desarrollo y documentación del análisis |
| Estadística | SciPy | Intervalos de confianza y significancia estadística |

---

## 📊 Dataset

**Archivo:** `Data/Churn_Modelling.csv`

**Dimensiones:** 10,000 clientes × 14 variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `RowNumber` | Entero | Índice de fila (identificador, no predictivo) |
| `CustomerId` | Entero | ID único del cliente (identificador, no predictivo) |
| `Surname` | Texto | Apellido del cliente (identificador, no predictivo) |
| `CreditScore` | Entero | Puntuación crediticia [300-850] |
| `Geography` | Categórico | País del cliente (France, Spain, Germany) |
| `Gender` | Categórico | Género (Male, Female) |
| `Age` | Entero | Edad del cliente [18-92] |
| `Tenure` | Entero | Antigüedad con el banco en años [0-10] |
| `Balance` | Decimal | Saldo en cuenta [0 - $250,898.09] |
| `NumOfProducts` | Entero | Número de productos contratados [1-4] |
| `HasCrCard` | Binario | Tiene tarjeta de crédito (1=Sí, 0=No) |
| `IsActiveMember` | Binario | Es miembro activo (1=Sí, 0=No) |
| `EstimatedSalary` | Decimal | Salario anual estimado [$11.58 - $199,992.48] |
| `Exited` | Binario | **Variable objetivo**: Cliente abandonó (1=Sí, 0=No) |

**Distribución de la variable objetivo (`Exited`):**
- No abandonó (0): 7,963 clientes (79.63%)
- Abandonó (1): 2,037 clientes (20.37%)
- Ratio de desbalanceo: 3.91:1

---

## 🔍 Metodología

El proyecto sigue un pipeline estructurado de Machine Learning:

### 1. Análisis Exploratorio de Datos (EDA)
- Carga y exploración inicial del dataset
- Verificación de valores nulos y duplicados (no se encontraron)
- Análisis de la variable objetivo y desbalanceo de clases
- Análisis de variables categóricas vs churn
- Análisis de variables numéricas vs churn
- Matriz de correlación para detectar multicolinealidad

### 2. Preprocesamiento
- **Limpieza**: Eliminación de columnas identificadoras (`RowNumber`, `CustomerId`, `Surname`) que no tienen poder predictivo
- **Encoding**:
  - `Gender`: Label Encoding (Female=0, Male=1)
  - `Geography`: One-Hot Encoding con `drop_first=True` (Geography_Germany, Geography_Spain)
- **Escalamiento**: `StandardScaler` (media=0, desviación estándar=1) ajustado solo sobre datos de entrenamiento

### 3. División Train/Test (80/20 con estratificación)
- 80% entrenamiento (8,000 registros), 20% prueba (2,000 registros)
- Uso de `stratify=y` para mantener la proporción de clases en ambos conjuntos
- `random_state=42` para reproducibilidad

### 4. Modelo Baseline: Regresión Logística
- Algoritmo: Regresión Logística (`sklearn.linear_model.LogisticRegression`)
- Solver: `lbfgs`
- Iteraciones máximas: 1,000
- Semilla: 42
- 11 features de entrada después del preprocesamiento

### 5. Evaluación del Modelo
- Matriz de confusión (TN, TP, FP, FN)
- Métricas: Accuracy, Precision, Recall, F1-Score
- Curva ROC y AUC
- Análisis de coeficientes y Odds Ratios
- Intervalos de confianza al 95% y prueba de significancia estadística

---

## 🤖 Modelos Implementados

### Regresión Logística (Baseline)

**Configuración:**
- `LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')`
- Entrenado sobre 8,000 registros (train) con 11 features escaladas
- Evaluado sobre 2,000 registros (test)

**Coeficientes del modelo (ordenados por importancia absoluta):**

| Variable | Coeficiente | Efecto | Odds Ratio | Cambio % |
|----------|-------------|--------|------------|----------|
| Age | +0.7388 | Aumenta churn | 2.094 | +109.4% |
| IsActiveMember | -0.5155 | Disminuye churn | 0.597 | -40.3% |
| Geography_Germany | +0.3567 | Aumenta churn | 1.429 | +42.9% |
| Gender | -0.2609 | Disminuye churn | 0.770 | -23.0% |
| Balance | +0.1606 | Aumenta churn | 1.174 | +17.4% |
| CreditScore | -0.0860 | Disminuye churn | 0.918 | -8.2% |
| NumOfProducts | -0.0703 | Disminuye churn | 0.932 | -6.8% |
| EstimatedSalary | +0.0477 | Aumenta churn | 1.049 | +4.9% |
| HasCrCard | -0.0322 | Disminuye churn | 0.968 | -3.2% |
| Tenure | -0.0201 | Disminuye churn | 0.980 | -2.0% |
| Geography_Spain | +0.0189 | Aumenta churn | 1.019 | +1.9% |

**Intercepto:** -1.6459

**Significancia estadística (IC 95%):**
- ✅ Significativas (7 de 11): CreditScore, Gender, Age, Balance, NumOfProducts, IsActiveMember, Geography_Germany
- ⚠️ No significativas (4 de 11): Tenure, HasCrCard, EstimatedSalary, Geography_Spain (su intervalo de confianza incluye cero)

> **Nota:** Las celdas de evaluación final (matriz de confusión, métricas de clasificación, curva ROC/AUC) del notebook presentan estado de ejecución pendiente. Los valores numéricos exactos de Accuracy, Precision, Recall, F1 y AUC se obtendrán al ejecutar dichas celdas.

---

## 📈 Resultados y Hallazgos

### Factores de Riesgo Identificados

Desde el análisis exploratorio y los coeficientes del modelo, los principales factores que aumentan la probabilidad de churn son:

1. **Edad avanzada**: Clientes mayores de 45 años tienen significativamente más probabilidad de abandonar (OR=2.094, cada año aumenta las odds un ~109%)
2. **País de Alemania**: Clientes en Alemania tienen el doble de tasa de churn que Francia/España (32.4% vs 16.2%) (OR=1.429)
3. **Balance alto**: Clientes con saldo elevado tienden a abandonar más (insight contraintuitivo) (OR=1.174)
4. **Género femenino**: Las mujeres abandonan un 52% más que los hombres (25.1% vs 16.5%)
5. **Inactividad**: Miembros inactivos tienen un 88% más de probabilidad de churn que los activos (14.3% vs 26.9%)
6. **3+ productos**: Contra intuitivamente, tener 3 o más productos se asocia con frustración, no lealtad (3 productos: 82.7% churn, 4 productos: 100% churn)

### Insights de Negocio

- **Perfil de mayor riesgo**: Clientes mayores (45+), con alto balance ($90K+), inactivos, y residentes en Alemania
- **Factor protector clave**: Ser miembro activo reduce drásticamente la probabilidad de churn
- **Paradoja del balance**: Los clientes con más dinero en cuenta son los que más abandonan, sugiriendo insatisfacción con rendimientos o servicios premium
- **Desbalanceo de clases**: El dataset está desbalanceado (20.4% churn), por lo que métricas como Precision, Recall y F1 son más informativas que Accuracy

---

## 💼 Impacto de Negocio

### Impacto Económico

- Cada cliente retenido evita la pérdida de su valor de vida (LTV)
- Adquirir un cliente nuevo cuesta 5-7 veces más que retener uno existente
- El modelo permite identificar clientes en riesgo antes de que abandonen, permitiendo intervención proactiva

### Recomendaciones para FinanceGuard

1. **Acción inmediata**: Implementar el modelo como baseline para scoring de riesgo de churn
2. **Segmentación prioritaria**: Enfocar campañas de retención en clientes con probabilidad de churn > 0.7
3. **Estrategias por factor de riesgo**:
   - Campañas específicas para clientes en Alemania (investigar causa raíz del 32.4% de churn)
   - Programas de reactivación para miembros inactivos
   - Revisión de productos para clientes con 3+ productos (posible frustración)
   - Atención especializada a clientes de alto balance y edad avanzada
4. **Monitoreo**: Trackear efectividad de campañas de retención y recalibrar el modelo periódicamente

---

## 🚀 Próximos Pasos

Mejoras futuras planificadas para el proyecto:

- **Balanceo de clases**: Aplicar técnicas como SMOTE, class_weight o sobremuestreo para manejar el desbalanceo (3.91:1)
- **Modelos de Gradient Boosting**: Implementar XGBoost, LightGBM y CatBoost (Avance 2)
- **Random Forest**: Comparar bagging vs boosting (Avance 2)
- **Optimización de hiperparámetros**: GridSearchCV y Optuna
- **Ensembles avanzados**: Stacking con múltiples modelos
- **Validación cruzada**: StratifiedKFold para evaluar robustez
- **Feature engineering**: Crear variables derivadas (ej. ratio balance/salario, productos por año de antigüedad)
- **Aprendizaje no supervisado**: K-Means y DBSCAN para segmentación de clientes, PCA y t-SNE para visualización (Avance 3)
- **Despliegue**: Exportar el modelo a producción para predicciones en tiempo real

---

## 👤 Autor

**LUIS CARLOS PACHECO LANZZIANO**

---

## 📄 Licencia

MIT License

Copyright (c) 2026 LUIS CARLOS PACHECO LANZZIANO

Consulte el archivo [LICENSE](LICENSE) para más detalles.

---

*Documentación generada en la rama `developer` como parte del control de versiones del proyecto FinanceGuard - Predicción de Churn.*