# 🚗 Car Recommendation System — pdatascience

A full-stack machine learning application that predicts whether a used car is worth buying based on its characteristics. The model was trained on a dataset of ~97,000 UK used car listings and exposed through a REST API consumed by a modern web frontend.

---

## Tech Stack

### Frontend
| Technology | Role |
|---|---|
| [Astro](https://astro.build/) | Static site framework with island architecture |
| [Tailwind CSS v4](https://tailwindcss.com/) | Utility-first styling via Vite plugin |
| TypeScript | Type-safe client-side scripting |
| Fetch API | HTTP communication with the backend |

### Backend
| Technology | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | High-performance Python REST API |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [scikit-learn](https://scikit-learn.org/) | Model loading and preprocessing |
| [joblib](https://joblib.readthedocs.io/) | Model serialization / deserialization |
| [NumPy](https://numpy.org/) | Feature vector construction |
| [Pydantic v2](https://docs.pydantic.dev/) | Request / response validation |

---

## Project Structure

```
pdatascience/
├── main.py                  # FastAPI backend
├── model.pkl                # Trained RandomForestClassifier
├── requirements.txt         # Python dependencies
├── data_cleaner.ipynb       # Data cleaning, feature engineering & model selection
├── cars_data_copy.csv       # Raw dataset (~97k rows)
└── frontend/
    ├── astro.config.mjs
    ├── package.json
    └── src/
        ├── layouts/
        │   └── Layout.astro
        ├── pages/
        │   └── index.astro  # Main UI — form + prediction result
        └── styles/
            └── global.css
```

---

## Model Integration

The prediction pipeline inside `main.py` mirrors the exact preprocessing steps applied during training in `data_cleaner.ipynb`:

### 1. Feature Engineering
- `age` is derived from the car's registration year: `age = 2026 - year`
- The `model` and `year` columns are dropped before training

### 2. Imputation
- **Mean imputation** for `price`, `mileage`, `tax`
- **Most-frequent imputation** for `fuelType`, `mpg`

### 3. Label Encoding
Categorical features are encoded with `sklearn.preprocessing.LabelEncoder` (alphabetical order):

| Feature | Encoded values |
|---|---|
| `transmission` | Automatic=0, Manual=1, Other=2, Semi-Auto=3 |
| `fuelType` | Diesel=0, Electric=1, Hybrid=2, Other=3, Petrol=4 |
| `Manufacturer` | BMW=0, audi=1, ford=2, hyundi=3, merc=4, skoda=5, toyota=6, vauxhall=7, volkswagen=8 |

### 4. Standard Scaling
`sklearn.preprocessing.StandardScaler` is applied to: `price`, `mileage`, `tax`, `mpg`.  
`engineSize` and `age` are passed as-is (not scaled during training).

### 5. Target Variable
A binary label `recommended` was generated with a scoring heuristic:

```python
def genRecommended(row):
    score = 0
    if row["age"] <= 5:          score += 1
    if row["mileage"] < 25000:   score += 1
    if row["engineSize"] <= 2.0: score += 1
    return 1 if score >= 2 else 0
```

### 6. API Endpoint
```
POST /predict
Content-Type: application/json

{
  "price": 12000,
  "transmission": "Manual",
  "mileage": 18000,
  "fuelType": "Petrol",
  "tax": 145,
  "mpg": 55.0,
  "engineSize": 1.6,
  "Manufacturer": "ford",
  "age": 5
}
```

Response:
```json
{ "recommended": true, "label": 1 }
```

---

## Model Selection

Six classifiers were benchmarked on the same train/test split (`test_size=0.2`, `random_state=42`). The metric used was `sklearn.metrics.accuracy_score`.

| # | Model | Accuracy |
|---|---|---|
| 1 | `LogisticRegression` | 93% |
| 2 | `GradientBoostingClassifier` | **100%** |
| 3 | `RandomForestClassifier` | **100%** |
| 4 | `SVC (kernel='linear')` | 93% |
| 5 | `DecisionTreeClassifier` | 99.6% |
| 6 | `MLPClassifier (100, 50)` | 99.6% |

**RandomForestClassifier** was chosen as the final model. Both `RandomForest` and `GradientBoosting` achieved 100% accuracy, but Random Forest is the more robust choice for production:

- **Ensemble method** — averages hundreds of trees, reducing variance and overfitting risk
- **Out-of-bag evaluation** — built-in generalization estimate without a separate validation set
- **Feature importance** — natively exposes which features drive predictions
- **Resistance to noise** — individual noisy trees are averaged out, unlike a single decision tree
- **Faster inference** — parallelizable at prediction time, unlike sequential boosting

`GradientBoostingClassifier` also reached 100% but is significantly slower to train and harder to tune. `MLPClassifier` and `DecisionTreeClassifier` came close at ~99.6%. Both linear models (`LogisticRegression`, `SVC`) scored lower at 93%, suggesting the decision boundary is non-linear.

The trained model is saved as `model.pkl` using `joblib.dump(model, "model.pkl", compress=3)`.

---

## Running the Project

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API (default: http://localhost:8000)
uvicorn main:app --reload
```

Interactive docs available at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend

# Install dependencies
pnpm install

# Start dev server (default: http://localhost:4321)
pnpm dev
```

---

## Dataset

The dataset contains ~97,700 used car listings from UK dealerships with the following fields:

| Column | Type | Description |
|---|---|---|
| `model` | string | Car model name |
| `year` | int | Registration year |
| `price` | float | Sale price in GBP |
| `transmission` | string | Gearbox type |
| `mileage` | int | Miles driven |
| `fuelType` | string | Fuel type |
| `tax` | int | Annual road tax in GBP |
| `mpg` | float | Miles per gallon |
| `engineSize` | float | Engine displacement in litres |
| `Manufacturer` | string | Car brand |