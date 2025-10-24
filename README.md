# BloodNet AI - Blood Bank Demand Prediction & Donor Matching

AI-powered blood demand forecasting and intelligent donor matching system for blood bank operations.

## Project Overview

BloodNet AI is a comprehensive machine learning system that:
- **Predicts blood demand** (RBC, Platelet, Plasma) across multiple time horizons
- **Matches donors** to predicted demand using intelligent scoring algorithms
- **Minimizes shortages** while maintaining high prediction accuracy (R² > 0.85)
- **Optimizes donor outreach** based on compatibility, proximity, and response history

## Key Features

### 1. Multi-Horizon Demand Prediction
- **Short-term (1-7 days)**: High-accuracy regression for daily operations
- **Mid-term (7-30 days)**: Regression with lag features for planning
- **Long-term (30-90 days)**: Classification for strategic decisions

### 2. High Performance Models
| Component | Model | R² Score | Undersupply Rate | Buffer |
|-----------|-------|----------|------------------|--------|
| RBC | LightGBM | 0.86 | 2.78% | +8% |
| Platelet | XGBoost | 0.86 | 0.00% (Perfect!) | +8% |
| Plasma | LightGBM | 0.88 | 15.28% | +5% |

### 3. Intelligent Donor Matching
- Blood type compatibility matrix
- Geographic proximity scoring (haversine distance)
- Historical response rate weighting
- Donation recency and frequency factors
- Urgency-based prioritization

### 4. Blood Group Disaggregation
Predicts demand for 8 blood groups: O+, O-, A+, A-, B+, B-, AB+, AB-

## Repository Structure

```
Bloodnet-AI/
├── src/                           # Source code
│   ├── generate_dataset.py       # Synthetic data generation
│   ├── split_dataset.py          # Dataset splitting by component
│   ├── fix_and_improve_correlations.py  # Data enhancement
│   ├── test_improved_predictions.py     # Model testing
│   ├── check_undersupply_metric.py      # Undersupply analysis
│   ├── optimize_for_undersupply.py      # Buffer optimization
│   ├── test_prophet_and_cnn.py          # Additional models
│   ├── donor_matching.py                # Donor matching algorithm
│   └── synthetic_donor_data.py          # Donor profile generation
│
├── data/                          # Datasets
│   ├── improved_rbc_demand.csv    # Enhanced RBC dataset
│   ├── improved_platelet_demand.csv
│   └── improved_plasma_demand.csv
│
├── docs/                          # Documentation
│   ├── PROJECT_DESCRIPTION.txt    # Comprehensive guide (55 KB)
│   └── IMPROVEMENTS_SUMMARY.md    # Technical details
│
├── models/                        # Trained models (gitignored)
├── notebooks/                     # Jupyter notebooks
└── tests/                         # Unit tests

```

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Bloodnet-AI
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Generate Enhanced Dataset
```bash
cd src
python generate_dataset.py           # Creates synthetic data
python split_dataset.py              # Splits by component
python fix_and_improve_correlations.py  # Enhances correlations
```

This creates `improved_rbc_demand.csv`, `improved_platelet_demand.csv`, `improved_plasma_demand.csv`

#### 2. Train and Test Models
```bash
python test_improved_predictions.py  # Test all main models
python test_prophet_and_cnn.py       # Test Prophet and CNN
```

#### 3. Optimize for Undersupply
```bash
python optimize_for_undersupply.py   # Find optimal buffers
```

#### 4. Donor Matching
```bash
python donor_matching.py             # Match donors to demand
```

Outputs: `ranked_donors.json`

## Model Performance

### Models Tested
- **LightGBM** (selected for RBC, Plasma)
- **XGBoost** (selected for Platelet)
- RandomForest
- CNN (decent but computationally expensive)
- Prophet (failed - dataset too short)
- SARIMAX (moderate performance)

### Performance Metrics

**Without Buffer** (high accuracy, high undersupply):
- RBC: R² = 0.97, Undersupply = 61%
- Platelet: R² = 0.99, Undersupply = 43%
- Plasma: R² = 0.93, Undersupply = 56%

**With Optimal Buffer** (balanced):
- RBC: R² = 0.86, Undersupply = 2.78%
- Platelet: R² = 0.86, Undersupply = 0.00% (Perfect!)
- Plasma: R² = 0.88, Undersupply = 15.28%

### Why Buffer Strategy?
In blood banking, **undersupply is 100-1000× more costly** than oversupply:
- Undersupply → Patient harm, cancelled procedures ($50k-500k per event)
- Oversupply → Some waste, but zero patient harm ($100-500 per unit)

We intentionally **overpredict by 5-8%** to prevent critical shortages.

## Undersupply Metric

**Definition**: Prediction is insufficient when `(prediction + tolerance) < actual`

**Tolerance**: 2 units (accounts for rounding/statistical noise)

**Why it matters**:
- Asymmetric costs: Shortage >> Waste
- Patient safety is paramount
- R² alone doesn't capture operational risk

## Dataset Features

### Temporal Features (7)
- date, day_of_week, week_of_year, month
- is_weekend, is_holiday

### Clinical Indicators (21)
- **RBC**: Hb counts, surgery types, dialysis, bleeding
- **Platelet**: Platelet counts, chemo patients, procedures
- **Plasma**: Coagulation abnormalities, trauma protocols

### Environmental (3)
- temperature, precipitation, air quality

### Targets (24)
- Total demand + 8 blood group disaggregations per component

## Methodology

### Data Enhancement Process
1. **Problem**: Initial synthetic data had weak correlations (<0.3)
2. **Solution**: Reconstruct targets using clinically-informed weighted formulas
3. **Result**: Correlations improved to 0.48-0.98

### RBC Demand Formula Example:
```
rbc_demand = 
  count_hb_below_7 × 25 +
  cardiac_surgeries × 40 +
  trauma_admissions × 30 +
  ... (11 weighted features)
  - is_weekend × 2000 +
  + random_noise(0, 200)
```

Weights based on clinical transfusion protocols.

## Donor Matching Algorithm

### Scoring Formula (0-100 points):
- **Blood compatibility**: 40% (exact match vs compatible)
- **Proximity**: 25% (distance to blood bank)
- **Response rate**: 20% (historical behavior)
- **Recency**: 10% (donation eligibility)
- **Frequency**: 5% (lifetime donations)

### Constraints:
- Donation interval: ≥56 days (whole blood)
- Notification cooldown: ≥14 days (prevent fatigue)
- Age: 18-65 years
- Oversampling: 3× (accounts for 30-40% response rate)

## Documentation

- **PROJECT_DESCRIPTION.txt**: Comprehensive 55 KB guide covering:
  - Step-by-step data generation
  - All 8 models tested (with rationale for selection/rejection)
  - Undersupply metric deep dive
  - Production deployment guide
  
- **IMPROVEMENTS_SUMMARY.md**: Technical correlation improvements

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Clinical transfusion protocols based on WHO guidelines
- Blood type distribution for Indian population
- Haversine distance for geographic calculations

---

**Note**: This project uses synthetic data for demonstration. In production, connect to real hospital information systems (EMR/HIS) for actual census and lab data.
