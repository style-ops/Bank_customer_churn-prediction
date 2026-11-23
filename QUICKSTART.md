# 🚀 Quick Start Guide

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Ensure you have:
- `data_splits/train_set.csv` (created from your notebook)
- `data_splits/test_set.csv` (created from your notebook)

OR place your CSV file in the project root directory.

### 3. Run the Dashboard

**Windows:**
```bash
run_dashboard.bat
```

**Linux/Mac:**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Or directly with Streamlit:**
```bash
streamlit run streamlit_xai_dashboard.py
```

## Dashboard Pages

### 📊 Overview
- Dataset statistics
- Target distribution
- Model performance metrics
- Confusion matrix and ROC curve

### 🔍 SHAP Explanations
- **Summary Plot**: Global feature importance
- **Waterfall Plot**: Individual prediction explanations
- **Feature Importance**: Top features bar chart
- **Dependence Plot**: Feature interaction analysis

### 🍋 LIME Explanations
- Local explanations for individual instances
- Feature contribution breakdown
- Prediction probabilities

### 📝 NLP Analysis
- Word clouds from text columns
- Sentiment analysis using VADER
- Top words frequency
- Text statistics

### 🔮 Single Prediction
- Interactive instance selection
- Real-time predictions
- Feature inspection

## Tips

1. **First Run**: The dashboard will train a model if one doesn't exist (takes a few minutes)
2. **SHAP/LIME**: Use smaller instance counts for faster explanations
3. **Performance**: Models are cached after first run for faster loading
4. **NLP**: Text columns are automatically detected for analysis

## Troubleshooting

- **No data found**: Ensure CSV files are in `data_splits/` or root directory
- **Model training slow**: Normal on first run; subsequent runs use cached model
- **SHAP errors**: Try reducing the number of instances or features

## Next Steps

1. Explore the Overview page to understand your data
2. Use SHAP to understand global model behavior
3. Use LIME to explain individual predictions
4. Analyze text data with NLP features
5. Test single predictions interactively

---

For detailed documentation, see [README.md](README.md)

