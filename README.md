# 🏦 XAI Dashboard for Bank Churn Prediction

A comprehensive Streamlit dashboard that provides Explainable AI (XAI) insights for bank churn prediction models. This dashboard complements the Jupyter notebook by offering interactive visualizations and explanations using SHAP, LIME, and NLP analysis.

## ✨ Features

### 1. **📊 Overview Dashboard**
- Dataset statistics and metrics
- Target distribution visualization
- **Model comparison** - side-by-side performance metrics for all three models
- Model performance metrics for selected model
- Confusion matrix and ROC curve
- Classification report

### 2. **🔍 SHAP Explanations**
- **Summary Plot**: Global feature importance across all instances
- **Waterfall Plot**: Detailed explanation for individual predictions
- **Feature Importance**: Bar plot showing most important features
- **Dependence Plot**: Feature interaction analysis

### 3. **🍋 LIME Explanations**
- Local interpretable model-agnostic explanations
- Instance-level feature importance
- Prediction probability breakdown
- Top contributing features visualization

### 4. **📝 NLP Analysis**
- **Word Cloud**: Visual representation of most frequent words
- **Sentiment Analysis**: Sentiment scores using VADER
- **Top Words**: Frequency analysis of words in text columns
- **Text Statistics**: Length and word count distributions

### 5. **🔮 Single Prediction**
- Interactive instance selection
- Real-time prediction with confidence scores
- Feature inspection for selected instances

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download NLTK Data

The dashboard will automatically download required NLTK data on first run. If you encounter issues, manually download:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## 📁 Project Structure

```
.
├── streamlit_xai_dashboard.py    # Main dashboard application
├── requirements.txt               # Python dependencies
├── README.md                     # This file
├── EDI_Bank_Churn_Prediction XGBoost.ipynb  # Original notebook
├── data_splits/                  # Train/test data (if available)
│   ├── train_set.csv
│   └── test_set.csv
└── models/                       # Saved models (auto-created)
    ├── lr_model.pkl              # Logistic Regression model
    ├── rf_model.pkl              # Random Forest model
    ├── xgb_model.pkl             # XGBoost model
    └── preprocessor.pkl          # Shared preprocessor
```

## 🎯 Usage

### Running the Dashboard

1. **Ensure your data files are in place:**
   - The dashboard will look for `data_splits/train_set.csv` and `data_splits/test_set.csv`
   - If these don't exist, it will try to load any CSV file in the directory

2. **Launch the Streamlit app:**

```bash
streamlit run streamlit_xai_dashboard.py
```

3. **Access the dashboard:**
   - The app will open in your default web browser
   - Default URL: `http://localhost:8501`

### Using the Dashboard

1. **Navigation**: Use the sidebar to switch between different pages
2. **Model Selection**: Choose from Logistic Regression, Random Forest, or XGBoost in the sidebar
3. **Model Loading**: The dashboard will automatically train all models if they don't exist
4. **Interactive Features**: Use sliders, dropdowns, and buttons to explore different aspects
5. **Model Comparison**: View side-by-side performance metrics in the Overview page

## 🔧 How It Works

### Model Training
- Supports three models: **Logistic Regression**, **Random Forest**, and **XGBoost**
- If no saved models exist, the dashboard trains all three models
- Uses the same preprocessing pipeline as the notebook
- Handles class imbalance with appropriate techniques for each model
- Models are cached for faster subsequent runs
- Model comparison view shows performance metrics side-by-side

### Data Processing
- Automatically identifies numeric and categorical features
- Applies StandardScaler to numeric features
- Applies OneHotEncoder to categorical features
- Handles missing values and infinite values

### SHAP Integration
- Uses TreeExplainer for tree-based models (XGBoost, Random Forest) - fast and accurate
- Uses KernelExplainer for Logistic Regression - model-agnostic approach
- Computes SHAP values for feature importance
- Provides multiple visualization types (Summary, Waterfall, Feature Importance, Dependence plots)

### LIME Integration
- Creates local explanations for individual instances
- Shows feature contributions to predictions
- Provides interpretable feature importance

### NLP Features
- Analyzes text columns: `transaction_description`, `merchant_category`, `feedback_type`, `resolution_status`
- Performs sentiment analysis using VADER
- Generates word clouds and frequency analysis

## 📊 Supported Data Format

The dashboard expects:
- **Target column**: `aml_flag` (or binary classification column)
- **Numeric features**: Standard numeric columns
- **Categorical features**: String/object columns
- **Text columns**: For NLP analysis (optional)

## 🛠️ Customization

### Adding New Features

1. **New Visualization**: Add functions in the main file following the pattern of existing functions
2. **Custom Metrics**: Modify the `show_overview()` function
3. **Additional Models**: Extend the `load_or_train_model()` function

### Model Parameters

Edit the model training parameters in the `train_model()` function:

```python
model = XGBClassifier(
    n_estimators=300,      # Number of trees
    learning_rate=0.1,     # Learning rate
    max_depth=6,           # Tree depth
    # ... other parameters
)
```

## ⚠️ Notes

- **Performance**: SHAP computations can be slow for large datasets. The dashboard limits instances for performance.
- **Memory**: Large models may require significant RAM. Consider reducing the number of features if needed.
- **Data Privacy**: Ensure sensitive data is handled according to your organization's policies.

## 🔍 Troubleshooting

### Issue: "No CSV files found"
- **Solution**: Ensure your data files are in the correct directory or upload via file uploader (if implemented)

### Issue: Model training is slow
- **Solution**: This is normal for first run. Subsequent runs will use cached models.

### Issue: SHAP/LIME explanations take too long
- **Solution**: Reduce the number of instances or features being analyzed

### Issue: NLTK data download errors
- **Solution**: Manually download required NLTK data as shown in installation section

## 📈 Future Enhancements

Potential improvements:
- [ ] File upload functionality for new datasets
- [ ] Model comparison (multiple models side-by-side)
- [ ] Export explanations as PDF/HTML
- [ ] Real-time prediction API
- [ ] Advanced NLP features (NER, topic modeling)
- [ ] Integration with other XAI libraries (Eli5, Captum)

## 📝 License

This project is part of the EDI Bank Churn Prediction workflow.

## 🤝 Contributing

Feel free to extend this dashboard with additional features or improvements!

## 📞 Support

For issues or questions, please refer to the original notebook or create an issue in the repository.

---

**Built with ❤️ using Streamlit, SHAP, LIME, and NLP libraries**

