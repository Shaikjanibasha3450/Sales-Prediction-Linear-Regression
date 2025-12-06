# 📊 Sales Prediction Using Linear Regression

## Overview

This is a comprehensive **Machine Learning** project that predicts product sales based on advertising spending across different channels (TV, Radio, and Newspaper). The project uses **Linear Regression** to build a predictive model that analyzes the relationship between advertising investments and actual sales.

## 🎯 Project Objectives

- **Analyze** the relationship between advertising spending and sales
- **Build** a predictive model using Linear Regression
- **Evaluate** model performance using multiple metrics
- **Visualize** data patterns and predictions
- **Provide** actionable insights for budget allocation

## 📁 Dataset

**Dataset Name:** Advertising Dataset

**Source:** [Kaggle - Advertising.csv](https://www.kaggle.com/datasets/bumba5341/advertisingcsv)

**Dataset Features:**
- **TV:** Advertising budget spent on TV (in thousands of dollars)
- **Radio:** Advertising budget spent on Radio (in thousands of dollars)
- **Newspaper:** Advertising budget spent on Newspaper (in thousands of dollars)
- **Sales:** Resulting sales (in thousands of dollars)

**Dataset Size:** 200 samples × 4 columns

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine Learning algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Google Colab** - Jupyter Notebook environment

## 📊 Model Performance

| Metric | Training Set | Testing Set |
|--------|-------------|-------------|
| **R² Score** | 0.9761 | 0.9824 |
| **RMSE** | 2.0992 | 1.6618 |
| **MAE** | 1.6542 | 1.4097 |
| **MSE** | 4.4067 | 2.7617 |

**Model Interpretation:**
- The model explains **98.24%** of the variance in sales on the testing set
- **Average prediction error** is only **$1.41k**
- Strong correlation between advertising spend and sales

## 🔍 Key Features of the Project

1. **Exploratory Data Analysis (EDA)**
   - Statistical summary of the dataset
   - Distribution analysis
   - Correlation heatmap
   - Scatter plots showing relationships

2. **Data Preprocessing**
   - Data splitting (80-20 train-test split)
   - Feature selection
   - Data normalization considerations

3. **Model Training**
   - Linear Regression implementation
   - Model coefficient extraction
   - Interpretation of feature importance

4. **Model Evaluation**
   - Multiple evaluation metrics (R², RMSE, MAE, MSE)
   - Actual vs Predicted visualization
   - Residual analysis

5. **Predictions**
   - Sample predictions on new data
   - Real-world scenario testing
   - Budget allocation recommendations

## 📈 Coefficient Insights

**Model Equation:**
```
Sales = 0.0490 × TV + 1.0901 × Radio + 0.0551 × Newspaper - 0.1755
```

**Interpretation:**
- **TV**: For every $1 increase in TV advertising → **$0.049k increase** in sales
- **Radio**: For every $1 increase in Radio advertising → **$1.090k increase** in sales
- **Newspaper**: For every $1 increase in Newspaper advertising → **$0.055k increase** in sales

## 💡 Business Recommendations

1. **Prioritize Radio Advertising** - Highest ROI with coefficient 1.0901
2. **Maintain TV Investment** - Strong impact with coefficient 0.0490
3. **Minimize Newspaper Spend** - Lowest impact with coefficient 0.0551
4. **Budget Allocation Strategy:**
   - 50% → Radio advertising
   - 40% → TV advertising
   - 10% → Newspaper advertising

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Run all cells sequentially
3. View visualizations and predictions

### Option 2: Local Machine
```bash
# Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the Python script
python sales_prediction.py
```

## 📊 Visualizations Included

- ✅ Feature vs Sales scatter plots (TV, Radio, Newspaper)
- ✅ Distribution of Sales histogram
- ✅ Correlation heatmap
- ✅ Actual vs Predicted scatter plots
- ✅ Residuals analysis plots
- ✅ Training vs Testing performance comparison

## 📝 Project Structure

```
Sales-Prediction-Linear-Regression/
├── README.md
├── sales_prediction.py          # Main Python script
├── sales_prediction.ipynb       # Jupyter Notebook
├── advertising_dataset.csv      # Dataset (optional)
└── requirements.txt             # Dependencies
```

## 🎓 Learning Outcomes

- Understanding of Linear Regression fundamentals
- Data preprocessing and exploration techniques
- Model training and evaluation
- Performance metrics interpretation
- Data visualization best practices
- Real-world machine learning workflow

## 📚 References

- [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Advertising Dataset - Kaggle](https://www.kaggle.com/datasets/bumba5341/advertisingcsv)
- [Linear Regression Concepts](https://en.wikipedia.org/wiki/Linear_regression)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Share your insights

## 📄 License

This project is open-source and available for educational purposes.

## 👨‍💻 Author

**Shaik Janibasha**
- GitHub: [@Shaikjanibasha3450](https://github.com/Shaikjanibasha3450)
- Project Link: [Sales-Prediction-Linear-Regression](https://github.com/Shaikjanibasha3450/Sales-Prediction-Linear-Regression)

---

**⭐ If you found this project helpful, please consider giving it a star!**

**Made with ❤️ for Machine Learning Enthusiasts**
