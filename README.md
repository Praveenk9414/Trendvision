# 📊 TrenkdVision – Google Trends Analysis & Recommendation System

TrenkdVision is a data-driven web application that analyzes **Google Trends** data to uncover **trending topics**, identify **peak activity periods**, and provide **content posting recommendations** based on category-wise user behavior insights.

🔗 **Live App**: [View on Streamlit]([https://your-streamlit-app-link.streamlit.app](https://trendvision.streamlit.app))  
🧠 **Built With:** Python, Plotly, Streamlit, Scikit-learn, Pandas

---

## 🚀 Features

- 📈 **Interactive Trend Visualizations**  
  Explore keyword popularity over time with dynamic charts powered by Plotly.

- 🧠 **NLP-based Clustering**  
  Similar trends are grouped using sentence embeddings (MiniLM) + KMeans for deeper insight.

- 🕒 **Best Time to Post Recommendations**  
  Rule-based logic suggests optimal posting days & hours per trend category.

- 🔍 **Category-wise Trend Exploration**  
  Filter and analyze top-performing trends by category, cluster, day, and hour.

- 🤖 *(Optional)*: **Trend Popularity Prediction**  
  Classify trends into High, Medium, or Low popularity using basic ML models (Logistic Regression, Random Forest).

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Visualization**: Plotly, Pandas  
- **Data Processing**: Python (Pandas, NumPy, datetime)  
- **NLP & Clustering**: Sentence Transformers (`all-MiniLM-L6-v2`), KMeans (Scikit-learn)  
- **ML (Optional)**: Scikit-learn (classification models)

---

## 📁 Dataset Overview

| Column         | Description                                                    |
|----------------|----------------------------------------------------------------|
| `Trends`       | Search term (e.g., "IPL 2025", "Oscars")                       |
| `Category`     | Category like Sports, News, Entertainment                      |
| `search_vol`   | Relative search interest (0–100 scale)                         |
| `date_time`    | Timestamp of trend recording                                   |
| `day`          | Day of the week                                                |
| `hour`         | Hour of the day (0–23)                                         |
| `cluster_seq`  | Cluster label (KMeans output based on embeddings)              |
| `popularity`   | Trend class: Low, Medium, or High                              |
| `count`        | Count of similar trends in the cluster (optional)              |


## 📦 How to Run Locally

```bash
git clone https://github.com/your-username/trenkdvision.git
cd trenkdvision
pip install -r requirements.txt
streamlit run app.py
