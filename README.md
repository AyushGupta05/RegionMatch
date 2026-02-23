# 🗺️ RegionMatch — Intelligent UK Location Discovery for Businesses

RegionMatch is a data-driven decision platform that helps business owners identify the most compatible regions in the UK to expand or relocate.

Instead of relying on intuition or static reports, RegionMatch combines machine learning, public datasets, and human sentiment signals to produce ranked recommendations tailored to a company’s specific needs — including industry type, hiring urgency, and desired employee count etc.

---

## 🚀 What Problem Are We Solving?

Choosing *where* to grow a business is complex.

Founders and operators must balance:

- Local industry presence  
- Availability of skilled labour  
- Company density and competition  
- Regional workforce size  
- Cost sensitivity  
- Hiring urgency  
- Less tangible “human” signals like community sentiment  

Most tools show raw statistics. RegionMatch turns these signals into actionable recommendations.

Instead of asking:

> “Where could I expand?”

We answer:

> “Where should I expand — based on my exact needs?”

---

## 🧠 How It Works

### 1. Data Collection

We aggregate multiple UK-focused datasets mostly from nomis and IBeX that was provided, including:

- Industry prevalence by Local Authority District (LAD)
- Company size distributions
- Labour force availability
- Regional business density
- Hiring-related indicators
- Lightweight Reddit sentiment data to inject human context

Each region is represented as a structured feature vector.

---

### 2. Custom Machine Learning Model

We built our own machine learning model to compute a compatibility score between:

- A business profile (industry, size, urgency)
- Every UK region

The model evaluates how well each region matches the business owner’s requirements and outputs a numeric score per region.

Higher score = better fit.

This ML model is entirely our own.

---

### 3. Gemini Integration (Human-Readable Output)

Gemini is used only to translate model outputs into readable explanations — it does not affect scoring or ranking..

---

### Output

#### 📊 Ranked Table

A table showing:

- Region name
- Compatibility score (from ~100 downward)

---

#### 🗺️ Interactive Map

A 3D hex-map visualization where:

- Each region is shown as a vertical column
- Height + colour represent compatibility
- Yellow → lower compatibility  
- Red → highest compatibility  

This gives users an instant geographic understanding of opportunity hotspots.

---

## 🛠 Tech Stack (High Level)

- Python (data processing + ML)
- Custom ML scoring model
- Streamlit (UI)
- Geospatial visualization (3D hex map)
- Gemini API
- Public UK datasets + Reddit signals

---

Ideated and built during LondonHack 2026 to showcase how ML + geospatial data can support smarter business expansion decisions across the UK.
