# Supply Chain Analytics Dashboard

A modern, full-stack application for supply chain analytics, featuring demand forecasting, inventory optimization, and route planning.

## Tech Stack
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL, Prophet (Forecasting), Scikit-learn
- **Frontend**: React, Material-UI, Recharts, Axios

## Prerequisites
- Python 3.8+
- Node.js 14+
- PostgreSQL

## Setup & Running

### 1. Backend

Navigate to the root directory:
```bash
cd d:\supply-chain-analytics
```

Create and activate a virtual environment (if not already done):
```bash
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Setup Environment Variables:
Ensure you have a `.env` file in the root directory with the following:
```env
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/supply_chain_db
KAGGLE_DATASET_PATH=D:\supply-chain-analytics\DataCoSupplyChainDataset.csv
```

Run the server:
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`.
API Documentation: `http://localhost:8000/docs`

### 2. Frontend

Navigate to the frontend directory:
```bash
cd supply-chain-dashboard
```

Install dependencies:
```bash
npm install
```

Start the development server:
```bash
npm start
```
The application will be available at `http://localhost:3000`.

## Features
- **Dashboard**: Real-time overview of key metrics.
- **Demand Forecast**: ML-powered demand prediction using Prophet.
- **Inventory Optimization**: Reorder point and safety stock calculations.
- **Anomaly Detection**: Identify irregularities in sales or orders.
- **Route Optimization**: Cluster customers for efficient delivery routing.
