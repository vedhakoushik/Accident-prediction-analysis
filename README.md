# Accident Risk Prediction Project

This project combines a FastAPI backend with a React + Vite frontend to predict accident risk for a city location or route using the included road accident dataset.

## Project structure

- `backend/`: FastAPI API for metadata, single-point prediction, and route prediction
- `frontend/`: React client built with Vite and React Leaflet
- `predict.py`: training and prediction pipeline used by the backend
- `main.py`: analysis and visualization script
- `df.csv`: source dataset used for training and analysis

## Requirements

- Python 3.11+
- Node.js 18+
- npm

## Backend setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn backend.main:app --reload
```

The backend runs on `http://127.0.0.1:8000` by default.

### Optional environment variable

Create a `.env` file or set this variable in your shell if you want OpenRouteService routing:

```env
ORS_API_KEY=your_openrouteservice_key
```

If no key is provided, the backend will fall back to its alternate routing behavior.

## Frontend setup

Install frontend dependencies:

```bash
cd frontend
npm install
```

Create a frontend env file from the example:

```bash
cp .env.example .env
```

Start the frontend:

```bash
npm run dev
```

The frontend runs on `http://127.0.0.1:5173` by default.

## Frontend environment variables

`frontend/.env.example`

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Update that value when your backend is hosted somewhere else.

## Notes before pushing to GitHub

- Do not commit `frontend/node_modules`, `frontend/dist`, or `__pycache__`
- Keep real secrets in `.env`, not in source control
- If you deploy the frontend, set `VITE_API_BASE_URL` to your deployed backend URL

