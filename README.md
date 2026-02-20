# Bloei Rekenmodule – Proof of Concept

A standalone Python proof-of-concept application for calculating investment fees and long-term average costs for Bloei Vermogen.

## Project Structure

```
bloei-calc-demo/
├── bloei_rekenmodel/
│   ├── __init__.py
│   ├── domain.py          # RekenInput and RekenOutput dataclasses
│   └── logic.py           # bereken_kosten calculation function
├── app_streamlit.py       # Streamlit UI application
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app_streamlit.py
```

The application will open in your default web browser, typically at `http://localhost:8501`.

## Usage

1. Enter the starting investment amount (Startvermogen)
2. Select the risk profile (Profiel): Defensief, Matig defensief, Neutraal, Offensief, or Zeer offensief
3. Adjust the investment horizon (Horizon) using the slider (1-40 years)
4. Set the number of scenarios (Aantal Scenario's)
5. Click "Bereken" to calculate and display the results

Note: The broker is fixed to Saxo Bank and is not a variable in the calculation.

## Important Notes

⚠️ **This is a demo proof-of-concept with dummy formulas and no real data source.**

- All calculation logic is simplified for demonstration purposes
- No Excel integration or external data sources
- No database connections
- All calculations are performed in-memory based on input parameters

The actual business logic will be implemented when this is integrated into the production backend for Power Apps and Dataverse.

## Development

The codebase is structured to be easily extensible:

- `domain.py`: Contains the data models (RekenInput, RekenOutput)
- `logic.py`: Contains pure calculation functions (no I/O, no side effects)
- `app_streamlit.py`: Contains the Streamlit UI layer

This structure allows the calculation logic to be reused in other contexts (e.g., REST API, Power Apps backend) without modification.
