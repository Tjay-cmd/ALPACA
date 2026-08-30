# ALPACA — AI Trading Agent

Placeholder description for an AI trading agent hackathon project using Alpaca's paper trading API.

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   On macOS/Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` (already done for this repo) and replace the placeholders with your Alpaca paper trading API key and secret.

4. Test the Alpaca connection:

   ```powershell
   python test_connection.py
   ```
