# AI Enquiry Classification: Support Ticket Routing

This project trains a simple text classifier to route customer support tickets into four categories:
**Billing**, **Technical Issue**, **Account Access**, and **General Inquiry**. It also includes a Streamlit app for single-text prediction.

## Project Structure
```
support_ticket_routing/
├── app/
│   └── app.py
├── data/
│   └── sample_tickets.csv
├── models/
│   └── (trained model will be saved here)
├── scripts/
│   └── train.py
├── requirements.txt
└── README.md
```

## Quickstart

1. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (uses the sample dataset by default)

   ```bash
   python scripts/train.py --data data/sample_tickets.csv --out models/ticket_router.joblib
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app/app.py
   ```

   If you've saved the model in a custom path, set an environment variable before launching:
   ```bash
   # Example
   export MODEL_PATH=models/ticket_router.joblib
   streamlit run app/app.py
   ```

## Using Your Own Data

Provide a CSV with two columns:
- `text`: the customer message
- `label`: one of `Billing`, `Technical Issue`, `Account Access`, `General Inquiry` 

Example:
```csv
text,label
"I can't log in to my account","Account Access"
"I was charged twice for my subscription","Billing"
```

Then re-train:
```bash
python scripts/train.py --data path/to/your.csv --out models/ticket_router.joblib
```

## Notes
- The baseline model is a TF–IDF + Logistic Regression classifier, which is fast and effective for short texts.
- For more data and better accuracy, expand `data/sample_tickets.csv` with real tickets.
- You can switch to `LinearSVC` for strong performance on small datasets (but you lose probability scores).

## Troubleshooting
- **Model not found**: Train the model first (step 3) or set `MODEL_PATH` correctly.
- **App doesn't start**: Ensure `streamlit` is installed and you're in the project root when running `streamlit run app/app.py`.
- **Poor accuracy**: Add more labeled examples, include domain-specific synonyms, or tune `TfidfVectorizer` and model parameters in `scripts/train.py`.