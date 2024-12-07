# mini-rag-bot

------------------------------------------------------------------------------------------------------------------------

### Project structure

```
mini-rag-bot/
│
├── .github/
│   └── workflows/
│       └── code_standard_check.yml     # Workflow for checking code
│
├── data/
│   ├── raw/                            # Raw data (HTML, PDFs, etc.)
│   └── processed/                      # Processed data (extracted text, chunks, etc.)
│
├── modules/
│   ├── data_acquisition/               # For data scraping, file handling, etc.
│   ├── data_processing/                # Text extraction and processing (HTML, PDF, etc.)
│   ├── llm/                            # LLM-related functions (embedding, fine-tuning, etc.)
│   ├── app_logic/                      # Main app logic related to handling queries and managing user interaction
│   └── utils/                          # Utility functions (logging, configuration, etc.)
│
├── deployment/
│   ├── llm/                            # Deployment files for LLM (e.g. Llama3.1 8b)
│   └── embedder/                       # REST API for embedding model
│
├── notebooks/                          # Jupyter notebooks for experiments and prototyping
│
├── tests/                              # Unit tests for various modules
│
├── app.py                              # Main application entry point
├── requirements-dev.txt                # Developer Dependencies
├── requirements.txt                    # Dependencies
└── README.md                           # Project documentation
```

------------------------------------------------------------------------------------------------------------------------

### Environment - setup

```bash
conda create -n mini-rag-dev -y python=3.12
conda activate mini-rag-dev
pip install --upgrade pip
pip install -r requirements-dev.txt
pre-commit install
```
