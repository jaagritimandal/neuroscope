FROM python:3.11-slim

LABEL maintainer="jaagritimandal"
LABEL description="NeuroScope — Brain Aging & Clinical Dementia Status Explorer"

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and config
COPY app.py .
COPY pages/ pages/
COPY .streamlit/ .streamlit/
COPY src/ src/

# Copy pre-generated analysis artifacts
# (data/, reports/, figures/ must exist at build time)
COPY data/oasis1_original_export.csv data/oasis1_original_export.csv
COPY data/oasis1_labeled_cohort.csv data/oasis1_labeled_cohort.csv
COPY reports/ reports/
COPY figures/ figures/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
