"""
Configuration for LLM evaluation framework.
"""
import os

# API Configuration
CHATBOT_API_URL = os.getenv("JENKINS_CHATBOT_URL", "http://localhost:8000/api/chatbot")

# LLM Judge Configuration
JUDGE_LLM_PROVIDER = os.getenv("JUDGE_LLM_PROVIDER", "openai")  # openai, anthropic, or local
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluation Thresholds
THRESHOLDS = {
    "faithfulness": 0.85,
    "context_recall": 0.80,
    "answer_relevance": 0.85,
}

# Dataset Configuration
GOLDEN_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "golden_qa.json"
)

# Timeout Configuration (seconds)
API_TIMEOUT = int(os.getenv("EVAL_API_TIMEOUT", "30"))
EVAL_TIMEOUT = int(os.getenv("EVAL_TOTAL_TIMEOUT", "600"))

# Reporting
REPORT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")
VERBOSE_OUTPUT = os.getenv("EVAL_VERBOSE", "false").lower() == "true"

# Categories to evaluate (None = all)
EVAL_CATEGORIES = os.getenv("EVAL_CATEGORIES", "").split(",") if os.getenv("EVAL_CATEGORIES") else None

# Ragas Configuration
RAGAS_EMBEDDINGS_MODEL = os.getenv("RAGAS_EMBEDDINGS_MODEL", "text-embedding-ada-002")
