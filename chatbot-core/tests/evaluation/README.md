# LLM Evaluation Framework for Jenkins Chatbot

This directory contains the automated evaluation system for the Jenkins AI Chatbot, implementing "LLM-as-a-judge" methodology to ensure quality and prevent regression in chatbot responses.

## Directory Structure

```
tests/evaluation/
├── datasets/
│   └── golden_qa.json          # Golden dataset of verified Q&A pairs
├── metrics/
│   ├── __init__.py
│   ├── evaluator.py            # Main evaluation logic
│   └── scoring.py              # Metric calculations
├── test_evaluation.py          # Pytest suite for evaluation
├── config.py                   # Configuration and thresholds
└── README.md                   # This file
```

## Quick Start

### Run Evaluation Locally

```bash
cd chatbot-core
pytest tests/evaluation/test_evaluation.py -v
```

### Run on Specific Category

```bash
pytest tests/evaluation/test_evaluation.py -k "jenkins_core"
```

## Dataset Format

The `golden_qa.json` file contains verified Question/Answer pairs:

```json
{
  "id": "core_001",
  "category": "jenkins_core",
  "difficulty": "easy",
  "question": "What is Jenkins?",
  "expected_answer": "Jenkins is an open-source automation server...",
  "expected_context_keywords": ["automation", "CI/CD"],
  "ground_truth_sources": ["jenkins.io/doc/"]
}
```

### Categories
- `jenkins_core`: Core Jenkins functionality
- `plugins`: Jenkins plugin usage
- `errors`: Error troubleshooting

## Metrics

We evaluate three key metrics using LLM-as-a-judge:

### 1. Faithfulness (Target: ≥ 0.85)
Measures if the response is grounded in the retrieved context without hallucination.

### 2. Context Recall (Target: ≥ 0.80)
Measures if the retrieval system found the relevant documents.

### 3. Answer Relevance (Target: ≥ 0.85)
Measures if the response actually answers the question asked.

## CI Integration

The evaluation runs automatically on PRs when:
- PR has label `evaluate-llm`
- PR comment contains `/run-evaluation`

### GitHub Actions Workflow

Checks run and fail if any metric falls below threshold:
```yaml
- Faithfulness: 0.85
- Context Recall: 0.80
- Answer Relevance: 0.85
```

## Adding New Questions

1. Edit `datasets/golden_qa.json`
2. Follow the existing schema
3. Include diverse difficulty levels
4. Verify the expected answer is accurate
5. Add relevant keywords for context checking

## Interpreting Results

Evaluation results show:
- **Overall Score**: Average across all metrics
- **Per-Question Scores**: Detailed breakdown
- **Category Performance**: Performance by category (core/plugins/errors)

Example output:
```
EVALUATION RESULTS
==================
Overall Faithfulness: 0.92 ✓
Overall Context Recall: 0.85 ✓
Overall Answer Relevance: 0.89 ✓

Category: jenkins_core
  Faithfulness: 0.94
  Context Recall: 0.87
  Answer Relevance: 0.91
```

## Framework Choice

We use **Ragas** framework because:
- Built specifically for RAG evaluation
- Supports faithfulness, context recall, and answer relevance
- Integrates well with LangChain
- Active community and good documentation

## Local Development

### Install Dependencies

```bash
pip install ragas langchain openai
```

### Set Environment Variables

```bash
export OPENAI_API_KEY="your-key"  # For judge LLM
export JENKINS_CHATBOT_URL="http://localhost:8000"
```

### Run Tests

```bash
pytest tests/evaluation/ -v --tb=short
```

## Cost Considerations

- Evaluation uses an LLM judge (GPT-4 or equivalent)
- ~100 questions × 3 metrics ≈ 300 LLM calls per run
- Estimated cost: $0.50 - $2.00 per full evaluation
- Runs triggered manually via label/comment to control costs

## Troubleshooting

**Issue**: Evaluation fails with API errors
- **Solution**: Check API keys and rate limits

**Issue**: Low scores after valid changes
- **Solution**: Review questions, update golden dataset if needed

**Issue**: Timeout errors
- **Solution**: Increase timeout in config.py

## References

- [Ragas Documentation](https://docs.ragas.io/)
- [LLM-as-a-Judge Pattern](https://arxiv.org/abs/2306.05685)
- [Jenkins Chatbot GSoC 2025 Report](../../gsoc_report.pdf)
