"""
Scoring metrics using Ragas framework for RAG evaluation.
"""
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import ragas, but make it optional for environments without it
try:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning("Ragas not installed. Install with: pip install ragas")
    RAGAS_AVAILABLE = False


@dataclass
class EvaluationScores:
    """Container for evaluation scores."""
    faithfulness: float
    answer_relevance: float
    context_recall: float
    overall: float


class MetricsCalculator:
    """Calculate evaluation metrics using Ragas."""

    def __init__(self):
        """Initialize the metrics calculator."""
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "Ragas is required for evaluation. "
                "Install with: pip install ragas langchain openai"
            )

    def calculate_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics for evaluation results using Ragas.

        Args:
            evaluation_results: List of evaluation results from evaluator

        Returns:
            Dictionary with overall and per-question scores
        """
        logger.info(f"Calculating metrics for {len(evaluation_results)} results")

        # Filter out results with errors
        valid_results = [r for r in evaluation_results if "error" not in r]
        if len(valid_results) < len(evaluation_results):
            logger.warning(
                f"Skipping {len(evaluation_results) - len(valid_results)} "
                "results with errors"
            )

        if not valid_results:
            logger.error("No valid results to evaluate")
            return {"error": "No valid results"}

        # Prepare data for Ragas
        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for result in valid_results:
            dataset_dict["question"].append(result["question"])
            dataset_dict["answer"].append(result["actual_answer"])
            # Ragas expects contexts as list of strings
            dataset_dict["contexts"].append(
                result.get("context", []) if isinstance(result.get("context"), list)
                else [result.get("context", "")]
            )
            dataset_dict["ground_truth"].append(result["expected_answer"])

        # Create Ragas dataset
        dataset = Dataset.from_dict(dataset_dict)

        # Run evaluation
        logger.info("Running Ragas evaluation...")
        try:
            ragas_results = evaluate(
                dataset,
                metrics=[
                    Faithfulness(),
                    AnswerRelevancy(),
                    ContextRecall(),
                ]
            )

            # Extract scores
            scores = {
                "faithfulness": ragas_results["faithfulness"],
                "answer_relevance": ragas_results["answer_relevancy"],
                "context_recall": ragas_results["context_recall"],
            }

            # Calculate overall score
            scores["overall"] = (
                scores["faithfulness"] +
                scores["answer_relevance"] +
                scores["context_recall"]
            ) / 3

            logger.info(f"Evaluation complete. Overall score: {scores['overall']:.3f}")

            return {
                "overall_scores": scores,
                "per_question_scores": self._extract_per_question_scores(
                    ragas_results, valid_results
                ),
                "summary": self._generate_summary(scores, valid_results),
            }

        except Exception as e:
            logger.error(f"Error during Ragas evaluation: {e}")
            return {"error": str(e)}

    def _extract_per_question_scores(
        self,
        ragas_results: Dict,
        valid_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract per-question scores from Ragas results."""
        per_question = []

        # Ragas returns a dataframe-like object
        df = ragas_results.to_pandas()

        for idx, result in enumerate(valid_results):
            scores = {
                "id": result["id"],
                "category": result["category"],
                "question": result["question"],
                "faithfulness": float(df.iloc[idx]["faithfulness"]),
                "answer_relevance": float(df.iloc[idx]["answer_relevancy"]),
                "context_recall": float(df.iloc[idx]["context_recall"]),
            }
            scores["overall"] = (
                scores["faithfulness"] +
                scores["answer_relevance"] +
                scores["context_recall"]
            ) / 3

            per_question.append(scores)

        return per_question

    def _generate_summary(
        self,
        scores: Dict[str, float],
        valid_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        # Group by category
        categories = {}
        for result in valid_results:
            category = result.get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        category_stats = {}
        for category, results in categories.items():
            category_stats[category] = {
                "count": len(results),
                "percentage": len(results) / len(valid_results) * 100,
            }

        return {
            "total_questions": len(valid_results),
            "categories": category_stats,
            "passed": all(
                scores["faithfulness"] >= 0.85,
                scores["answer_relevance"] >= 0.85,
                scores["context_recall"] >= 0.80,
            ),
        }

    def check_thresholds(self, scores: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if scores meet the defined thresholds.

        Args:
            scores: Dictionary of metric scores

        Returns:
            Dictionary indicating pass/fail for each metric
        """
        from tests.evaluation import config

        results = {
            "faithfulness": scores.get("faithfulness", 0) >= config.THRESHOLDS["faithfulness"],
            "answer_relevance": scores.get("answer_relevance", 0) >= config.THRESHOLDS["answer_relevance"],
            "context_recall": scores.get("context_recall", 0) >= config.THRESHOLDS["context_recall"],
        }

        results["all_passed"] = all(results.values())

        return results
