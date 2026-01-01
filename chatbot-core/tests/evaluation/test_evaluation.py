"""
Pytest test suite for LLM evaluation.
"""
import os
import json
import pytest
import logging
from datetime import datetime

from tests.evaluation.metrics.evaluator import ChatbotEvaluator
from tests.evaluation.metrics.scoring import MetricsCalculator, RAGAS_AVAILABLE
from tests.evaluation import config

logger = logging.getLogger(__name__)


@pytest.fixture
def evaluator():
    """Create evaluator instance."""
    evaluator = ChatbotEvaluator()
    yield evaluator
    evaluator.close()


@pytest.fixture
def metrics_calculator():
    """Create metrics calculator instance."""
    if not RAGAS_AVAILABLE:
        pytest.skip("Ragas not installed")
    return MetricsCalculator()


@pytest.fixture
def golden_dataset():
    """Load golden dataset."""
    with open(config.GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


class TestEvaluation:
    """Test suite for chatbot evaluation."""

    def test_golden_dataset_format(self, golden_dataset):
        """Test that golden dataset has correct format."""
        assert "questions" in golden_dataset
        assert "metadata" in golden_dataset
        assert len(golden_dataset["questions"]) > 0

        # Check first question has required fields
        question = golden_dataset["questions"][0]
        required_fields = ["id", "category", "question", "expected_answer"]
        for field in required_fields:
            assert field in question, f"Missing required field: {field}"

    def test_dataset_categories(self, golden_dataset):
        """Test that dataset covers all required categories."""
        questions = golden_dataset["questions"]
        categories = {q["category"] for q in questions}

        required_categories = {"jenkins_core", "plugins", "errors"}
        assert required_categories.issubset(categories), \
            f"Missing categories: {required_categories - categories}"

    @pytest.mark.skipif(
        not os.getenv("RUN_LIVE_EVAL"),
        reason="Live evaluation disabled. Set RUN_LIVE_EVAL=1 to run"
    )
    def test_evaluate_jenkins_core(self, evaluator, metrics_calculator):
        """Evaluate Jenkins Core questions."""
        # Filter for jenkins_core questions only
        questions = [
            q for q in evaluator.load_golden_dataset()
            if q.get("category") == "jenkins_core"
        ]

        # Run evaluation
        results = evaluator.evaluate_dataset(questions)
        assert len(results) > 0

        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(results)
        assert "overall_scores" in metrics

        # Check thresholds
        threshold_check = metrics_calculator.check_thresholds(
            metrics["overall_scores"]
        )

        # Log results
        logger.info(f"\nJenkins Core Evaluation Results:")
        logger.info(f"  Faithfulness: {metrics['overall_scores']['faithfulness']:.3f}")
        logger.info(f"  Answer Relevance: {metrics['overall_scores']['answer_relevance']:.3f}")
        logger.info(f"  Context Recall: {metrics['overall_scores']['context_recall']:.3f}")
        logger.info(f"  Overall: {metrics['overall_scores']['overall']:.3f}")

        # Assert all thresholds passed
        assert threshold_check["all_passed"], \
            f"Thresholds not met: {threshold_check}"

    @pytest.mark.skipif(
        not os.getenv("RUN_LIVE_EVAL"),
        reason="Live evaluation disabled. Set RUN_LIVE_EVAL=1 to run"
    )
    def test_evaluate_plugins(self, evaluator, metrics_calculator):
        """Evaluate Plugins questions."""
        questions = [
            q for q in evaluator.load_golden_dataset()
            if q.get("category") == "plugins"
        ]

        results = evaluator.evaluate_dataset(questions)
        metrics = metrics_calculator.calculate_metrics(results)
        threshold_check = metrics_calculator.check_thresholds(
            metrics["overall_scores"]
        )

        logger.info(f"\nPlugins Evaluation Results:")
        logger.info(f"  Faithfulness: {metrics['overall_scores']['faithfulness']:.3f}")
        logger.info(f"  Answer Relevance: {metrics['overall_scores']['answer_relevance']:.3f}")
        logger.info(f"  Context Recall: {metrics['overall_scores']['context_recall']:.3f}")

        assert threshold_check["all_passed"], \
            f"Thresholds not met: {threshold_check}"

    @pytest.mark.skipif(
        not os.getenv("RUN_LIVE_EVAL"),
        reason="Live evaluation disabled. Set RUN_LIVE_EVAL=1 to run"
    )
    def test_evaluate_errors(self, evaluator, metrics_calculator):
        """Evaluate Error questions."""
        questions = [
            q for q in evaluator.load_golden_dataset()
            if q.get("category") == "errors"
        ]

        results = evaluator.evaluate_dataset(questions)
        metrics = metrics_calculator.calculate_metrics(results)
        threshold_check = metrics_calculator.check_thresholds(
            metrics["overall_scores"]
        )

        logger.info(f"\nErrors Evaluation Results:")
        logger.info(f"  Faithfulness: {metrics['overall_scores']['faithfulness']:.3f}")
        logger.info(f"  Answer Relevance: {metrics['overall_scores']['answer_relevance']:.3f}")
        logger.info(f"  Context Recall: {metrics['overall_scores']['context_recall']:.3f}")

        assert threshold_check["all_passed"], \
            f"Thresholds not met: {threshold_check}"

    @pytest.mark.skipif(
        not os.getenv("RUN_FULL_EVAL"),
        reason="Full evaluation disabled. Set RUN_FULL_EVAL=1 to run"
    )
    def test_full_evaluation(self, evaluator, metrics_calculator):
        """Run full evaluation on all questions."""
        # Load all questions
        results = evaluator.evaluate_dataset()

        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(results)
        assert "overall_scores" in metrics
        assert "per_question_scores" in metrics
        assert "summary" in metrics

        # Save report
        self._save_evaluation_report(metrics)

        # Check thresholds
        threshold_check = metrics_calculator.check_thresholds(
            metrics["overall_scores"]
        )

        # Log detailed results
        logger.info("\n" + "="*60)
        logger.info("FULL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Faithfulness: {metrics['overall_scores']['faithfulness']:.3f} "
                   f"(threshold: {config.THRESHOLDS['faithfulness']})")
        logger.info(f"  Answer Relevance: {metrics['overall_scores']['answer_relevance']:.3f} "
                   f"(threshold: {config.THRESHOLDS['answer_relevance']})")
        logger.info(f"  Context Recall: {metrics['overall_scores']['context_recall']:.3f} "
                   f"(threshold: {config.THRESHOLDS['context_recall']})")
        logger.info(f"  Overall Score: {metrics['overall_scores']['overall']:.3f}")

        logger.info(f"\nSummary:")
        logger.info(f"  Total Questions: {metrics['summary']['total_questions']}")
        logger.info(f"  Passed: {metrics['summary']['passed']}")

        logger.info(f"\nCategory Breakdown:")
        for category, stats in metrics['summary']['categories'].items():
            logger.info(f"  {category}: {stats['count']} questions ({stats['percentage']:.1f}%)")

        # Assert thresholds
        assert threshold_check["all_passed"], \
            f"Evaluation failed to meet thresholds: {threshold_check}"

    def _save_evaluation_report(self, metrics: dict):
        """Save evaluation report to file."""
        os.makedirs(config.REPORT_OUTPUT_DIR, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(
            config.REPORT_OUTPUT_DIR,
            f"evaluation_report_{timestamp}.json"
        )

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"\nEvaluation report saved to: {report_file}")
