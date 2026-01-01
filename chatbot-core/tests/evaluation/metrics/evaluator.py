"""
Evaluator for Jenkins Chatbot using LLM-as-a-judge methodology.
"""
import json
import logging
from typing import Dict, List, Any
import requests
from dataclasses import dataclass

from tests.evaluation import config

logger = logging.getLogger(__name__)


@dataclass
class ChatbotResponse:
    """Response from the chatbot API."""
    question: str
    answer: str
    context: List[str]
    sources: List[str]
    metadata: Dict[str, Any]


class ChatbotEvaluator:
    """Evaluates chatbot responses against golden dataset."""

    def __init__(self, api_url: str = None, timeout: int = None):
        """
        Initialize the evaluator.

        Args:
            api_url: URL of the chatbot API base URL
            timeout: Timeout for API requests in seconds
        """
        self.api_url = api_url or config.CHATBOT_API_URL
        self.timeout = timeout or config.API_TIMEOUT
        self.session = requests.Session()
        self.session_id = None

    def load_golden_dataset(self, dataset_path: str = None) -> List[Dict[str, Any]]:
        """
        Load the golden Q&A dataset.

        Args:
            dataset_path: Path to the golden dataset JSON file

        Returns:
            List of question-answer pairs
        """
        path = dataset_path or config.GOLDEN_DATASET_PATH
        logger.info(f"Loading golden dataset from {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = data.get("questions", [])
        logger.info(f"Loaded {len(questions)} questions from dataset")

        # Filter by categories if specified
        if config.EVAL_CATEGORIES:
            questions = [
                q for q in questions
                if q.get("category") in config.EVAL_CATEGORIES
            ]
            logger.info(f"Filtered to {len(questions)} questions for categories: {config.EVAL_CATEGORIES}")

        return questions

    def create_session(self) -> str:
        """
        Create a new chat session.

        Returns:
            Session ID

        Raises:
            requests.RequestException: If the API request fails
        """
        try:
            response = self.session.post(
                f"{self.api_url}/sessions",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            session_id = data.get("session_id")
            logger.debug(f"Created session: {session_id}")
            return session_id
        except requests.RequestException as e:
            logger.error(f"Error creating session: {e}")
            raise

    def query_chatbot(self, question: str, session_id: str = None) -> ChatbotResponse:
        """
        Query the chatbot API.

        Args:
            question: The question to ask
            session_id: Session ID for the chat (creates new if None)

        Returns:
            ChatbotResponse object

        Raises:
            requests.RequestException: If the API request fails
        """
        # Create session if not provided
        if session_id is None:
            if self.session_id is None:
                self.session_id = self.create_session()
            session_id = self.session_id

        payload = {
            "message": question,
        }

        logger.debug(f"Querying chatbot: {question[:50]}...")

        try:
            response = self.session.post(
                f"{self.api_url}/sessions/{session_id}/message",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            return ChatbotResponse(
                question=question,
                answer=data.get("response", ""),
                context=data.get("context", []),
                sources=data.get("sources", []),
                metadata=data.get("metadata", {})
            )

        except requests.RequestException as e:
            logger.error(f"Error querying chatbot: {e}")
            raise

    def evaluate_dataset(self, questions: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate all questions in the dataset.

        Args:
            questions: List of question dictionaries. If None, loads from config.

        Returns:
            List of evaluation results
        """
        if questions is None:
            questions = self.load_golden_dataset()

        results = []

        for idx, question_data in enumerate(questions, 1):
            question = question_data["question"]
            question_id = question_data.get("id", f"q_{idx}")

            logger.info(f"[{idx}/{len(questions)}] Evaluating: {question_id}")

            try:
                # Create a new session for each question
                session_id = self.create_session()
                
                # Query the chatbot
                response = self.query_chatbot(question, session_id=session_id)

                # Store result
                result = {
                    "id": question_id,
                    "category": question_data.get("category"),
                    "difficulty": question_data.get("difficulty"),
                    "question": question,
                    "expected_answer": question_data.get("expected_answer"),
                    "actual_answer": response.answer,
                    "context": response.context,
                    "sources": response.sources,
                    "expected_keywords": question_data.get("expected_context_keywords", []),
                    "ground_truth_sources": question_data.get("ground_truth_sources", []),
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to evaluate question {question_id}: {e}")
                results.append({
                    "id": question_id,
                    "category": question_data.get("category"),
                    "question": question,
                    "error": str(e),
                })

        logger.info(f"Completed evaluation of {len(results)} questions")
        return results

    def close(self):
        """Close the session."""
        self.session.close()
