from langsmith import evaluate
from langsmith.schemas import Example, Run
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent import app

dataset_name = "ds-internal-caffeine-94"


class ClassificationEvaluator:
    """Simple evaluator that checks if classification matches expected value."""

    def __call__(self, run: Run, example: Example) -> dict:
        predicted = run.outputs.get("classification_tag", "").lower().strip()
        expected = example.outputs.get("expected_classification", "").lower().strip()
        confidence = run.outputs.get("classification_confidence", None)
        is_correct = predicted == expected

        result = {
            "key": "classification_accuracy",
            "score": 1.0 if is_correct else 0.0,
            "comment": f"Predicted: '{run.outputs.get('classification_tag', '')}', Expected: '{example.outputs.get('expected_classification', '')}'",
        }
        # Add confidence to the LangSmith result
        if confidence is not None:
            result["classification_confidence"] = confidence
        return result


classification_evaluator = ClassificationEvaluator()


def predict_function(inputs: dict) -> dict:
    user_message = inputs.get("user_message", "")

    initial_state = {
        "user_message": user_message,
        "classification_tag": "",
        "classification_confidence": 0.0,
        "context": "",
        "response": "",
    }

    result = app.invoke(initial_state)

    return {
        "response": result.get("response", ""),
        "classification_tag": result.get("classification_tag", ""),
        "classification_confidence": result.get("classification_confidence", 0.0),
        "context": result.get("context", ""),
    }


def run_classification_evaluation_with_dataset(
    dataset_name: str = dataset_name,
):
    results = evaluate(
        predict_function,
        data=dataset_name,
        evaluators=[classification_evaluator],
    )

    return results


if __name__ == "__main__":
    try:
        results = run_classification_evaluation_with_dataset()

        if hasattr(results, "__iter__"):
            results_list = list(results)
            print(f"Number of results: {len(results_list)}")
            # Print first few results
            for i, result in enumerate(results_list[:3]):
                print(f"\nResult {i + 1}: {result}")
    except Exception as e:
        print(f"Error during classification evaluation: {e}")
        import traceback

        traceback.print_exc()
