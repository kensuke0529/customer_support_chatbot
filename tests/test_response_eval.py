from langsmith import evaluate
from langsmith.schemas import Example, Run
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import sys
from pathlib import Path


load_dotenv()

# Add src path for agent import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent import app

dataset_name = "Response_eval"


def predict_function(inputs: dict) -> dict:
    user_message = inputs.get("user_message", "")
    initial_state = {
        "user_message": user_message,
        "classification_tag": "",
        "classification_confidence": 0.0,
        "context": "",
        "response": "",
    }

    # Run the app (chain, graph, or agent)
    result = app.invoke(initial_state)

    # Extract only the conversational response
    final_response = result.get("response", "").strip()
    return {"output": final_response}


# Initialize OpenAI LLM for evaluation
eval_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def criteria_evaluator(run: Run, example: Example) -> dict:
    """Custom criteria evaluator using OpenAI directly"""
    # Get prediction output (may be string or dict)
    prediction_raw = run.outputs.get("output", "")
    prediction = (
        prediction_raw.get("output", "")
        if isinstance(prediction_raw, dict)
        else prediction_raw
    )

    # Get reference if available
    reference = example.outputs.get("expected_response") or example.outputs.get(
        "reference"
    )
    user_message = example.inputs.get("user_message", "")

    # Create evaluation prompt
    eval_prompt = f"""You are evaluating a customer support response against three criteria. Rate each criterion from 0-10.

USER MESSAGE: {user_message}

ACTUAL RESPONSE: {prediction}

REFERENCE ANSWER: {reference if reference else "Not provided"}

Evaluate on these criteria:

1. POLICY ACCURACY (0-10): Does the response EXACTLY follow the company policy without deviation?
Score 10: Response matches policy precisely - correct paths, timeframes, processes, and prioritizes self-service when available
Score 7-9: Response is mostly correct but missing minor details or slightly imprecise
Score 4-6: Response is partially correct but has significant deviations from policy
Score 0-3: Response contradicts policy, provides wrong information, or tells user to contact support when self-service is available in policy

CRITICAL: If policy shows self-service option (e.g., 'Settings > Account > Email') but response says 'contact support', score must be 0-3.
CRITICAL: Check that exact settings paths are mentioned when policy provides them.
CRITICAL: Verify timeframes match policy exactly (e.g., '5-7 business days' not '1-3 days').

2. SPECIFICITY (0-10): How specific and actionable is the response?
Score 10: Includes exact settings paths (e.g., 'Settings > Billing > Invoices'), precise timeframes (e.g., '3-5 business days'), step-by-step instructions
Score 7-9: Specific but could be more precise (e.g., says 'in your settings' instead of exact path)
Score 4-6: Somewhat vague, uses terms like 'soon', 'quickly', 'contact support' without specific guidance
Score 0-3: Very vague, no actionable steps, generic responses

3. COMPLETENESS (0-10): Does the response include ALL necessary information to fully resolve the issue?
Score 10: Includes everything needed - timeframes, exact steps, what to expect, any requirements
Score 7-9: Includes most necessary information but missing one minor detail
Score 4-6: Missing multiple important details that user would need
Score 0-3: Lacks critical information, user cannot take action

Provide your response in this exact format:
POLICY_ACCURACY: [score]
SPECIFICITY: [score]
COMPLETENESS: [score]
AVERAGE: [average of three scores]
REASONING: [brief explanation of scores]"""

    # Get evaluation from LLM
    response = eval_llm.invoke(eval_prompt)
    result_text = response.content

    # Parse scores
    scores = {}
    lines = result_text.split("\n")
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in ["policy_accuracy", "specificity", "completeness", "average"]:
                try:
                    scores[key] = float(value)
                except:
                    pass

    # Get reasoning
    reasoning = ""
    if "REASONING:" in result_text:
        reasoning = result_text.split("REASONING:")[1].strip()

    # Calculate average score (normalized to 0-1)
    avg_score = scores.get("average", 0) / 10.0

    return {
        "key": "criteria_evaluation",
        "score": avg_score,
        "comment": f"Policy Accuracy: {scores.get('policy_accuracy', 0)}/10, "
        f"Specificity: {scores.get('specificity', 0)}/10, "
        f"Completeness: {scores.get('completeness', 0)}/10. "
        f"{reasoning}",
    }


if __name__ == "__main__":
    results = evaluate(
        predict_function,
        data=dataset_name,
        evaluators=[criteria_evaluator],
        experiment_prefix="response-eval-llm-judge",
    )

    if hasattr(results, "__iter__"):
        results_list = list(results)
        print(f"\n{'=' * 60}")
        print(f"EVALUATION COMPLETE: {len(results_list)} responses evaluated")
        print(f"{'=' * 60}\n")

        for i, result in enumerate(results_list):
            print(f"\n--- Result {i + 1} ---")
            print(result)
