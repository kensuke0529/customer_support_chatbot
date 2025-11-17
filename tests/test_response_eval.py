from langsmith import evaluate
from langsmith.schemas import Example, Run
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import sys
from pathlib import Path
import traceback
import re
import uuid

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agent import app

dataset_name = "eval_set"
eval_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def detect_escalation(response: str) -> bool:
    """Detect if escalation happened based on response content."""
    if not response:
        return False
    indicators = [
        "escalated to our support team",
        "escalated to our team",
        "escalated to the support team",
        "escalated to a human agent",
        "has been escalated",
        "query has been escalated",
        "escalation to",
    ]
    return any(ind in response.lower() for ind in indicators)


def predict_function(inputs: dict) -> dict:
    """Predict function that runs the agent and returns the response."""
    user_message = inputs.get("user_message") or inputs.get("user_query", "")
    if not user_message:
        return {"output": "Error: No user message provided", "escalated": False}

    # Generate a unique thread_id for this evaluation run
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_message": user_message,
        "classification_tag": "",
        "context": "",
        "response": "",
        "response_validation": "",
        "response_validation_reason": "",
        "response_retry_count": 0,
        "contact_info_source": "none",
        "needs_contact_info": False,
        "user_email": None,
        "user_name": None,
        "order_id": None,
        "session_id": None,
        "messages": [],
        "thread_id": thread_id,
    }

    try:
        result = app.invoke(initial_state, config=config)
        final_response = result.get("response", "").strip()
        if not final_response:
            return {"output": "Error: No response generated", "escalated": False}
        return {
            "output": final_response,
            "escalated": detect_escalation(final_response),
        }
    except Exception as e:
        print(f"ERROR in predict_function: {e}\n{traceback.format_exc()}")
        return {"output": f"Error running agent: {str(e)}", "escalated": False}


def _parse_bool(value):
    """Parse boolean from string or return bool value."""
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def escalation_evaluator(run: Run, example: Example) -> dict:
    """Evaluator that checks if escalation happened correctly."""
    try:
        outputs = run.outputs if isinstance(run.outputs, dict) else {}
        actual = (
            outputs.get("escalated")
            if "escalated" in outputs
            else detect_escalation(str(outputs.get("output", "")))
        )

        expected = False
        for source in [example.outputs, example.inputs]:
            if source and "escalate" in source:
                expected = _parse_bool(source["escalate"])
                break

        match = actual == expected
        status = "Escalated" if actual else "Not escalated"
        comment = (
            f"{status} (as expected)"
            if match
            else f"Expected {'escalation' if expected else 'no escalation'}, got {'escalation' if actual else 'no escalation'}"
        )

        return {
            "key": "escalation_check",
            "score": 1.0 if match else 0.0,
            "comment": comment,
        }
    except Exception as e:
        print(f"ERROR in escalation_evaluator: {e}\n{traceback.format_exc()}")
        return {"key": "escalation_check", "score": 0.0, "comment": str(e)}


def criteria_evaluator(run: Run, example: Example) -> dict:
    """Custom criteria evaluator using OpenAI directly"""
    try:
        pred = run.outputs.get("output", "")
        prediction = pred.get("output", "") if isinstance(pred, dict) else str(pred)
        if not prediction or prediction.startswith("Error:"):
            return {
                "key": "criteria_evaluation",
                "score": 0.0,
                "comment": f"Failed: {prediction}",
            }

        reference = example.outputs.get("expected_response") or example.outputs.get(
            "reference"
        )
        user_msg = example.inputs.get("user_message", "")

        prompt = f"""You are evaluating a customer support response against three criteria. Rate each criterion from 0-10.

USER MESSAGE: {user_msg}
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

        response = eval_llm.invoke(prompt)
        result_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        scores = {}
        for line in result_text.split("\n"):
            if ":" in line:
                try:
                    key, val = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    match = re.search(r"[\d.]+", val.strip())
                    if match and key in [
                        "policy_accuracy",
                        "specificity",
                        "completeness",
                        "average",
                    ]:
                        scores[key] = float(match.group())
                except (ValueError, AttributeError):
                    continue

        reasoning = (
            result_text.upper().split("REASONING:")[1].strip()
            if "REASONING:" in result_text.upper()
            else ""
        )
        avg_score = (
            scores.get("average", 0) / 10.0
            if "average" in scores
            else sum(
                [
                    scores.get("policy_accuracy", 0),
                    scores.get("specificity", 0),
                    scores.get("completeness", 0),
                ]
            )
            / 30.0
        )

        return {
            "key": "criteria_evaluation",
            "score": avg_score,
            "comment": f"Policy Accuracy: {scores.get('policy_accuracy', 0)}/10, Specificity: {scores.get('specificity', 0)}/10, Completeness: {scores.get('completeness', 0)}/10. {reasoning}",
        }
    except Exception as e:
        print(f"ERROR in criteria_evaluator: {e}\n{traceback.format_exc()}")
        return {"key": "criteria_evaluation", "score": 0.0, "comment": str(e)}


if __name__ == "__main__":
    try:
        results = evaluate(
            predict_function,
            data=dataset_name,
            evaluators=[criteria_evaluator, escalation_evaluator],
            experiment_prefix="response-eval-llm-judge",
        )
        results_list = list(results) if hasattr(results, "__iter__") else []

        if results_list:
            print(
                f"\n{'=' * 60}\nEVALUATION COMPLETE: {len(results_list)} responses evaluated\n{'=' * 60}\n"
            )

            criteria_scores, escalation_scores = [], []
            for result in results_list:
                feedbacks = (
                    result.feedback_stats
                    if hasattr(result, "feedback_stats")
                    else result.get("feedback_stats", [])
                ) or []
                for fb in feedbacks:
                    key, score = fb.get("key"), fb.get("score", 0)
                    (
                        criteria_scores
                        if key == "criteria_evaluation"
                        else escalation_scores
                    ).append(score) if key in [
                        "criteria_evaluation",
                        "escalation_check",
                    ] else None

            if criteria_scores:
                avg, mn, mx = (
                    sum(criteria_scores) / len(criteria_scores),
                    min(criteria_scores),
                    max(criteria_scores),
                )
                print(
                    f"Response Quality - Average: {avg:.3f} ({avg * 10:.1f}/10), Min: {mn:.3f} ({mn * 10:.1f}/10), Max: {mx:.3f} ({mx * 10:.1f}/10)"
                )
            if escalation_scores:
                acc = sum(escalation_scores) / len(escalation_scores)
                print(
                    f"Escalation Accuracy: {acc:.1%} ({int(sum(escalation_scores))}/{len(escalation_scores)} correct)"
                )
            print(f"{'=' * 60}\n")
            [
                print(f"\n--- Result {i + 1} ---\n{r}")
                for i, r in enumerate(results_list)
            ]
        else:
            print("No results returned from evaluation")
    except Exception as e:
        print(f"ERROR: {e}\n{traceback.format_exc()}")
        sys.exit(1)
