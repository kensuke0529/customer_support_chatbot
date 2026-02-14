import json
from langchain_openai import ChatOpenAI
from state import ChatbotInfo
from prompts import CLASSIFICATION_PROMPT
from .base import openai_api_key

classification_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0,
)
classification_chain = CLASSIFICATION_PROMPT | classification_llm

def classify_intent(state: ChatbotInfo):
    """Classifies the user's intent from their message."""
    response = classification_chain.invoke({"input": state.user_message})
    response = response.content
    result = json.loads(response)

    return {
        "classification_tag": result.get("intent", ""),
    }

def classify_and_extract_parallel(state: ChatbotInfo):

    from concurrent.futures import ThreadPoolExecutor
    from .extraction import extract_user_info

    results = {}

    def run_classification():
        try:
            return classify_intent(state)
        except Exception as e:
            print(f"Classification error: {e}")
            return {"classification_tag": "general"}

    def run_extraction():
        try:
            return extract_user_info(state)
        except Exception as e:
            print(f"Extraction error: {e}")
            return {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        classify_future = executor.submit(run_classification)
        extract_future = executor.submit(run_extraction)

        try:
            classify_result = classify_future.result(timeout=30)
            results.update(classify_result)
        except Exception as e:
            print(f"Classification timeout/error: {e}")
            results["classification_tag"] = "general"

        try:
            extract_result = extract_future.result(timeout=30)
            results.update(extract_result)
        except Exception as e:
            print(f"Extraction timeout/error: {e}")

    print(f"Parallel classify+extract completed: tag={results.get('classification_tag')}")
    return results
