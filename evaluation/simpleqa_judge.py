"""GPT-4.1 autorater for SimpleQA Verified (google/simpleqa-verified).

Grading prompt and F1 formula copied verbatim from the openai/simple-evals
convention, per the Kaggle starter notebook for SimpleQA Verified:
https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code
"""
import os
import re

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
Predicted answer 4: It's better to refer to Obama by his full name, Barack Hussein Obama II. He has two daughters: Malia and Sasha.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Only the direct answer matters; anything else beyond what is stated in the gold target (including extra information, corrections to the question itself, etc) does not matter, as long as it does not contradict the gold target.
    - Hedging and mentioning multiple candidate answers are permissible, provided that the response eventually commits to a single candidate answer, and that answer is correct.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target.
    - A hedged statement (e.g., "it is possible that", "although i'm not sure, i think") that commits to an incorrect answer is also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
Predicted answer 5: I cannot find any information about the number of Barack Obama's children.
Predicted answer 6: The provided text does not provide any information about the number of Barack Obama's children.
Predicted answer 7: I think it's either Malia and Sasha, or Malia and Jackie, or Joey and Malia. I'm not sure which one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.
    - Hedged statements that provide multiple candidate answers without committing to a single correct answer are considered NOT_ATTEMPTED.


Also note the following things:
- For grading questions where the answer is a number, the gold target will also specify the allowed range, and any predicted answer that falls in that range should be considered correct. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k (acceptable range: anything between 118k and 122k)".
    - Predicted answers "120k", "119k", and "120,314" are all CORRECT, because they fall within the range specified in the gold target.
    - Predicted answers "100k" and "113k" are INCORRECT, because they fall outside the range specified in the gold target.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m (acceptable range: anything between 1.72 m and 1.74 m)". The predicted answer "1.74" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))
DEFAULT_GRADE_IF_UNPARSEABLE = "C"  # NOT_ATTEMPTED

GRADER_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-2025-04-14")


def grade(question: str, gold: str, predicted: str) -> str:
    """Grade one (question, gold, predicted) triple with the judge model.

    Defaults to GPT-4.1 via OpenAI. Set OPENAI_BASE_URL to point at a
    locally hosted vLLM server instead (to avoid API cost at scale).

    Returns one of "correct" / "incorrect" / "not_attempted".
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key and not base_url:
        raise RuntimeError(
            "OPENAI_API_KEY not set; required for the SimpleQA Verified GPT-4.1 autorater."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = GRADER_TEMPLATE.format(question=question, target=gold, predicted_answer=predicted)
    response = client.chat.completions.create(
        model=GRADER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"(A|B|C)", text)
    if match:
        letter = match.group(0)
    else:
        upper = text.upper()
        if "CORRECT" in upper and "INCORRECT" not in upper:
            letter = "A"
        elif "INCORRECT" in upper:
            letter = "B"
        elif "NOT_ATTEMPTED" in upper or "NOT ATTEMPTED" in upper:
            letter = "C"
        else:
            letter = DEFAULT_GRADE_IF_UNPARSEABLE
    return CHOICE_LETTER_TO_STRING[letter].lower()


def score_simpleqa(records: list[dict]) -> dict:
    """records: list of {"question", "golden_answers": [str], "predicted_answer"}.

    Returns summary dict with label counts, accuracy_given_attempted, and F1,
    per the openai/simple-evals SimpleQA scoring convention.
    """
    n = len(records)
    n_correct = n_incorrect = n_not_attempted = 0
    graded = []
    for rec in records:
        label = grade(rec["question"], rec["golden_answers"][0], rec["predicted_answer"])
        graded.append(label)
        if label == "correct":
            n_correct += 1
        elif label == "incorrect":
            n_incorrect += 1
        else:
            n_not_attempted += 1

    attempted = n_correct + n_incorrect
    accuracy_given_attempted = n_correct / attempted if attempted else 0.0
    overall_accuracy = n_correct / n if n else 0.0
    f1 = (
        2 * accuracy_given_attempted * overall_accuracy / (accuracy_given_attempted + overall_accuracy)
        if (accuracy_given_attempted + overall_accuracy)
        else 0.0
    )

    return {
        "n": n,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_not_attempted": n_not_attempted,
        "attempted": attempted,
        "accuracy_given_attempted": accuracy_given_attempted,
        "overall_accuracy": overall_accuracy,
        "f1": f1,
        "grades": graded,
    }
