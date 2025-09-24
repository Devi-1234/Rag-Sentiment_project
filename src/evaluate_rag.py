import json
import re

def normalize(text):
    """Lowercase + remove punctuation for easy matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def evaluate(questions_file, answers_file):
    # Load data
    with open(questions_file, "r") as f:
        questions = json.load(f)
    with open(answers_file, "r") as f:
        answers = json.load(f)

    total = 0
    correct = 0

    print("\n--- Evaluation Report ---\n")
    for q in questions:
        qid = q["id"]
        gold_keywords = [normalize(a) for a in q["answers"]]
        pred = normalize(answers.get(qid, ""))

        # Check if at least one keyword appears in prediction
        hit = any(gk in pred for gk in gold_keywords)

        total += 1
        if hit:
            correct += 1
            status = "✅ Correct"
        else:
            status = "❌ Wrong"

        print(f"{qid}: {status}")
        print(f"   Q: {q['question']}")
        print(f"   Gold: {q['answers']}")
        print(f"   Pred: {answers.get(qid, '')}\n")

    print(f"\nFinal Accuracy: {correct}/{total} = {correct/total:.2%}")

if __name__ == "__main__":
    evaluate("data/corpus/questions.json", "submissions/rag_answers.json")
