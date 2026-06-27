"""
Evaluate classifier accuracy against the sample_queries fixture.
Usage: python scripts/evaluate_classifier.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier.intent_classifier import IntentClassifier
from src.utils.logger import configure_logging, get_logger

configure_logging("INFO")
log = get_logger(__name__)

FIXTURES = Path("tests/fixtures/sample_queries.json")


async def main() -> None:
    samples = json.loads(FIXTURES.read_text())
    classifier = IntentClassifier()
    correct = 0

    for sample in samples:
        result = await classifier.classify(sample["query"])
        match = result.intent == sample["expected_intent"]
        if match:
            correct += 1
        status = "PASS" if match else "FAIL"
        print(f"[{status}] '{sample['query'][:60]}' → {result.intent} (expected: {sample['expected_intent']})")

    accuracy = correct / len(samples)
    print(f"\nAccuracy: {correct}/{len(samples)} = {accuracy:.1%}")
    log.info("eval_complete", accuracy=accuracy, total=len(samples), correct=correct)


if __name__ == "__main__":
    asyncio.run(main())
