"""Interactive CLI demo — good for a live interview walkthrough without
needing a frontend. Prints each stage of the LangGraph state so the
interviewer can see the pipeline working, not just the final answer.

Usage: python -m src.cli_demo
"""
from __future__ import annotations

from src.graph import run_query


def main() -> None:
    print("TVH Findability demo — type a scenario, or 'quit' to exit.\n")
    print('Example: "I need a warning sign for the charging area, our forklift '
          'battery gives off fumes"\n')

    while True:
        query = input("> ").strip()
        if not query or query.lower() in {"quit", "exit"}:
            break

        state = run_query(query)

        print("\n--- parsed intent ---")
        print(state.get("parsed_intent"))

        print("\n--- candidates (post semantic search + Cloud SQL enrich) ---")
        for c in state.get("candidates", []):
            print(f"  {c['ref_no']}: {c['description']}")

        print("\n--- selected match ---")
        print(f"  {state.get('best_ref_no')} — {state.get('reasoning')}")

        print("\n--- recommendations ---")
        for r in state.get("recommendations", []):
            print(f"  {r['ref_no']}: {r['description']} (score {r['score']:.2f})")

        print("\n--- final answer ---")
        print(state.get("answer"))
        print()


if __name__ == "__main__":
    main()
