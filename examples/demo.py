"""Example client for the recommendation API.

Start the API first (in another terminal):  make serve
Then run:                                    python examples/demo.py

Shows a personalized (warm) user next to an unknown (cold) user, so the warm/cold
behaviour is visible side by side. Uses only the standard library, so it needs no
extra dependencies; point it elsewhere with API_URL=http://host:port.
"""

import json
import os
import urllib.request

BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
WARM_USER = 1488844      # a user the shipped model was trained on
COLD_USER = 999999999    # never seen -> popularity fallback


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}") as resp:
        return json.load(resp)


def show(label: str, user_id: int, k: int = 5) -> None:
    data = get(f"/recommend/{user_id}?k={k}")
    print(f"\n{label}  (user {user_id} -> segment: {data['segment']})")
    for it in data["items"]:
        print(f"  {it['rank']:>2}. {it['title']:<40} {it['score']:.2f}")


def main() -> None:
    health = get("/health")
    print(f"API up — model knows {health['n_users']:,} users, {health['n_items']:,} movies")
    show("Personalized (warm)", WARM_USER)
    show("Cold start (unknown user)", COLD_USER)


if __name__ == "__main__":
    main()
