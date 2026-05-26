"""Quick sanity-check: fetch last 10 games for a Lichess user and print a table."""

import sys
import requests
from rich.console import Console
from rich.table import Table


LICHESS_API = "https://lichess.org/api"


def fetch_games(username: str, max_games: int = 10) -> list[dict]:
    url = f"{LICHESS_API}/games/user/{username}"
    params = {"max": max_games, "pgnInJson": False, "clocks": False, "evals": False}
    headers = {"Accept": "application/x-ndjson"}
    response = requests.get(url, params=params, headers=headers, stream=True, timeout=10)
    response.raise_for_status()

    games = []
    for line in response.iter_lines():
        if line:
            import json
            games.append(json.loads(line))
    return games


def result_for(game: dict, username: str) -> str:
    winner = game.get("winner")
    if winner is None:
        return "½-½"
    white_name = game["players"]["white"].get("user", {}).get("name", "?").lower()
    played_white = white_name == username.lower()
    won = (winner == "white" and played_white) or (winner == "black" and not played_white)
    return "Win" if won else "Loss"


def build_table(games: list[dict], username: str) -> Table:
    table = Table(title=f"Last {len(games)} games for [bold]{username}[/bold]")
    table.add_column("Date", style="dim")
    table.add_column("Variant")
    table.add_column("Time control")
    table.add_column("Colour")
    table.add_column("Opponent")
    table.add_column("Result")

    for game in games:
        ts = game.get("createdAt", 0) // 1000
        from datetime import datetime, timezone
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

        variant = game.get("variant", "standard")
        clock = game.get("clock")
        time_ctrl = f"{clock['initial']//60}+{clock['increment']}" if clock else "—"

        white_name = game["players"]["white"].get("user", {}).get("name", "?")
        black_name = game["players"]["black"].get("user", {}).get("name", "?")
        played_white = white_name.lower() == username.lower()
        colour = "White" if played_white else "Black"
        opponent = black_name if played_white else white_name

        result = result_for(game, username)
        result_style = {"Win": "green", "Loss": "red", "½-½": "yellow"}[result]

        table.add_row(date, variant, time_ctrl, colour, opponent, f"[{result_style}]{result}[/{result_style}]")

    return table


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python -m chess_coach.hello_lichess <username>")
        sys.exit(1)

    username = sys.argv[1]
    console = Console()

    console.print(f"Fetching games for [bold]{username}[/bold] from Lichess…")
    try:
        games = fetch_games(username)
    except requests.HTTPError as exc:
        console.print(f"[red]HTTP error:[/red] {exc}")
        sys.exit(1)

    if not games:
        console.print("[yellow]No games found.[/yellow]")
        return

    console.print(build_table(games, username))


if __name__ == "__main__":
    main()
