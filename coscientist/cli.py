"""Command-line interface for the AI Co-Scientist."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from coscientist.core.models import AgentEvent, AgentKind

app = typer.Typer(add_completion=False, help="AI Co-Scientist: multi-agent protein binder design")
console = Console()

_AGENT_STYLE = {
    AgentKind.SUPERVISOR: "bold white",
    AgentKind.GENERATION: "cyan",
    AgentKind.REFLECTION: "yellow",
    AgentKind.PROXIMITY: "magenta",
    AgentKind.RANKING: "green",
    AgentKind.EVOLUTION: "blue",
    AgentKind.META_REVIEW: "bold magenta",
}


@app.command()
def run(
    goal: str = typer.Argument("", help="Research goal in natural language (uses config default if empty)"),
    config: str = typer.Option("default", "--config", "-c", help="Config preset or YAML path"),
    rounds: int = typer.Option(None, "--rounds", "-r", help="Override number of rounds"),
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider: bedrock | mock"),
    scorer: str = typer.Option(None, "--scorer", "-s", help="Scorer: heuristic | esm | predictor"),
    db: str = typer.Option("data/coscientist.db", "--db", help="SQLite path"),
):
    """Run a full co-scientist session and print the ranked results."""
    from coscientist.engine import run_session

    leaderboard: list[dict] = []
    log_lines: list[str] = []

    def render() -> Panel:
        table = Table(title="Hypothesis Leaderboard (Elo)", expand=True)
        table.add_column("#", width=3)
        table.add_column("Title")
        table.add_column("Elo", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("W/L", justify="right")
        for i, h in enumerate(leaderboard[:10], 1):
            score = f"{h['score']:.3f}" if h.get("score") is not None else "-"
            table.add_row(str(i), h["title"][:48], f"{h['elo']:.0f}", score, f"{h['wins']}/{h['losses']}")
        log = "\n".join(log_lines[-8:])
        return Panel.fit(Table.grid()) if False else Panel(_stack(table, log))

    def _stack(table, log):
        from rich.console import Group

        return Group(table, Panel(log, title="Activity", height=10))

    def on_event(ev: AgentEvent):
        nonlocal leaderboard
        style = _AGENT_STYLE.get(ev.agent, "white")
        if ev.kind == "leaderboard":
            leaderboard = ev.data.get("leaderboard", [])
        if ev.message:
            log_lines.append(f"[{style}]{ev.agent.value}[/{style}] r{ev.round}: {ev.message}")
        live.update(render())

    console.print(Panel(f"[bold]Goal:[/bold] {goal or '(config default)'}\n[dim]config={config}[/dim]"))
    with Live(console=console, refresh_per_second=8) as live:
        live.update(render())
        overview, ctx = run_session(
            goal, config=config, rounds=rounds, provider=provider, scorer=scorer, db_path=db,
            emit=on_event,
        )

    console.rule("[bold green]Research Overview")
    console.print(overview.markdown)
    console.rule()
    usage = ctx.llm.usage.summary()
    console.print(
        f"[dim]LLM calls: {usage['calls']}  tokens(in/out): "
        f"{usage['input_tokens']}/{usage['output_tokens']}  ~${usage['cost_usd']}[/dim]"
    )
    console.print(f"[dim]Session: {ctx.session.id}  artifacts: data/artifacts/{ctx.session.id}/final/[/dim]")


@app.command()
def sessions(db: str = typer.Option("data/coscientist.db", "--db")):
    """List past sessions."""
    from coscientist.core.store import Store

    store = Store(db)
    table = Table(title="Sessions")
    table.add_column("ID")
    table.add_column("Goal")
    table.add_column("Rounds")
    table.add_column("State")
    for s in store.list_sessions():
        table.add_row(s.id, s.goal[:60], str(s.rounds), s.state)
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Launch the web UI + API server."""
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
