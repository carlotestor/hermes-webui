from types import SimpleNamespace as NS

from tests.conftest import requires_agent_modules


@requires_agent_modules
def test_webui_loop_matches_cli(tmp_path, monkeypatch):
    from hermes_cli import goals, loops as agent
    from api import background_process, loops, models, profiles, routes
    msgs, started = [], []
    monkeypatch.setattr(goals, "_DB_CACHE", {})
    monkeypatch.setattr(profiles, "get_hermes_home_for_profile", lambda p: tmp_path)
    monkeypatch.setattr(profiles, "_profiles_root", lambda: tmp_path / "none")
    monkeypatch.setattr(models, "get_session", lambda sid, **kw: NS(profile=None, messages=msgs))
    monkeypatch.setattr(background_process, "_session_has_active_turn", lambda sid: False)
    monkeypatch.setattr(routes, "start_session_turn", lambda sid, m, source: started.append(m) or {})

    def tick(reply, max_ticks=100):  # a due tick fires, its turn ends with `reply`, the scheduler judges it
        s = agent.load_loop("s1")
        s.next_due_at, s.max_ticks = 0, max_ticks
        agent.save_loop("s1", s)
        loops.run_due_loops()
        msgs.extend([{"role": "user", "content": started[-1]}, {"role": "assistant", "content": reply}])
        loops.run_due_loops()
        return agent.load_loop("s1")

    out = loops.run_loop_command("s1", "5m check the deploy")
    with loops._home(None):
        assert out == agent.dispatch_loop_command(agent.LoopManager(session_id="c1"), "5m check the deploy")["output"]
        assert "slash" in loops.run_loop_command("s1", "10m /recap")
        assert tick("still rolling").next_due_at > 0 and started[0].startswith("[/loop wakeup #1, every 5m]")
        assert "1/100 budget" in loops.run_loop_command("s1", "status")
        assert tick("still rolling", max_ticks=2).paused_reason == "tick budget exhausted (2/2)"  # pause, as CLI
        loops.run_loop_command("s1", "resume")
        assert tick("Task cancelled.").paused_reason == "user-interrupted (Stop)"  # CLI Ctrl+C parity
        loops.run_loop_command("s1", "resume")
        assert tick("done\nLOOP_COMPLETE").status == "done" and len(started) == 4
        for cmd in ("5m poll CI --times 2", "5m watch the queue --until queue is empty"):  # CLI flags, CLI text
            loops.run_loop_command("s1", "stop")
            agent.LoopManager(session_id="c1").clear()
            assert loops.run_loop_command("s1", cmd) == agent.dispatch_loop_command(
                agent.LoopManager(session_id="c1"), cmd)["output"]
        monkeypatch.setattr(goals, "judge_goal", lambda goal, reply, **kw: ("done", f"{goal}: yes", False, None, False))
        s = tick("queue drained")
        assert "Stop condition: queue is empty" in started[-1] and s.status == "done"
        assert s.last_stop_reason == "stop condition met: queue is empty: yes"
        loops.run_loop_command("s1", "5m poll CI --times 2")
        assert "0/2 runs" in loops.run_loop_command("s1", "status")
        assert tick("a").status == "active" and tick("b").last_stop_reason == "completed the requested 2 runs"
