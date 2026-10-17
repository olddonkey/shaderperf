import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from shaderperf.llm import CodexChat, build_codex_argv, parse_codex_jsonl


def jsonl(*events):
    return "\n".join(json.dumps(event) for event in events)


class CodexJsonlTests(unittest.TestCase):
    def test_normal_flow_uses_last_agent_message_and_records_usage(self):
        parsed = parse_codex_jsonl(
            jsonl(
                {"type": "thread.started", "thread_id": "thread-123"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "draft"},
                },
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "final"},
                },
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 19018,
                        "cached_input_tokens": 12,
                        "output_tokens": 5,
                        "reasoning_output_tokens": 3,
                    },
                },
            )
        )

        self.assertEqual(parsed["thread_id"], "thread-123")
        self.assertEqual(parsed["text"], "final")
        self.assertEqual(parsed["input_tokens"], 19018)
        self.assertEqual(parsed["cached_input_tokens"], 12)
        self.assertEqual(parsed["output_tokens"], 5)
        self.assertEqual(parsed["reasoning_output_tokens"], 3)
        self.assertEqual(parsed["warnings"], [])

    def test_nonfatal_error_items_are_warnings(self):
        parsed = parse_codex_jsonl(
            jsonl(
                {
                    "type": "item.completed",
                    "item": {"type": "error", "message": "feature warning"},
                },
                {"type": "turn.completed", "usage": {}},
            )
        )

        self.assertEqual(parsed["warnings"], ["feature warning"])

    def test_missing_turn_completed_uses_zero_usage_and_warns(self):
        parsed = parse_codex_jsonl(
            'not json\n{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}'
        )

        self.assertEqual(parsed["input_tokens"], 0)
        self.assertEqual(parsed["output_tokens"], 0)
        self.assertIn("missing turn.completed", parsed["warnings"][0])


class CodexArgvTests(unittest.TestCase):
    def test_first_turn_has_all_pinned_flags(self):
        argv = build_codex_argv(
            model="gpt-5.6-sol",
            effort="low",
            workspace_dir="/tmp/codex-scratch",
            prompt="hello",
        )

        self.assertEqual(argv[:2], ["codex", "exec"])
        self.assertIn("--json", argv)
        self.assertIn("--skip-git-repo-check", argv)
        self.assertEqual(argv[argv.index("-s") + 1], "read-only")
        self.assertEqual(argv[argv.index("-C") + 1], "/tmp/codex-scratch")
        self.assertEqual(argv[argv.index("-m") + 1], "gpt-5.6-sol")
        self.assertIn('model_reasoning_effort="low"', argv)
        self.assertIn("mcp_servers={}", argv)
        self.assertEqual(argv[-1], "hello")

    def test_resume_has_thread_id_and_no_workspace_flag(self):
        argv = build_codex_argv(
            model="gpt-5.6-sol",
            effort="low",
            workspace_dir="/tmp/codex-scratch",
            prompt="feedback",
            thread_id="thread-123",
        )

        self.assertEqual(argv[:4], ["codex", "exec", "resume", "thread-123"])
        self.assertIn("--json", argv)
        self.assertIn("--skip-git-repo-check", argv)
        self.assertNotIn("-s", argv)
        self.assertIn('sandbox_mode="read-only"', argv)
        self.assertEqual(argv[argv.index('sandbox_mode="read-only"') - 1], "-c")
        self.assertNotIn("-C", argv)
        self.assertEqual(argv[-1], "feedback")


class CodexStateTests(unittest.TestCase):
    def test_first_prompt_then_resume_uses_only_last_message(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name) / "scratch"
        outputs = [
            jsonl(
                {"type": "thread.started", "thread_id": "thread-123"},
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "answer one"},
                },
                {"type": "turn.completed", "usage": {}},
            ),
            jsonl(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "answer two"},
                },
                {"type": "turn.completed", "usage": {}},
            ),
        ]
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, outputs.pop(0), "noise")

        client = CodexChat("gpt-5.6-sol", "low", workspace)
        client._run = fake_run
        messages = [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "first user text"},
        ]

        first = client.complete(messages, 100)
        messages.extend(
            [
                {"role": "assistant", "content": first["text"]},
                {"role": "user", "content": "feedback only"},
            ]
        )
        second = client.complete(messages, 50)

        self.assertEqual(calls[0][0][-1], "system text\n\nfirst user text")
        self.assertEqual(calls[1][0][-1], "feedback only")
        self.assertEqual(calls[1][0][2:4], ["resume", "thread-123"])
        self.assertEqual(calls[0][1]["cwd"], workspace.resolve())
        self.assertEqual(second["thread_id"], "thread-123")
        self.assertEqual(client.max_output_tokens_requests, [100, 50])


if __name__ == "__main__":
    unittest.main()
