import json
import tempfile
import unittest
from pathlib import Path

from shaderperf.sealed import (
    commit_task,
    load_sealed_times,
    reveal_task,
    sealed_times,
    verify_commitment,
)


class SealedTimesTests(unittest.TestCase):
    def test_same_inputs_produce_same_times(self):
        first = sealed_times("secret", "task-a", 12, (0.0, 10.0))
        second = sealed_times("secret", "task-a", 12, (0.0, 10.0))

        self.assertEqual(first, second)

    def test_exactly_one_time_lands_in_each_stratum(self):
        n = 12
        t0, t1 = 0.0, 10.0
        width = (t1 - t0) / n
        times = sealed_times("secret", "task-a", n, (t0, t1))

        self.assertEqual(len(times), n)
        for index, value in enumerate(times):
            self.assertGreaterEqual(value, t0 + index * width)
            self.assertLess(value, t0 + (index + 1) * width)

    def test_different_task_or_seed_produces_different_times(self):
        original = sealed_times("secret", "task-a", 12, (0.0, 10.0))

        self.assertNotEqual(
            original, sealed_times("secret", "task-b", 12, (0.0, 10.0))
        )
        self.assertNotEqual(
            original, sealed_times("other", "task-a", 12, (0.0, 10.0))
        )


class SealedStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _read(self, name):
        return json.loads((self.root / "sealed" / name).read_text())

    def test_commit_writes_public_record_and_private_secret(self):
        record = commit_task(
            "task-a", secret_seed="secret-value", root=self.root
        )

        commitments = self._read("commitments.json")
        stored_secrets = self._read("secrets.json")
        self.assertEqual(commitments["task-a"], record)
        self.assertIsNone(record["revealed_seed"])
        self.assertNotIn("secret-value", json.dumps(record))
        self.assertEqual(stored_secrets, {"task-a": "secret-value"})

    def test_load_matches_direct_generation(self):
        commit_task("task-a", secret_seed="secret-value", root=self.root)

        self.assertEqual(
            load_sealed_times("task-a", root=self.root),
            sealed_times("secret-value", "task-a", 12, (0.0, 10.0)),
        )

    def test_tampered_public_commitment_makes_load_fail(self):
        commit_task("task-a", secret_seed="secret-value", root=self.root)
        commitments = self._read("commitments.json")
        commitments["task-a"]["commitment"] = "0" * 64
        (self.root / "sealed" / "commitments.json").write_text(
            json.dumps(commitments)
        )

        with self.assertRaisesRegex(ValueError, "commitment mismatch"):
            load_sealed_times("task-a", root=self.root)

    def test_reveal_publishes_seed_and_verifies(self):
        record = commit_task(
            "task-a", secret_seed="secret-value", root=self.root
        )

        reveal = reveal_task("task-a", root=self.root)

        commitments = self._read("commitments.json")
        self.assertEqual(commitments["task-a"]["revealed_seed"], "secret-value")
        self.assertEqual(reveal["seed"], "secret-value")
        self.assertEqual(reveal["commitment"], record["commitment"])
        self.assertEqual(reveal["times"], load_sealed_times("task-a", self.root))
        self.assertTrue(verify_commitment("task-a", "secret-value", self.root))

    def test_wrong_seed_does_not_verify(self):
        commit_task("task-a", secret_seed="secret-value", root=self.root)

        self.assertFalse(verify_commitment("task-a", "wrong", self.root))

    def test_commit_refuses_to_overwrite_without_force(self):
        original = commit_task(
            "task-a", secret_seed="secret-value", root=self.root
        )

        with self.assertRaisesRegex(ValueError, "already exists"):
            commit_task("task-a", secret_seed="other", root=self.root)

        self.assertEqual(self._read("commitments.json")["task-a"], original)
        self.assertEqual(
            self._read("secrets.json")["task-a"], "secret-value"
        )


if __name__ == "__main__":
    unittest.main()
