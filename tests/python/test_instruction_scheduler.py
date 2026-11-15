import unittest


class InstructionFineTuningSchedulerTests(unittest.TestCase):
    def test_stage_schedule_uses_the_requested_optimizer_step_budget(self):
        from syntra_ai.instruction import _finetuning_scheduler_settings
        from syntra_ai.training import _scheduled_learning_rate

        learning_rate = 1e-4
        cases = (
            (1, 1, 2),
            (20, 2, 20),
            (250, 25, 250),
            (1000, 100, 1000),
        )
        for steps, expected_warmup, expected_decay in cases:
            with self.subTest(steps=steps):
                settings = _finetuning_scheduler_settings(steps, learning_rate)
                self.assertEqual(settings["steps"], steps)
                self.assertEqual(settings["warmup_steps"], expected_warmup)
                self.assertEqual(settings["decay_steps"], expected_decay)
                self.assertEqual(settings["scheduler"], "stage-cosine-warmup-v1")
                self.assertEqual(
                    settings["optimizer_scope"],
                    "fresh-per-finetune-invocation",
                )

                first_rate = _scheduled_learning_rate(
                    learning_rate,
                    1,
                    minimum_learning_rate=settings["minimum_learning_rate"],
                    warmup_steps=settings["warmup_steps"],
                    decay_steps=settings["decay_steps"],
                )
                final_rate = _scheduled_learning_rate(
                    learning_rate,
                    steps,
                    minimum_learning_rate=settings["minimum_learning_rate"],
                    warmup_steps=settings["warmup_steps"],
                    decay_steps=settings["decay_steps"],
                )
                self.assertGreaterEqual(first_rate, settings["minimum_learning_rate"])
                self.assertLessEqual(first_rate, learning_rate)
                self.assertAlmostEqual(final_rate, settings["minimum_learning_rate"])

    def test_one_step_sft_stage_uses_full_base_rate_without_fake_decay(self):
        from syntra_ai.instruction import _finetuning_scheduler_settings
        from syntra_ai.training import _scheduled_learning_rate

        settings = _finetuning_scheduler_settings(1, 1e-4)
        actual = _scheduled_learning_rate(
            1e-4,
            1,
            minimum_learning_rate=settings["minimum_learning_rate"],
            warmup_steps=settings["warmup_steps"],
            decay_steps=settings["decay_steps"],
        )
        self.assertAlmostEqual(actual, 1e-4)

    def test_stage_schedule_rejects_invalid_step_and_learning_rate_values(self):
        from syntra_ai.instruction import _finetuning_scheduler_settings

        for steps in (0, -1, True, 1.5, "10"):
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    _finetuning_scheduler_settings(steps, 1e-4)
        for learning_rate in (0, -1e-4, True, float("nan"), float("inf"), "1e-4"):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaises(ValueError):
                    _finetuning_scheduler_settings(10, learning_rate)

    def test_floor_never_exceeds_a_small_configured_base_rate(self):
        from syntra_ai.instruction import _finetuning_scheduler_settings
        from syntra_ai.training import _scheduled_learning_rate

        settings = _finetuning_scheduler_settings(100, 1e-6)
        self.assertEqual(settings["minimum_learning_rate"], 1e-6)
        self.assertAlmostEqual(
            _scheduled_learning_rate(
                1e-6,
                100,
                minimum_learning_rate=settings["minimum_learning_rate"],
                warmup_steps=settings["warmup_steps"],
                decay_steps=settings["decay_steps"],
            ),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
