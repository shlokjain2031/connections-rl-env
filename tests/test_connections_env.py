import unittest

import numpy as np

from connections_rl import (
    ActionMaskMode,
    ActionMaskToInfoWrapper,
    CategoryLabelVisibility,
    ConnectionsEnv,
    ConnectionsVectorEnv,
    FlattenConnectionsObservation,
    OneAwayMode,
    RewardConfig,
    RewardMode,
    get_preset_env_kwargs,
)
from connections_rl.puzzles import make_puzzle_bank
from connections_rl.registration import register_envs


class ConnectionsEnvTests(unittest.TestCase):
    def test_seeded_reset_is_deterministic(self) -> None:
        env = ConnectionsEnv()
        env.reset(seed=42)
        first = env.current_puzzle_id
        env.reset(seed=42)
        second = env.current_puzzle_id
        self.assertEqual(first, second)

    def test_options_puzzle_id_overrides_sampling(self) -> None:
        env = ConnectionsEnv()
        env.reset(seed=1, options={"puzzle_id": "puzzle-002"})
        self.assertEqual(env.current_puzzle_id, "puzzle-002")

    def test_action_encode_decode_bijection(self) -> None:
        env = ConnectionsEnv()
        env.reset(options={"puzzle_id": "puzzle-001"})
        indices = (0, 1, 2, 3)
        action = env.indices_to_action(indices)
        back = env.action_to_indices(action)
        self.assertEqual(back, indices)

    def test_action_encode_decode_property_sample(self) -> None:
        env = ConnectionsEnv()
        env.reset(options={"puzzle_id": "puzzle-001"})
        rng = np.random.default_rng(123)
        sample = rng.choice(1820, size=128, replace=False)
        for action_idx in sample.tolist():
            indices = env.action_to_indices(int(action_idx))
            roundtrip = env.indices_to_action(indices)
            self.assertEqual(roundtrip, int(action_idx))

    def test_valid_group_solves_group(self) -> None:
        env = ConnectionsEnv(
            reward_config=RewardConfig(
                reward_mode=RewardMode.SHAPED,
                step_penalty=0.0,
                solved_group_reward=1.0,
                win_reward=0.0,
            )
        )
        env.reset(options={"puzzle_id": "puzzle-001"})

        action = env.indices_to_action((0, 1, 2, 3))
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["transition_reason"], "solved_group")
        self.assertEqual(obs["group_solved_mask"][0], 1)
        self.assertEqual(obs["word_solved_mask"][:4].sum(), 4)
        self.assertEqual(obs["attempts_used"].item(), 1)
        self.assertEqual(obs["mistakes_used"].item(), 0)

    def test_wrong_group_increments_mistakes(self) -> None:
        env = ConnectionsEnv(mistake_budget=4)
        env.reset(options={"puzzle_id": "puzzle-001"})

        # One word from each true group -> valid but wrong.
        obs, _, terminated, truncated, info = env.step((0, 4, 8, 12))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["transition_reason"], "wrong_group")
        self.assertEqual(obs["mistakes_used"].item(), 1)
        self.assertEqual(obs["attempts_used"].item(), 1)

    def test_invalid_duplicate_indices_does_not_consume_attempt(self) -> None:
        env = ConnectionsEnv(max_invalid_actions=3)
        env.reset(options={"puzzle_id": "puzzle-001"})

        obs, reward, terminated, truncated, info = env.step((0, 0, 1, 2))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["transition_reason"], "invalid_action")
        self.assertEqual(info["invalid_reason"], "duplicate_word_indices")
        self.assertLess(reward, 0.0)
        self.assertEqual(obs["attempts_used"].item(), 0)
        self.assertEqual(obs["invalid_actions_used"].item(), 1)

    def test_invalid_truncation_budget(self) -> None:
        env = ConnectionsEnv(max_invalid_actions=2)
        env.reset(options={"puzzle_id": "puzzle-001"})

        env.step((0, 0, 1, 2))
        _, _, terminated, truncated, info = env.step((0, 0, 1, 2))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["transition_reason"], "invalid_truncation")

    def test_contains_solved_word_is_invalid(self) -> None:
        env = ConnectionsEnv()
        env.reset(options={"puzzle_id": "puzzle-001"})
        env.step((0, 1, 2, 3))

        obs, _, terminated, truncated, info = env.step((0, 4, 5, 6))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["transition_reason"], "invalid_action")
        self.assertEqual(info["invalid_reason"], "contains_solved_word")
        self.assertEqual(obs["attempts_used"].item(), 1)

    def test_puzzle_solved_terminal_reason(self) -> None:
        env = ConnectionsEnv(
            reward_config=RewardConfig(
                reward_mode=RewardMode.SHAPED,
                step_penalty=0.0,
                solved_group_reward=1.0,
                win_reward=1.0,
            )
        )
        env.reset(options={"puzzle_id": "puzzle-001"})
        env.step((0, 1, 2, 3))
        env.step((4, 5, 6, 7))
        env.step((8, 9, 10, 11))
        _, reward, terminated, truncated, info = env.step((12, 13, 14, 15))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["transition_reason"], "puzzle_solved")
        self.assertEqual(reward, 2.0)

    def test_mistake_budget_terminates_with_reason(self) -> None:
        env = ConnectionsEnv(mistake_budget=2)
        env.reset(options={"puzzle_id": "puzzle-001"})
        env.step((0, 4, 8, 12))
        _, _, terminated, truncated, info = env.step((1, 5, 9, 13))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["transition_reason"], "mistakes_exhausted")

    def test_max_steps_truncation_ignores_invalid_steps(self) -> None:
        env = ConnectionsEnv(max_steps=1, max_invalid_actions=10)
        env.reset(options={"puzzle_id": "puzzle-001"})

        env.step((0, 0, 1, 2))  # invalid, should not count toward max_steps
        _, _, terminated, truncated, info = env.step((0, 4, 8, 12))  # first valid attempt
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["transition_reason"], "max_steps_truncation")

    def test_action_mask_valid_mode(self) -> None:
        env = ConnectionsEnv(action_mask_mode=ActionMaskMode.VALID)
        env.reset(options={"puzzle_id": "puzzle-001"})
        full = env.valid_action_mask()
        self.assertEqual(int(full.sum()), 1820)

        env.step((0, 1, 2, 3))
        after = env.valid_action_mask()
        # After solving one group, legal actions choose 4 from remaining 12 words.
        self.assertEqual(int(after.sum()), 495)

    def test_action_mask_cache_reused_and_reset_clears(self) -> None:
        env = ConnectionsEnv(action_mask_mode=ActionMaskMode.VALID, include_action_mask=False)
        env.reset(options={"puzzle_id": "puzzle-001"})
        self.assertEqual(len(env._mask_cache), 0)

        m1 = env.valid_action_mask()
        self.assertEqual(m1.shape, (1820,))
        self.assertEqual(m1.dtype, np.uint8)
        self.assertEqual(len(env._mask_cache), 1)

        # Same solved-group state and mode should hit cache, not add keys.
        _ = env.valid_action_mask()
        self.assertEqual(len(env._mask_cache), 1)

        # Wrong guess keeps solved-group state unchanged.
        env.step((0, 4, 8, 12))
        _ = env.valid_action_mask()
        self.assertEqual(len(env._mask_cache), 1)

        # Solving a group changes solved-group bitmask and creates one new key.
        env.step((0, 1, 2, 3))
        _ = env.valid_action_mask()
        self.assertEqual(len(env._mask_cache), 2)

        # Reset explicitly clears cache.
        env.reset(options={"puzzle_id": "puzzle-001"})
        self.assertEqual(len(env._mask_cache), 0)

    def test_info_includes_dataset_hash_and_counters(self) -> None:
        env = ConnectionsEnv()
        _, info = env.reset(options={"puzzle_id": "puzzle-001"})
        self.assertIn("dataset_hash", info)
        self.assertIn("dataset_hash_full", info)
        self.assertEqual(len(str(info["dataset_hash"])), 8)

        _, _, _, _, step_info = env.step((0, 4, 8, 12))
        self.assertEqual(step_info["attempts_used"], 1)
        self.assertEqual(step_info["invalid_actions_used"], 0)

    def test_sparse_reward_is_terminal_only(self) -> None:
        env = ConnectionsEnv(
            reward_config=RewardConfig(
                reward_mode=RewardMode.SPARSE,
                win_reward=3.0,
                lose_reward=-2.0,
            )
        )
        env.reset(options={"puzzle_id": "puzzle-001"})

        _, reward1, _, _, _ = env.step((0, 1, 2, 3))
        self.assertEqual(reward1, 0.0)

        env.reset(options={"puzzle_id": "puzzle-001"})
        env.step((0, 1, 2, 3))
        env.step((4, 5, 6, 7))
        env.step((8, 9, 10, 11))
        _, reward2, terminated, _, _ = env.step((12, 13, 14, 15))
        self.assertTrue(terminated)
        self.assertEqual(reward2, 3.0)

    def test_one_away_info_only(self) -> None:
        env = ConnectionsEnv(one_away_mode=OneAwayMode.INFO_ONLY)
        env.reset(options={"puzzle_id": "puzzle-001"})
        _, _, _, _, info = env.step((0, 1, 2, 4))
        self.assertIn("one_away", info)
        self.assertTrue(info["one_away"])

    def test_one_away_disabled_not_exposed(self) -> None:
        env = ConnectionsEnv(one_away_mode=OneAwayMode.DISABLED)
        env.reset(options={"puzzle_id": "puzzle-001"})
        _, _, _, _, info = env.step((0, 1, 2, 4))
        self.assertNotIn("one_away", info)

    def test_one_away_observation_field(self) -> None:
        env = ConnectionsEnv(one_away_mode=OneAwayMode.OBSERVATION)
        obs, _ = env.reset(options={"puzzle_id": "puzzle-001"})
        self.assertIn("one_away", obs)
        self.assertEqual(obs["one_away"].item(), 0)

        obs2, _, _, _, _ = env.step((0, 1, 2, 4))
        self.assertEqual(obs2["one_away"].item(), 1)

    def test_one_away_reward_shaping_replaces_wrong_penalty(self) -> None:
        env = ConnectionsEnv(
            one_away_mode=OneAwayMode.REWARD_SHAPING,
            reward_config=RewardConfig(
                reward_mode=RewardMode.SHAPED,
                step_penalty=0.0,
                wrong_group_penalty=-1.0,
                one_away_bonus=0.25,
            ),
        )
        env.reset(options={"puzzle_id": "puzzle-001"})
        _, reward, _, _, _ = env.step((0, 1, 2, 4))
        self.assertEqual(reward, 0.25)

    def test_category_label_visibility_solved_only(self) -> None:
        env = ConnectionsEnv(category_label_visibility=CategoryLabelVisibility.SOLVED_ONLY)
        obs, _ = env.reset(options={"puzzle_id": "puzzle-001"})
        self.assertTrue((obs["group_labels"] == 27).all())

        obs2, _, _, _, _ = env.step((0, 1, 2, 3))
        self.assertFalse((obs2["group_labels"][0] == 27).all())
        self.assertTrue((obs2["group_labels"][1:] == 27).all())

    def test_category_label_visibility_always(self) -> None:
        env = ConnectionsEnv(category_label_visibility=CategoryLabelVisibility.ALWAYS)
        obs, _ = env.reset(options={"puzzle_id": "puzzle-001"})
        self.assertFalse((obs["group_labels"] == 27).all())

    def test_action_mask_consistent_aliases_valid(self) -> None:
        env_valid = ConnectionsEnv(action_mask_mode=ActionMaskMode.VALID)
        env_consistent = ConnectionsEnv(action_mask_mode=ActionMaskMode.CONSISTENT)
        env_valid.reset(options={"puzzle_id": "puzzle-001"})
        env_consistent.reset(options={"puzzle_id": "puzzle-001"})
        env_valid.step((0, 1, 2, 3))
        env_consistent.step((0, 1, 2, 3))
        self.assertTrue((env_valid.valid_action_mask() == env_consistent.valid_action_mask()).all())

    def test_auto_mask_strictness_mapping(self) -> None:
        env_open = ConnectionsEnv(action_mask_mode=ActionMaskMode.AUTO, mask_strictness="open")
        env_balanced = ConnectionsEnv(action_mask_mode=ActionMaskMode.AUTO, mask_strictness="balanced")
        env_open.reset(options={"puzzle_id": "puzzle-001"})
        env_balanced.reset(options={"puzzle_id": "puzzle-001"})
        env_open.step((0, 1, 2, 3))
        env_balanced.step((0, 1, 2, 3))
        self.assertEqual(int(env_open.valid_action_mask().sum()), 1820)
        self.assertEqual(int(env_balanced.valid_action_mask().sum()), 495)

    def test_observation_dtypes_and_shapes_stable(self) -> None:
        env = ConnectionsEnv(
            one_away_mode=OneAwayMode.OBSERVATION,
            include_action_mask=True,
            category_label_visibility=CategoryLabelVisibility.SOLVED_ONLY,
        )
        obs, _ = env.reset(options={"puzzle_id": "puzzle-001"})
        self.assertEqual(obs["words"].shape, (16, env.max_word_length))
        self.assertEqual(obs["group_labels"].shape, (4, env.max_label_length))
        self.assertEqual(obs["words"].dtype, np.uint8)
        self.assertEqual(obs["group_labels"].dtype, np.uint8)
        self.assertEqual(obs["attempts_used"].dtype, np.int32)
        self.assertEqual(obs["action_mask"].dtype, np.uint8)

        obs2, _, _, _, _ = env.step((0, 1, 2, 4))
        self.assertEqual(obs2["words"].shape, (16, env.max_word_length))
        self.assertEqual(obs2["group_labels"].shape, (4, env.max_label_length))
        self.assertEqual(obs2["one_away"].dtype, np.uint8)

    def test_vector_env_matches_single_env_semantics(self) -> None:
        vec = ConnectionsVectorEnv(
            num_envs=2,
            auto_reset=False,
            include_action_mask=False,
            max_invalid_actions=2,
        )
        obs, infos = vec.reset(options=[{"puzzle_id": "puzzle-001"}, {"puzzle_id": "puzzle-001"}], seed=7)
        self.assertEqual(obs["words"].shape[0], 2)
        self.assertEqual(len(infos), 2)

        next_obs, rewards, terminated, truncated, step_infos = vec.step([(0, 1, 2, 3), (0, 4, 8, 12)])
        self.assertEqual(next_obs["mistakes_used"][0].item(), 0)
        self.assertEqual(next_obs["mistakes_used"][1].item(), 1)
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(terminated.shape, (2,))
        self.assertEqual(truncated.shape, (2,))
        self.assertEqual(len(step_infos), 2)
        vec.close()

    def test_wrappers_work_when_gym_available(self) -> None:
        if ActionMaskToInfoWrapper is None or FlattenConnectionsObservation is None:
            self.skipTest("gymnasium wrappers unavailable")
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            self.skipTest("gymnasium not installed")

        base = ConnectionsEnv(include_action_mask=True)
        wrapped = ActionMaskToInfoWrapper(base)
        _, info = wrapped.reset(options={"puzzle_id": "puzzle-001"})
        self.assertIn("action_mask", info)

        flat = FlattenConnectionsObservation(ConnectionsEnv(include_action_mask=False))
        flat_obs, _ = flat.reset(options={"puzzle_id": "puzzle-001"})
        self.assertEqual(len(flat_obs.shape), 1)

    def test_registration_and_make_when_gym_available(self) -> None:
        try:
            import gymnasium as gym
        except ModuleNotFoundError:
            self.skipTest("gymnasium not installed")

        register_envs(force=True)
        env = gym.make("ConnectionsRL-v0")
        obs, _ = env.reset(seed=0)
        self.assertIn("words", obs)
        env.close()

    def test_gym_checker_passes_when_available(self) -> None:
        try:
            from gymnasium.utils.env_checker import check_env
        except ModuleNotFoundError:
            self.skipTest("gymnasium not installed")

        env = ConnectionsEnv()
        check_env(env)

    def test_schema_validation_rejects_invalid_puzzles(self) -> None:
        with self.assertRaises(ValueError):
            make_puzzle_bank(
                [
                    {
                        "id": "bad-1",
                        "groups": [
                            {"label": "A", "words": ["a", "b", "c", "d"]},
                            {"label": "B", "words": ["e", "f", "g", "h"]},
                            {"label": "C", "words": ["i", "j", "k", "l"]},
                        ],
                    }
                ]
            )

        with self.assertRaises(ValueError):
            make_puzzle_bank(
                [
                    {
                        "id": "dup",
                        "groups": [
                            {"label": "A", "words": ["a", "b", "c", "d"]},
                            {"label": "B", "words": ["e", "f", "g", "h"]},
                            {"label": "C", "words": ["i", "j", "k", "l"]},
                            {"label": "D", "words": ["m", "n", "o", "p"]},
                        ],
                    },
                    {
                        "id": "dup",
                        "groups": [
                            {"label": "E", "words": ["q", "r", "s", "t"]},
                            {"label": "F", "words": ["u", "v", "w", "x"]},
                            {"label": "G", "words": ["y", "z", "aa", "bb"]},
                            {"label": "H", "words": ["cc", "dd", "ee", "ff"]},
                        ],
                    },
                ]
            )

    def test_presets_can_construct_env(self) -> None:
        for preset_name in [
            "research_balanced",
            "research_strict",
            "benchmark_sparse",
            "benchmark_open",
        ]:
            kwargs = get_preset_env_kwargs(preset_name)
            env = ConnectionsEnv(**kwargs)
            obs, _ = env.reset(options={"puzzle_id": "puzzle-001"})
            self.assertIn("words", obs)

        with self.assertRaises(ValueError):
            make_puzzle_bank(
                [
                    {
                        "id": "bad-2",
                        "groups": [
                            {"label": "A", "words": ["a", "b", "c", "d"]},
                            {"label": "B", "words": ["e", "f", "g", "h"]},
                            {"label": "C", "words": ["i", "j", "k", "l"]},
                            {"label": "D", "words": ["m", "n", "o", "a"]},
                        ],
                    }
                ]
            )

    def test_golden_trajectory_puzzle_001(self) -> None:
        env = ConnectionsEnv(
            one_away_mode=OneAwayMode.INFO_ONLY,
            reward_config=RewardConfig(
                reward_mode=RewardMode.SHAPED,
                step_penalty=0.0,
                solved_group_reward=1.0,
                wrong_group_penalty=-0.2,
                one_away_bonus=-0.05,
                invalid_action_penalty=-0.5,
                win_reward=2.0,
                lose_reward=-1.0,
            ),
        )
        obs0, _ = env.reset(options={"puzzle_id": "puzzle-001"}, seed=123)
        self.assertEqual(obs0["groups_solved_count"].item(), 0)

        sequence = [
            ((0, 1, 2, 4), "wrong_group", -0.2, 0, 1, True),
            ((0, 1, 2, 3), "solved_group", 1.0, 1, 1, False),
            ((0, 4, 5, 6), "invalid_action", -0.5, 1, 1, False),
            ((4, 5, 6, 7), "solved_group", 1.0, 2, 1, False),
            ((8, 9, 10, 11), "solved_group", 1.0, 3, 1, False),
            ((12, 13, 14, 15), "puzzle_solved", 3.0, 4, 1, False),
        ]

        for action, reason, reward, solved_count, mistakes, one_away in sequence:
            obs, got_reward, terminated, truncated, info = env.step(action)
            self.assertEqual(info["transition_reason"], reason)
            self.assertAlmostEqual(got_reward, reward)
            self.assertEqual(obs["groups_solved_count"].item(), solved_count)
            self.assertEqual(obs["mistakes_used"].item(), mistakes)
            if reason == "invalid_action":
                self.assertFalse(terminated)
                self.assertFalse(truncated)
            if "one_away" in info:
                self.assertEqual(bool(info["one_away"]), bool(one_away))

    def test_golden_trajectory_puzzle_002_sparse(self) -> None:
        env = ConnectionsEnv(
            reward_config=RewardConfig(
                reward_mode=RewardMode.SPARSE,
                win_reward=1.0,
                lose_reward=-1.0,
                invalid_action_penalty=-0.3,
            )
        )
        env.reset(options={"puzzle_id": "puzzle-002"}, seed=7)

        actions = [(0, 4, 8, 12), (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)]
        expected = ["wrong_group", "solved_group", "solved_group", "solved_group", "puzzle_solved"]
        rewards = []
        reasons = []
        for action in actions:
            _, reward, _, _, info = env.step(action)
            rewards.append(reward)
            reasons.append(info["transition_reason"])

        self.assertEqual(reasons, expected)
        self.assertEqual(rewards[:-1], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(rewards[-1], 1.0)

    def test_golden_trajectory_puzzle_003_mistake_exhausted(self) -> None:
        env = ConnectionsEnv(
            mistake_budget=2,
            reward_config=RewardConfig(
                reward_mode=RewardMode.SHAPED,
                step_penalty=0.0,
                wrong_group_penalty=-0.4,
                lose_reward=-1.0,
            ),
        )
        env.reset(options={"puzzle_id": "puzzle-003"}, seed=11)
        _, reward1, term1, trunc1, info1 = env.step((0, 4, 8, 12))
        _, reward2, term2, trunc2, info2 = env.step((1, 5, 9, 13))

        self.assertEqual(info1["transition_reason"], "wrong_group")
        self.assertEqual(reward1, -0.4)
        self.assertFalse(term1)
        self.assertFalse(trunc1)

        self.assertEqual(info2["transition_reason"], "mistakes_exhausted")
        self.assertEqual(reward2, -1.4)
        self.assertTrue(term2)
        self.assertFalse(trunc2)


if __name__ == "__main__":
    unittest.main()
