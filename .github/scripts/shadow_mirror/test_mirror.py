import base64
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from mirror import (
    API,
    FORK,
    HOLD,
    HOLD_MARKER,
    SOURCE,
    WORKFLOW,
    APIError,
    Mirror,
    is_dependency_pr,
    mirror_revision,
    recording_workflow,
    scrub_title,
)


class DiscoveryTest(unittest.TestCase):
    def test_manifest_and_lockfile_only_changes(self):
        paths = (
            "requirements-dev.txt",
            "pyproject.toml",
            "uv.lock",
            "enterprise/poetry.lock",
            "ui/litellm-dashboard/package.json",
            "litellm/ui/litellm-dashboard/package-lock.json",
        )
        pr = {
            "changed_files": len(paths),
            "user": {"login": "developer", "type": "User"},
        }
        files = [{"filename": path} for path in paths]
        self.assertTrue(is_dependency_pr(pr, files))
        self.assertFalse(is_dependency_pr(pr, files[:-1]))
        files[0]["previous_filename"] = "script.py"
        self.assertFalse(is_dependency_pr(pr, files))

    def test_named_apps_can_make_mixed_changes(self):
        files = [{"filename": "README.md"}]
        for login, type_, qualified in (
            ("dependabot[bot]", "Bot", True),
            ("renovate[bot]", "Bot", True),
            ("renovate[bot]", "User", False),
            ("another[bot]", "Bot", False),
        ):
            with self.subTest(login=login, type=type_):
                pr = {"changed_files": 1, "user": {"login": login, "type": type_}}
                self.assertEqual(is_dependency_pr(pr, files), qualified)

    def test_candidate_pages_keep_bots_and_verify_manifests_later(self):
        api = API("unused", True)

        def page(nodes, cursor=None):
            return {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "nodes": nodes,
                            "pageInfo": {
                                "hasNextPage": bool(cursor),
                                "endCursor": cursor,
                            },
                        }
                    }
                }
            }

        def node(number, path, author=None):
            return {
                "number": number,
                "author": {"login": author} if author else None,
                "files": {"nodes": [{"path": path}]},
            }

        api._request = Mock(
            side_effect=[
                page(
                    [node(1, "README.md"), node(2, "README.md", "renovate[bot]")],
                    "second",
                ),
                page([node(3, "pyproject.toml")]),
            ]
        )
        self.assertEqual(list(api.dependency_candidates()), [2, 3])
        self.assertEqual(
            api._request.call_args.args[-1]["variables"], {"cursor": "second"}
        )

    def test_upstream_and_dry_run_writes_are_blocked(self):
        for api, repo in ((API("unused", False), SOURCE), (API("unused", True), FORK)):
            with self.assertRaisesRegex(ValueError, "live writes to the mirror"):
                api.call(repo, "pulls", "POST", {})

    def test_scrub_title_removes_upstream_references(self):
        self.assertEqual(
            scrub_title(
                f"Bump @scope/name [skip ci] {SOURCE}#123 @person https://github.com/{SOURCE}/pull/123"
            ),
            "Bump @scope/name",
        )

    def test_mirror_revision_requires_exact_same_repo_pair(self):
        pr = {
            "base": {"repo": {"full_name": FORK}, "ref": "shadow/base/1-r2"},
            "head": {"repo": {"full_name": FORK}, "ref": "shadow/pr/1-r2"},
        }
        self.assertEqual(mirror_revision(pr, 1), 2)
        self.assertIsNone(mirror_revision(pr, 2))
        self.assertIsNone(
            mirror_revision({**pr, "head": {**pr["head"], "ref": "shadow/pr/1"}}, 1)
        )
        pr["head"]["repo"]["full_name"] = SOURCE
        self.assertIsNone(mirror_revision(pr, 1))


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.api = Mock(dry_run=False)
        self.api.pages.return_value = []
        self.mirror = Mirror(self.api)
        self.api.reset_mock()
        self.source_base = "a" * 40
        self.merge_base = "b" * 40
        self.source_head = "c" * 40
        self.pr = {
            "state": "open",
            "title": "Bump fastapi",
            "changed_files": 1,
            "user": {"login": "developer", "type": "User"},
            "base": {"sha": self.source_base},
            "head": {"sha": self.source_head},
        }
        self.files = [{"filename": "uv.lock"}]
        self.workflow_path = (
            Path(__file__).resolve().parents[2] / "workflows" / WORKFLOW
        )

    def test_baseline_uses_merge_base_and_head_uses_original_head(self):
        def read(repo, path):
            if path == "pulls/1":
                return self.pr
            if path == f"compare/{self.source_base}...{self.source_head}":
                return {"merge_base_commit": {"sha": self.merge_base}}
            if path.startswith("contents/"):
                return {
                    "content": base64.b64encode(
                        self.workflow_path.read_bytes()
                    ).decode()
                }
            if path.startswith("git/ref/"):
                raise APIError("GET", path, 404, "missing ref")
            if path.startswith("actions/"):
                return {"workflow_runs": [{"id": 1}]}
            raise AssertionError((repo, path))

        self.api.call.side_effect = read
        self.files[0]["filename"] = "<!-- garnet:record-summary:opt-in -->/package.json"
        self.api.pages.side_effect = [self.files, []]
        self.api.write.side_effect = [
            {"sha": "baseline"},
            {},
            {"sha": "head"},
            {},
            {"number": 2, "html_url": "https://github.com/garnet-labs/litellm/pull/2"},
            {},
        ]
        with patch.object(
            self.mirror, "tree", side_effect=["base-tree", "head-tree"]
        ) as tree:
            self.assertEqual(self.mirror.sync(1)["status"], "created")
        self.assertEqual(
            tree.call_args_list, [call(self.merge_base), call(self.source_head)]
        )
        writes = self.api.write.call_args_list
        self.assertEqual(
            [write.args[0] for write in writes],
            [
                "git/commits",
                "git/refs",
                "git/commits",
                "git/refs",
                "pulls",
                "issues/2/comments",
            ],
        )
        self.assertEqual(writes[0].args[1]["parents"], [])
        self.assertEqual(writes[2].args[1]["parents"], ["baseline"])
        self.assertEqual(writes[4].args[1]["base"], "shadow/base/1")
        self.assertEqual(writes[4].args[1]["head"], "shadow/pr/1")
        self.assertNotIn(
            "<!-- garnet:record-summary:opt-in -->", writes[4].args[1]["body"]
        )
        self.assertEqual(writes[5], call("issues/2/comments", {"body": HOLD}))

    def test_changed_base_creates_new_pair_without_updating_open_refs(self):
        old = {"state": "open", "number": 5, "body": "original"}
        self.mirror.workflow = "trusted workflow"
        self.api.call.return_value = {"message": "old baseline"}
        self.api.write.side_effect = [
            {"sha": "new-base"},
            {},
            {"sha": "new-head"},
            {},
            {"number": 6, "html_url": "new mirror"},
            {},
            {},
        ]
        with (
            patch.object(
                self.mirror, "get_ref", side_effect=["old-baseline", None, None]
            ),
            patch.object(self.mirror, "tree", side_effect=["base-tree", "head-tree"]),
            patch.object(self.mirror, "baseline_started", return_value=True),
        ):
            self.assertEqual(
                self.mirror.replay(1, self.pr, self.files, self.merge_base, {1: old})[
                    "status"
                ],
                "created",
            )
        writes = self.api.write.call_args_list
        self.assertEqual(
            writes[1],
            call("git/refs", {"ref": "refs/heads/shadow/base/1-r2", "sha": "new-base"}),
        )
        self.assertEqual(
            writes[3],
            call("git/refs", {"ref": "refs/heads/shadow/pr/1-r2", "sha": "new-head"}),
        )
        self.assertEqual(writes[-1], call("pulls/5", {"state": "closed"}, "PATCH"))

    def test_dry_run_never_writes(self):
        self.api.dry_run = True
        self.api.call.side_effect = [
            self.pr,
            self.pr,
            {"merge_base_commit": {"sha": self.merge_base}},
        ]
        self.api.pages.return_value = self.files
        self.assertEqual(
            self.mirror.sync(1),
            {
                "status": "dry-run",
                "upstream_base": self.merge_base,
                "upstream_head": self.source_head,
                "files": ["uv.lock"],
            },
        )
        self.api.write.assert_not_called()

    def test_replaced_tree_has_only_the_recorder_in_workflows(self):
        self.mirror.workflow = "trusted workflow"
        self.api.call.side_effect = [
            {"tree": {"sha": "original-root"}},
            {"tree": [{"path": ".github", "type": "tree", "sha": "original-github"}]},
        ]
        self.api.write.side_effect = [
            {"sha": "workflows"},
            {"sha": "github"},
            {"sha": "root"},
        ]
        self.assertEqual(self.mirror.tree(self.merge_base), "root")
        self.assertEqual(
            self.api.write.call_args_list,
            [
                call(
                    "git/trees",
                    {
                        "tree": [
                            {
                                "path": WORKFLOW,
                                "mode": "100644",
                                "type": "blob",
                                "content": "trusted workflow",
                            }
                        ]
                    },
                ),
                call(
                    "git/trees",
                    {
                        "tree": [
                            {
                                "path": "workflows",
                                "mode": "040000",
                                "type": "tree",
                                "sha": "workflows",
                            }
                        ],
                        "base_tree": "original-github",
                    },
                ),
                call(
                    "git/trees",
                    {
                        "base_tree": "original-root",
                        "tree": [
                            {
                                "path": ".github",
                                "mode": "040000",
                                "type": "tree",
                                "sha": "github",
                            }
                        ],
                    },
                ),
            ],
        )

    def test_recorder_runs_on_base_and_removes_persisted_checkout_credentials(self):
        workflow = recording_workflow(self.workflow_path.read_text())
        self.assertIn("'shadow/base/**'", workflow)
        self.assertIn("  workflow_dispatch:", workflow)
        self.assertIn("  pull_request:", workflow)
        for job in ("python-install", "dashboard-install", "extension-install"):
            self.assertIn(f"  {job}:", workflow)
        self.assertEqual(workflow.count("persist-credentials: false"), 3)

    def test_baseline_can_be_dispatched_if_ref_creation_does_not_trigger_it(self):
        self.api.call.side_effect = [{"workflow_runs": []}] * 6 + [
            {"workflow_runs": [{"id": 1}]}
        ]
        with patch("mirror.time.sleep"):
            self.assertTrue(self.mirror.baseline_started("shadow/base/1", "base-sha"))
        self.api.write.assert_called_once_with(
            f"actions/workflows/{WORKFLOW}/dispatches", {"ref": "shadow/base/1"}
        )

    def test_gate_updates_its_comment_without_changing_other_comments(self):
        self.api.pages.return_value = [
            {"id": 1, "body": "unrelated"},
            {"id": 2, "body": f"{HOLD_MARKER}\nold gate text"},
        ]
        self.mirror.hold(3)
        self.api.pages.assert_called_once_with(FORK, "issues/3/comments")
        self.api.write.assert_called_once_with(
            "issues/comments/2", {"body": HOLD}, "PATCH"
        )
        self.api.reset_mock()
        self.api.pages.return_value = [{"id": 2, "body": HOLD}]
        self.mirror.hold(3)
        self.api.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
