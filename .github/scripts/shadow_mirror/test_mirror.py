import base64
import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch

from mirror import (
    API,
    FORK,
    GATE_MARKER,
    HOLD,
    HOLD_MARKER,
    MARK_READY,
    SOURCE,
    WORKFLOW,
    APIError,
    Mirror,
    is_dependency_pr,
    main,
    mirror_revision,
    publish_verdict,
    recording_workflow,
    scrub_title,
)


class DiscoveryTest(unittest.TestCase):
    def test_manifest_and_lockfile_only_changes(self):
        paths = (
            "pyproject.toml",
            "uv.lock",
            "poetry.lock",
            "ui/litellm-dashboard/package.json",
            "litellm/ui/litellm-dashboard/package-lock.json",
            "vscode-extension/package-lock.json",
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
        files = [{"filename": "README.md"}, {"filename": "uv.lock"}]
        for login, type_, qualified in (
            ("dependabot[bot]", "Bot", True),
            ("renovate[bot]", "Bot", True),
            ("renovate[bot]", "User", False),
            ("another[bot]", "Bot", False),
        ):
            with self.subTest(login=login, type=type_):
                pr = {"changed_files": 2, "user": {"login": login, "type": type_}}
                self.assertEqual(is_dependency_pr(pr, files), qualified)

    def test_unrecorded_inputs_do_not_qualify_for_bot_or_human(self):
        for path in (
            "docs/my-website/package-lock.json",
            "enterprise/poetry.lock",
            "requirements-dev.txt",
            "ui/litellm-dashboard/yarn.lock",
        ):
            for user in (
                {"login": "dependabot[bot]", "type": "Bot"},
                {"login": "developer", "type": "User"},
            ):
                with self.subTest(path=path, user=user):
                    pr = {"changed_files": 1, "user": user}
                    self.assertFalse(is_dependency_pr(pr, [{"filename": path}]))

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

    def test_ready_mutation_only_accepts_live_mirror_pairs(self):
        pr = {
            "node_id": "PR_node",
            "base": {"ref": "shadow/base/1", "repo": {"full_name": FORK}},
            "head": {"ref": "shadow/pr/1", "repo": {"full_name": FORK}},
        }
        api = API("unused", True)
        api._request = Mock()
        with self.assertRaisesRegex(ValueError, "live writes to the mirror"):
            api.mark_ready(pr, 1)
        api.dry_run = False
        with self.assertRaisesRegex(ValueError, "live writes to the mirror"):
            api.mark_ready(
                {**pr, "head": {**pr["head"], "repo": {"full_name": SOURCE}}}, 1
            )
        api._request.assert_not_called()
        api._request.return_value = {
            "data": {
                "markPullRequestReadyForReview": {"pullRequest": {"isDraft": False}}
            }
        }
        api.mark_ready(pr, 1)
        api._request.assert_called_once_with(
            "graphql", "POST", {"query": MARK_READY, "variables": {"id": "PR_node"}}
        )
        api._request.return_value = {"errors": [{"message": "permission denied"}]}
        with self.assertRaisesRegex(ValueError, "permission denied"):
            api.mark_ready(pr, 1)

    def test_scrub_title_removes_upstream_references(self):
        self.assertEqual(
            scrub_title(
                f"Bump @scope/name [skip ci] {SOURCE}#123 @person https://github.com/{SOURCE}/pull/123"
            ),
            "Bump @scope/name",
        )

    def test_scrub_title_cannot_recreate_ci_skip_directives(self):
        for directive in (
            "[skip  ci]",
            "[ci\tskip]",
            "[No\nCI]",
            "[skip @user actions]",
            "[actions [skip ci] skip]",
        ):
            with self.subTest(directive=directive):
                self.assertEqual(
                    scrub_title(f"Bump fastapi {directive}"), "Bump fastapi"
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
        self.pr.update(
            changed_files=2, user={"login": "dependabot[bot]", "type": "Bot"}
        )
        self.files.append(
            {"filename": "<!-- garnet:record-summary:opt-in -->/package.json"}
        )
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
        self.assertIs(writes[4].args[1]["draft"], True)
        self.assertNotIn(
            "<!-- garnet:record-summary:opt-in -->", writes[4].args[1]["body"]
        )
        self.assertEqual(writes[5], call("issues/2/comments", {"body": HOLD}))
        self.api.mark_ready.assert_not_called()

    def test_later_sync_promotes_only_with_trusted_finalized_current_head(self):
        head = "d" * 40
        existing = {
            "number": 2,
            "node_id": "PR_node",
            "html_url": "mirror",
            "draft": True,
            "state": "open",
            "base": {"ref": "shadow/base/1", "repo": {"full_name": FORK}},
            "head": {"ref": "shadow/pr/1", "repo": {"full_name": FORK}, "sha": head},
            "body": (
                f"<!-- garnet-shadow-mirror upstream=1 upstream_head={self.source_head} "
                f"upstream_base={self.merge_base} -->"
            ),
        }

        def evidence(
            login="garnet-runtime-review[bot]",
            head_marker=head,
            summary_head=head,
            status="finalized",
            owned=True,
        ):
            summary = json.dumps({"commit": summary_head, "status": status})
            body = [
                "<!-- garnet-runtime-review -->",
                f"<!-- garnet:commit {head_marker} -->",
                f"<!-- garnet:summary {summary} -->",
            ]
            if owned:
                body.append("<!-- garnet-control-plane-pr-comment:v1:app.garnet.ai -->")
            return {"user": {"login": login}, "body": "\n".join(body)}

        malformed = {
            **evidence(),
            "body": evidence()["body"].replace('"status"', "status"),
        }
        updated = {**existing, "head": {**existing["head"], "sha": self.source_head}}
        cases = (
            ("finalized", evidence(), existing, existing, True),
            ("pending", evidence(status="pending"), existing, existing, False),
            ("untrusted", evidence(login="contributor"), existing, existing, False),
            ("no owner marker", evidence(owned=False), existing, existing, False),
            (
                "stale marker",
                evidence(head_marker=self.source_head),
                existing,
                existing,
                False,
            ),
            (
                "stale summary",
                evidence(summary_head=self.source_head),
                existing,
                existing,
                False,
            ),
            ("malformed", malformed, existing, existing, False),
            ("head moved", evidence(), existing, updated, False),
            (
                "already ready",
                evidence(),
                {**existing, "draft": False},
                existing,
                False,
            ),
        )
        self.mirror.workflow = "trusted workflow"
        fingerprint = hashlib.sha256(self.mirror.workflow.encode()).hexdigest()
        baseline = {
            "message": f"chore: shadow baseline\n\nSource tree at {self.merge_base}; recorder {fingerprint}"
        }
        for name, comment, listed, current, promoted in cases:
            with self.subTest(name=name):
                self.api.reset_mock()
                self.api.call.side_effect = [baseline, current]
                self.api.pages.return_value = [{"body": HOLD}, comment]
                with patch.object(self.mirror, "get_ref", return_value="mirror-base"):
                    result = self.mirror.replay(
                        1, self.pr, self.files, self.merge_base, {1: listed}
                    )
                self.assertEqual(result["status"], "ready" if promoted else "unchanged")
                if promoted:
                    self.api.mark_ready.assert_called_once_with(current, 1)
                else:
                    self.api.mark_ready.assert_not_called()
                self.api.write.assert_not_called()

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

    def test_missing_upstream_pr_does_not_abort_batch(self):
        self.api.dry_run = True
        self.api.call.side_effect = [
            APIError("GET", "pulls/1", 404, "Not Found"),
            self.pr,
            self.pr,
            {"merge_base_commit": {"sha": self.merge_base}},
        ]
        self.api.pages.side_effect = [[], self.files]
        with (
            patch("mirror.API", return_value=self.api),
            patch(
                "mirror.sys.argv",
                ["mirror", "sync", "--dry-run", "--pr", "1", "--pr", "2"],
            ),
            patch.dict("mirror.os.environ", {"SHADOW_MIRROR_TOKEN": "unused"}),
            redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(main(), 0)
        results = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(
            [(r["pr"], r["status"]) for r in results], [(1, "skipped"), (2, "dry-run")]
        )
        self.assertEqual(results[0]["reason"], "upstream PR not found")
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

    def test_gate_comment_is_only_created_when_absent(self):
        unrelated = {"id": 1, "body": "unrelated"}
        self.api.pages.return_value = [unrelated]
        self.mirror.ensure_gate_comment(3)
        self.api.pages.assert_called_once_with(FORK, "issues/3/comments")
        self.api.write.assert_called_once_with("issues/3/comments", {"body": HOLD})

        for marker in (HOLD_MARKER, GATE_MARKER):
            for verdict in ("**PASS** · Nothing new recorded", "**HOLD** · Changed"):
                with self.subTest(marker=marker, verdict=verdict):
                    self.api.reset_mock()
                    self.api.pages.return_value = [
                        unrelated,
                        {"id": 2, "body": f"{marker}\n{verdict}"},
                    ]
                    self.mirror.ensure_gate_comment(3)
                    self.api.write.assert_not_called()


class VerdictTest(unittest.TestCase):
    head = "a" * 40

    def setUp(self):
        self.api = Mock(spec=API)
        self.pr = {
            "state": "open",
            "base": {"ref": "shadow/base/18-r2", "repo": {"full_name": FORK}},
            "head": {
                "ref": "shadow/pr/18-r2",
                "repo": {"full_name": FORK},
                "sha": self.head,
            },
        }
        self.api.call.return_value = self.pr
        self.api.viewer_login.return_value = "mirror-owner"
        self.api.pages.return_value = []
        self.legacy = {
            "id": 42,
            "body": HOLD,
            "user": {"login": "mirror-owner"},
        }

    def publish(self):
        publish_verdict(
            self.api, 173, self.head, "PASS", "Nothing new.\n\n[App](url)", "b" * 32
        )

    def test_migrates_the_owned_legacy_comment_in_place_and_is_idempotent(self):
        self.api.pages.return_value = [{"body": ""}, self.legacy]
        self.publish()
        path, data, method = self.api.write.call_args.args
        self.assertEqual((path, method), ("issues/comments/42", "PATCH"))
        self.assertEqual(
            data["body"],
            f"{GATE_MARKER}\nHead: `{self.head}`\n\n**PASS** · Nothing new."
            "\n\n[App](url)\n\nNot PR approval; merging policy still applies."
            f"\n\n[Written by Devin](https://app.devin.ai/sessions/{'b' * 32})",
        )
        self.assertEqual(self.api.call.call_args_list, [call(FORK, "pulls/173")] * 2)
        self.api.pages.assert_called_once_with(FORK, "issues/173/comments")
        self.api.write.reset_mock()
        self.api.pages.return_value = [{**self.legacy, "body": data["body"]}]
        self.publish()
        self.api.write.assert_not_called()

    def test_creates_only_when_no_marker_exists(self):
        self.api.pages.return_value = [{"body": "Unrelated"}]
        self.publish()
        path, data = self.api.write.call_args.args
        self.assertEqual(path, "issues/173/comments")
        self.assertTrue(data["body"].startswith(GATE_MARKER))

    def test_rejects_foreign_or_duplicate_comments(self):
        for comments in (
            [{**self.legacy, "user": {"login": "someone-else"}}],
            [self.legacy, {**self.legacy, "id": 43, "body": GATE_MARKER}],
        ):
            with self.subTest(comments=comments):
                self.api.pages.return_value = comments
                with self.assertRaises(ValueError):
                    self.publish()
                self.api.write.assert_not_called()

    def test_rejects_non_mirror_closed_and_stale_pulls(self):
        for pr in (
            {**self.pr, "state": "closed"},
            {**self.pr, "base": {**self.pr["base"], "ref": "main"}},
            {**self.pr, "head": {**self.pr["head"], "repo": {"full_name": SOURCE}}},
            {**self.pr, "head": {**self.pr["head"], "sha": "b" * 40}},
        ):
            with self.subTest(pr=pr):
                self.api.call.return_value = pr
                with self.assertRaises(ValueError):
                    self.publish()
                self.api.pages.assert_not_called()
                self.api.write.assert_not_called()

    def test_rechecks_head_before_writing(self):
        self.api.pages.return_value = [self.legacy]
        self.api.call.side_effect = [
            self.pr,
            {**self.pr, "head": {**self.pr["head"], "sha": "b" * 40}},
        ]
        with self.assertRaisesRegex(ValueError, "PR head changed"):
            self.publish()
        self.api.write.assert_not_called()

    def test_rejects_invalid_input_without_api_calls(self):
        for number, head, verdict, reason in (
            (0, self.head, "PASS", "reason"),
            ("173", self.head, "PASS", "reason"),
            (173, "a" * 7, "PASS", "reason"),
            (173, self.head, "APPROVE", "reason"),
            (173, self.head, "HOLD", " "),
            (173, self.head, "HOLD", GATE_MARKER),
            (173, self.head, "HOLD", "é" * 8192),
        ):
            with self.subTest(number=number, head=head, verdict=verdict):
                with self.assertRaises(ValueError):
                    publish_verdict(self.api, number, head, verdict, reason)
                self.api.assert_not_called()
                self.assertEqual(self.api.mock_calls, [])

    def test_verdict_command_requires_mirror_secret_not_gh_token(self):
        argv = [
            "mirror.py",
            "verdict",
            "--pr=173",
            f"--head={self.head}",
            "--verdict=HOLD",
            "--body=Awaiting evidence",
        ]
        with (
            patch("mirror.sys.argv", argv),
            patch.dict("mirror.os.environ", {"GH_TOKEN": "unused"}, clear=True),
            patch("mirror.publish_verdict") as publish,
            patch("mirror.sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            main()
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
