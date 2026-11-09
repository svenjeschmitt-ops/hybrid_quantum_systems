"""Offline checks for authentication, context, process permissions and cleanup."""
import os
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import codex_assistant as codex
from codex_assistant import build_messages
from hybrid_settings import validate


class CodexTests(unittest.TestCase):
    def test_context_keeps_edited_settings_separate_from_run(self):
        previous=validate({"FC":5000})
        current=validate({"FC":4900})
        run={"settings":previous,"mode":"Transmon to atom","status":"running",
             "output":"x"*4000+"fidelity=0.9"}
        messages=build_messages("Explain",current,"Atom to transmon",run)
        snapshot=json.loads(messages[1]["content"].split("\n",1)[1].split("\nQuestion:")[0])
        self.assertEqual(snapshot["current_panel"]["settings"]["FC"],4900)
        self.assertEqual(snapshot["last_simulation"]["settings"]["FC"],5000)
        self.assertEqual(snapshot["last_simulation"]["status"],"running")
        self.assertTrue(snapshot["last_simulation"]["output"].endswith("fidelity=0.9"))
        self.assertLessEqual(len(snapshot["last_simulation"]["output"]),3000)
        self.assertEqual(snapshot["parameter_units"]["FC"],"MHz")
        self.assertEqual(len(run["output"]),4012)

    def test_no_run_is_explicit(self):
        self.assertIn('"last_simulation": null',build_messages("Explain",{},"write")[1]["content"])

    def test_prompt_preserves_snapshot_and_allows_only_reading(self):
        prompt=codex.build_prompt(build_messages("What is FC?",{},"Both transfer directions"))
        self.assertIn('"FC": 5000.0',prompt)
        self.assertIn("may read relevant project code",prompt)
        self.assertIn("Do not edit files, run simulations",prompt)
        self.assertNotIn("or access files",prompt)

    @patch.dict(os.environ,{"OPENAI_API_KEY":"test-key","CODEX_API_KEY":"test-key"})
    def test_api_credentials_not_inherited(self):
        env=codex.codex_environment()
        self.assertNotIn("OPENAI_API_KEY",env)
        self.assertNotIn("CODEX_API_KEY",env)
        self.assertIn("OPENAI_API_KEY",os.environ)

    @patch("codex_assistant.find_codex",return_value=None)
    def test_missing_cli_actionable(self, find):
        with self.assertRaisesRegex(RuntimeError,"not found on PATH"):
            codex.check_codex()

    @patch("codex_assistant.shutil.which",return_value="/custom/codex")
    def test_path_installation_takes_precedence(self, which):
        self.assertEqual(codex.find_codex(),"/custom/codex")

    @patch("codex_assistant.shutil.which",return_value=None)
    @patch("codex_assistant.sys.platform","darwin")
    @patch("codex_assistant.platform.machine",return_value="arm64")
    def test_extension_discovered_with_missing_path(self, machine, which):
        with tempfile.TemporaryDirectory() as folder, patch("codex_assistant.Path.home",return_value=Path(folder)):
            for version in ("1.9.0","1.10.0"):
                path=Path(folder)/f".vscode/extensions/openai.chatgpt-{version}-darwin-arm64/bin/macos-aarch64/codex"
                path.parent.mkdir(parents=True)
                path.write_text("#!/bin/sh\n")
                path.chmod(0o755)
            # Isolate this test from system-wide installations on the test machine.
            original=Path.is_file
            with patch("codex_assistant.Path.is_file",lambda p: str(p).startswith(folder) and original(p)):
                self.assertIn("1.10.0",codex.find_codex())
                path.chmod(0o644)
                self.assertIn("1.9.0",codex.find_codex())

    @patch("codex_assistant.shutil.which",return_value="/bin/codex")
    @patch("codex_assistant.subprocess.run")
    def test_only_chatgpt_auth_accepted(self, run, which):
        run.return_value=Mock(returncode=0,stdout="",stderr="Logged in using ChatGPT")
        self.assertEqual(codex.check_codex(),"/bin/codex")
        for status in ("Logged in using an API key","Not logged in"):
            run.return_value=Mock(returncode=0,stdout=status,stderr="")
            with self.assertRaisesRegex(RuntimeError,"ChatGPT sign-in"):
                codex.check_codex()

    @patch("codex_assistant.check_codex",return_value="/bin/codex")
    @patch("codex_assistant.subprocess.Popen")
    def test_cli_permissions_and_final_answer(self, popen, check):
        def start(command,**kwargs):
            self.assertIn("read-only",command)
            self.assertIn('forced_login_method="chatgpt"',command)
            self.assertIn("--ignore-user-config",command)
            self.assertIn("--ephemeral",command)
            self.assertIn("--no-daemon",command)
            self.assertNotIn("shell",kwargs)
            self.assertIn("What is FC?",kwargs["stdin"].read())
            output=Path(command[command.index("--output-last-message")+1])
            output.write_text("FC is 5000 MHz.")
            return Mock(wait=Mock(return_value=0),poll=Mock(return_value=0))
        popen.side_effect=start
        result=list(codex.answer(build_messages("What is FC?",{},"write"),threading.Event(),Path.cwd()))
        self.assertEqual(result,["FC is 5000 MHz."])

    @patch("codex_assistant.check_codex")
    def test_cancel_before_start_does_not_spawn(self, check):
        cancel=threading.Event()
        cancel.set()
        self.assertEqual(list(codex.answer([],cancel,Path.cwd())),[])
        check.assert_not_called()

    @patch("codex_assistant.check_codex",return_value="/bin/codex")
    @patch("codex_assistant._stop")
    @patch("codex_assistant.subprocess.Popen")
    def test_cancel_during_request_cleans_up(self, popen, stop, check):
        cancel=threading.Event()
        def wait(**kwargs):
            cancel.set()
            raise subprocess.TimeoutExpired("codex",.2)
        process=Mock(wait=wait)
        popen.return_value=process
        self.assertEqual(list(codex.answer(build_messages("Explain",{},"write"),cancel,Path.cwd())),[])
        stop.assert_called_once_with(process)

    @patch("codex_assistant.check_codex",return_value="/bin/codex")
    @patch("codex_assistant._stop")
    @patch("codex_assistant.subprocess.Popen")
    def test_cli_failure_is_not_presented_as_answer(self, popen, stop, check):
        popen.return_value=Mock(wait=Mock(return_value=1))
        with self.assertRaisesRegex(RuntimeError,"could not complete"):
            list(codex.answer(build_messages("Explain",{},"write"),threading.Event(),Path.cwd()))
        stop.assert_called_once()

    @patch("codex_assistant.check_codex",return_value="/bin/codex")
    @patch("codex_assistant._stop")
    @patch("codex_assistant.time.monotonic",side_effect=[0,1000])
    @patch("codex_assistant.subprocess.Popen")
    def test_timeout_cleans_up(self, popen, monotonic, stop, check):
        with self.assertRaisesRegex(RuntimeError,"timed out"):
            list(codex.answer(build_messages("Explain",{},"write"),threading.Event(),Path.cwd()))
        stop.assert_called_once_with(popen.return_value)

    @unittest.skipUnless(os.name=="posix","POSIX process group cleanup")
    @patch("codex_assistant.os.killpg")
    def test_stubborn_process_group_is_killed(self, killpg):
        process=Mock(pid=123,poll=Mock(return_value=None))
        process.wait.side_effect=[subprocess.TimeoutExpired("codex",2),0]
        codex._stop(process)
        self.assertEqual(killpg.call_args_list[0].args,(123,codex.signal.SIGTERM))
        self.assertEqual(killpg.call_args_list[1].args,(123,codex.signal.SIGKILL))


if __name__=="__main__":
    unittest.main()
