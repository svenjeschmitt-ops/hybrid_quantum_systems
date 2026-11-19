"""Closing the panel terminates calculation and figure processes."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

from hybrid_panel import ParameterPanel


class PanelCloseTests(unittest.TestCase):
    def test_close_terminates_active_and_previous_plot_processes(self):
        panel = object.__new__(ParameterPanel)
        panel.process = Mock()
        panel.plot_processes = [Mock(), Mock()]
        panel.assistant_cancel = threading.Event()
        panel.temp_path = None
        panel.pulse_preview = Mock()
        panel.root = Mock()
        processes = [panel.process, *panel.plot_processes]
        with patch.object(panel, 'terminate_plot_process') as terminate, \
             patch('hybrid_panel.messagebox.askyesno', side_effect=AssertionError('No confirmation expected')):
            panel.close()
        self.assertEqual([call.args[0] for call in terminate.call_args_list], processes)
        self.assertTrue(panel.assistant_cancel.is_set())
        panel.pulse_preview.close.assert_called_once()
        panel.root.destroy.assert_called_once()
        self.assertEqual(panel.plot_processes, [])
        self.assertIsNone(panel.process)

    @unittest.skipUnless(os.name == 'posix', 'POSIX parent-lifetime check')
    def test_figure_process_exits_when_panel_process_disappears(self):
        child = r"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from panel_lifetime import watch_panel_parent
watch_panel_parent()
Path(sys.argv[2]).write_text('ready')
time.sleep(1.5)
Path(sys.argv[3]).write_text('survived')
"""
        parent = r"""
import os, subprocess, sys, time
from pathlib import Path
child, source, ready, survived = sys.argv[1:]
env = dict(os.environ, HYBRID_PANEL_PARENT_PID=str(os.getpid()))
subprocess.Popen([sys.executable, '-c', child, source, ready, survived],
                 env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
limit = time.monotonic()+5
while not Path(ready).exists() and time.monotonic()<limit:
    time.sleep(.02)
assert Path(ready).exists()
"""
        with tempfile.TemporaryDirectory() as folder:
            ready = Path(folder)/'ready'
            survived = Path(folder)/'survived'
            subprocess.run([sys.executable, '-c', parent, child, str(Path(__file__).parent),
                            str(ready), str(survived)], check=True, timeout=10)
            self.assertTrue(ready.exists())
            time.sleep(2)
            self.assertFalse(survived.exists(), 'Child survived its panel')

    @unittest.skipUnless(os.name == 'posix', 'POSIX process group check')
    def test_process_ignoring_termination_is_force_closed(self):
        code = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready", flush=True); time.sleep(60)'
        proc = subprocess.Popen([sys.executable, '-u', '-c', code], stdout=subprocess.PIPE,
                                text=True, start_new_session=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            ParameterPanel.terminate_plot_process(proc)
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()


if __name__ == '__main__':
    unittest.main()
