"""Keep panel-owned simulation processes from outliving their panel."""
import os
import threading
import time


def watch_panel_parent():
    """Terminate this process if the originating panel process disappears.

    The environment is set only on processes launched by the panel. Detached
    CLI simulations do not install a watcher. Sweep workers inherit the panel
    PID and are monitored too.
    """
    if os.name != 'posix':
        return
    value = os.environ.get('HYBRID_PANEL_PARENT_PID')
    if not value:
        return
    try:
        parent_pid = int(value)
    except ValueError:
        return
    if parent_pid <= 1 or parent_pid == os.getpid():
        return
    def watch():
        while True:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                os._exit(0)
            except PermissionError:
                pass
            time.sleep(.25)
    threading.Thread(target=watch, name='panel-lifetime', daemon=True).start()
