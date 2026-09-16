"""xdist fail-fast pytest plugin for ops-tilelang.

Makes ``pytest -n ... -x`` interrupt the pytest controller after the first
failure report, which is closer to Ctrl+C than xdist's default scheduling stop.
"""

import os
import signal
import threading


def pytest_configure(config):
    num_processes = getattr(config.option, 'numprocesses', None)
    maxfail = getattr(config.option, 'maxfail', 0)
    should_interrupt_on_fail = maxfail == 1 and num_processes
    if not should_interrupt_on_fail:
        return

    if hasattr(config.option, 'maxworkerrestart'):
        config.option.maxworkerrestart = 0
    if hasattr(config, 'workerinput'):
        return

    if not config.pluginmanager.has_plugin('tk_xdist_failfast'):
        config.pluginmanager.register(_InterruptOnFailPlugin(), 'tk_xdist_failfast')


class _InterruptOnFailPlugin:
    """Controller-side plugin that turns the first failure into SIGINT."""

    def __init__(self):
        self.triggered = False
        self.first_failure_report = None
        self.timer = None
        self.session_finished = False

    def pytest_runtest_logreport(self, report):
        if self.triggered:
            return
        if report.when not in ('setup', 'call', 'teardown') or not report.failed:
            return

        self.triggered = True
        self.first_failure_report = report
        self.timer = threading.Timer(0.1, self._interrupt_pytest_session)
        self.timer.start()

    def pytest_sessionfinish(self, session, exitstatus):
        self.session_finished = True
        if self.timer is not None:
            self.timer.cancel()

    def pytest_terminal_summary(self, terminalreporter, config):
        if self.first_failure_report is None:
            return

        # The full traceback is already printed by pytest's native FAILURES
        # section (the failure report reaches the controller before the
        # delayed SIGINT fires), so only point at the triggering test here.
        terminalreporter.section('First Failure Before Session Interrupt')
        terminalreporter.write_line(self.first_failure_report.nodeid)

    def _interrupt_pytest_session(self):
        if self.session_finished:
            return
        os.kill(os.getpid(), signal.SIGINT)
