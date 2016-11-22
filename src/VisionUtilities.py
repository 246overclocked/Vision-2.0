#
# This is a small utilities package that currently contains a threading extension for the main vision class.
# This should contain any unrelated extension that are necessary for the main vision class to run.
#

import threading

__author__ = "Jacob Nazarenko"
__email__ = "jacobn@bu.edu"
__license__ = "MIT"


class StoppableThread(threading.Thread):

    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, target=None):
        super(StoppableThread, self).__init__()
        self._stop = threading.Event()
        self.my_target = target

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        err_code = 0
        while not self.stopped() and not err_code:
            err_code = self.my_target()