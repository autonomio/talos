# Copyright (c) 2012-2013 Luke Macken <lmacken@redhat.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" A multi-threaded Quantum Random Generator character device in userspace """

import sys
import six
import cuse
import time
import threading
import traceback
import quantumrandom

from cuse import cuse_api as libcuse

MAX_BUFFER = 100
threads = []
buffer = []


class RandomDataFetcher(threading.Thread):
    """A thread that fills a buffer with binary data"""
    running = False

    def __init__(self, id):
        super(RandomDataFetcher, self).__init__()
        self.id = id

    def run(self):
        log("[Thread %d] Starting" % self.id)
        global buffer
        self.running = True
        try:
            while self.running:
                if len(buffer) > MAX_BUFFER:
                    log("[Thread %d] Buffer at capacity; thread sleeping" %
                        self.id)
                    time.sleep(self.id + 1)
                    continue
                buffer.append(quantumrandom.binary())
                log("[Thread %d] New random data buffered" % self.id)
        except:
            log(traceback.format_exc())
        self.running = False
        log("[Thread %d] Done!" % self.id)


class QuantumRandomDevice(object):

    def __init__(self, num_threads=3):
        self.num_threads = num_threads

    def read(self, req, size, off, file_info):
        log("read(%s)" % size)
        global buffer, threads
        if not threads:
            log("Creating %d threads" % self.num_threads)
            for i, t in enumerate(list(range(self.num_threads))):
                thread = RandomDataFetcher(i)
                thread.setDaemon(True)
                thread.start()
                threads.append(thread)
        data = six.b('')
        while len(data) < size:
            try:
                data += buffer.pop(0)
                break
            except IndexError:
                log("no data")
                time.sleep(0.1)
                continue
        if len(data) > size:
            buffer.append(data[size:])
            data = data[:size]
        libcuse.fuse_reply_buf(req, data, len(data))

    def release(self, req, file_info):
        global threads
        dead = []
        for thread in threads:
            thread.running = False
            dead.append(thread)
        for thread in dead:
            threads.remove(thread)
        libcuse.fuse_reply_err(req, 0)
        log("/dev/qrandom released")


def log(msg):
    print(msg)


def main():
    num_threads = 3
    if '-h' in sys.argv:
        raise SystemExit('Usage: %s [-v] [-t THREADS]')
    if '-t' in sys.argv:
        num_threads = int(sys.argv[sys.argv.index('-t') + 1])
    if '-v' not in sys.argv:
        global log
        def noop(msg): pass
        log = noop

    operations = QuantumRandomDevice(num_threads)
    cuse.init(operations, 'qrandom', [])

    try:
        cuse.main(True)
    except:
        log(traceback.format_exc())
        log("CUSE main ended")


if __name__ == '__main__':
    main()
