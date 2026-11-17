"""Trusted launcher. No candidate code executes before chroot + irreversible drop."""
import ctypes
import errno
import os
import resource
import sys

root = os.path.realpath(sys.argv[1])
assert os.geteuid() == 0, 'Task requires CAP_SYS_CHROOT plus SETUID/SETGID/SETPCAP'
libc = ctypes.CDLL(None, use_errno=True)
os.chdir(root)
os.chroot('.')
os.chdir('/')
os.setgroups([])
for cap in range(64):
    if libc.prctl(24, cap, 0, 0, 0) < 0:  # PR_CAPBSET_DROP
        assert ctypes.get_errno() == errno.EINVAL
        break
assert libc.prctl(47, 4, 0, 0, 0) == 0  # clear ambient capabilities
assert libc.prctl(38, 1, 0, 0, 0) == 0  # no_new_privs
os.setresgid(65534, 65534, 65534)
os.setresuid(65534, 65534, 65534)
class Header(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
class Cap(ctypes.Structure):
    _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]
assert libc.capset(ctypes.byref(Header(0x20080522, 0)), (Cap * 2)()) == 0
for kind, limit in [(resource.RLIMIT_CPU, 5), (resource.RLIMIT_AS, 512 * 1024**2),
                    (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NPROC, 0),
                    (resource.RLIMIT_CORE, 0), (resource.RLIMIT_NOFILE, 64)]:
    resource.setrlimit(kind, (limit, limit))
# Popen(close_fds=True) leaves only stdin/stdout/stderr; never inherit a host dir fd.
os.execve('/usr/local/bin/python3', ['python3', '-I', '-B', '/sandbox/guest.py'],
          {'PATH': '/usr/local/bin:/usr/bin', 'LANG': 'C.UTF-8', 'HOME': '/nonexistent', 'PYTHONHASHSEED': '0'})
