"""Trusted guest bootstrap. Install kernel filtering before reading any candidate."""
import contextlib
import ctypes as C
import errno
import io
import json
import os
import resource
import sys

libc = C.CDLL(None, use_errno=True)
assert os.getuid() == os.geteuid() == os.getgid() == os.getegid() == 65534
assert os.getgroups() == []
assert libc.prctl(39, 0, 0, 0, 0) == 1  # PR_GET_NO_NEW_PRIVS
class Header(C.Structure):
    _fields_ = [('version', C.c_uint32), ('pid', C.c_int)]
class Cap(C.Structure):
    _fields_ = [('effective', C.c_uint32), ('permitted', C.c_uint32), ('inheritable', C.c_uint32)]
caps = (Cap * 2)()
assert libc.capget(C.byref(Header(0x20080522, 0)), caps) == 0
assert all(c.effective == c.permitted == c.inheritable == 0 for c in caps)
for cap in range(64):
    value = libc.prctl(23, cap, 0, 0, 0)  # bounding set
    if value < 0:
        assert C.get_errno() == errno.EINVAL
        break
    assert value == 0

lib = C.CDLL('/sandbox/libseccomp.so.2', use_errno=True)
lib.seccomp_init.argtypes = [C.c_uint32]; lib.seccomp_init.restype = C.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [C.c_char_p]
class Compare(C.Structure):
    _fields_ = [('arg', C.c_uint), ('op', C.c_int), ('a', C.c_uint64), ('b', C.c_uint64)]
lib.seccomp_rule_add_array.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint, C.POINTER(Compare)]
lib.seccomp_load.argtypes = [C.c_void_p]
lib.seccomp_release.argtypes = [C.c_void_p]
ctx = lib.seccomp_init(0x00050000 | errno.EPERM)
assert ctx
def allow(name, *comparisons):
    nr = lib.seccomp_syscall_resolve_name(name.encode())
    assert nr >= 0, name
    array = (Compare * len(comparisons))(*comparisons)
    assert lib.seccomp_rule_add_array(ctx, 0x7fff0000, nr, len(array), array) == 0, name

# No socket, fork/clone, exec, ptrace, process_vm, kill, mount, namespace,
# credential, BPF, io_uring, device ioctl, or filesystem mutation operations.
for name in ('read readv close fstat newfstatat stat lstat statx lseek pread64 '
             'readlink readlinkat access faccessat getdents64 getcwd '
             'mmap mprotect munmap mremap brk madvise '
             'rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack futex '
             'clock_gettime clock_getres clock_nanosleep nanosleep gettimeofday time '
             'getrandom uname getpid gettid getuid geteuid getgid getegid '
             'exit exit_group sched_yield sched_getaffinity getrusage getrlimit').split():
    allow(name)
for name in ('write', 'writev'):
    for fd in (1, 2): allow(name, Compare(0, 4, fd, 0))
for name, arg in [('open', 1), ('openat', 2)]:
    mask = os.O_ACCMODE | os.O_CREAT | os.O_TRUNC | os.O_APPEND | os.O_EXCL | (os.O_TMPFILE & ~os.O_DIRECTORY)
    allow(name, Compare(arg, 7, mask, 0))
for command in (1, 3):  # F_GETFD, F_GETFL; cannot duplicate inherited descriptors
    allow('fcntl', Compare(1, 4, command, 0))
allow('prlimit64', Compare(0, 4, 0, 0), Compare(2, 4, 0, 0))
assert lib.seccomp_load(ctx) == 0
lib.seccomp_release(ctx)
print('SANDBOX_READY', file=sys.stderr, flush=True)
payload = json.load(sys.stdin)
passed = False
try:
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(payload['code'], 'candidate.py', 'exec'), namespace)
        exec(compile(payload['tests'], 'tests.py', 'exec'), namespace)
        namespace['check'](namespace[payload['entry']])
    passed = True
except BaseException:
    passed = False
print('RESULT_JSON:' + json.dumps({'finished': True, 'passed': passed}), flush=True)
