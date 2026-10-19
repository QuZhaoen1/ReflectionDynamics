import contextlib
import io
import json
import resource

resource.setrlimit(resource.RLIMIT_CPU,(5,5))
resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
payload=json.load(open('/input/payload.json'))
passed=False
try:
    namespace={}
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(payload['code'],'candidate.py','exec'),namespace)
        exec(compile(payload['tests'],'tests.py','exec'),namespace)
        namespace['check'](namespace[payload['entry']])
    passed=True
except BaseException:
    passed=False
print('RESULT_JSON:'+json.dumps({'finished':True,'passed':passed}),flush=True)
