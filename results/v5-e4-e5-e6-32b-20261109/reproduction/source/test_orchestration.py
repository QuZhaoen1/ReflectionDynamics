import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pipeline


class ResumeIntegrity(unittest.TestCase):
    def test_verified_stage_skips_execution_and_rejects_changed_output(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pipeline,'ROOT',Path(d)):
            p=Path(d)/'result.json';calls=[]
            def work():
                calls.append(1);p.write_text('original');return {'ok':True},[p]
            pipeline.stage('test',work,{'protocol':1})
            pipeline.stage('test',work,{'protocol':1})
            self.assertEqual(len(calls),1)
            p.write_text('changed')
            with self.assertRaises(ValueError):pipeline.stage('test',work,{'protocol':1})

    def test_changed_protocol_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pipeline,'ROOT',Path(d)):
            pipeline.stage('test',lambda:({'ok':True},[]),{'protocol':1})
            with self.assertRaises(ValueError):pipeline.stage('test',lambda:({},[]),{'protocol':2})

    def test_failed_stage_is_retried_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pipeline,'ROOT',Path(d)):
            def fail():raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):pipeline.stage('test',fail,{})
            self.assertFalse((Path(d)/'completed/test.json').exists())
            result=pipeline.stage('test',lambda:({'recovered':True},[]),{})
            self.assertTrue(result['recovered'])

    def test_parallel_failure_terminates_sibling(self):
        import sys,time
        with tempfile.TemporaryDirectory() as d:
            start=time.monotonic()
            with self.assertRaises(RuntimeError):
                pipeline.parallel([([sys.executable,'-c','raise SystemExit(2)'],Path(d)/'failed.log',{}),
                                   ([sys.executable,'-c','import time; time.sleep(120)'],Path(d)/'sibling.log',{})])
            self.assertLess(time.monotonic()-start,20)

    def test_batched_drafts_keep_question_ids_and_resume_without_generation(self):
        import worker
        from cavr.io import save_json,save_rows,load_events,sha256,seed_for
        calls=[]
        class FakeEngine:
            def __init__(self,cfg):pass
            def select(self,ck):return {'model':'fixture','adapter_sha256':None}
            def generate(self,requests,lookup):
                calls.append(len(requests))
                return [{'text':'2','tokens':1,'prompt_tokens':1,'seconds':0.,'ended':True,'truncated':False} for _ in requests]
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);identifier=next(str(i) for i in range(100) if seed_for(str(i))%8==0)
            q={'id':identifier,'question':'1+1?','gold':'2','task':'math','split':'test'}
            save_rows(r/'questions.jsonl',[q]);save_json(r/'config.json',{'model_path':'fixture'})
            job={'config':str(r/'config.json'),'checkpoint':'base','adapter':None,'adapter_sha256':None,
                 'items':[{'kind':'draft','input':str(r/'questions.jsonl'),'input_sha256':sha256(r/'questions.jsonl'),
                           'bank':'draft','reps':2,'prefix':str(r/'draft')}]}
            save_json(r/'job.json',job)
            with patch.object(worker,'Engine',FakeEngine),patch.object(worker,'grade',lambda *a:{'answer':'2','correct':True,'valid':True}),\
                 patch('sys.argv',['worker','--job',str(r/'job.json'),'--shard','0']):
                worker.main();worker.main()
            rows=load_events(r/'draft-0.jsonl')
            self.assertEqual(calls,[2])
            self.assertEqual({x['id'] for x in rows},{identifier+'-d0',identifier+'-d1'})
            self.assertTrue(all(x['problem_id']==identifier for x in rows))


if __name__=='__main__':unittest.main()
