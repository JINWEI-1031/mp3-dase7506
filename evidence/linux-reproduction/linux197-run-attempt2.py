"""Run once on Linux CPU with bounded affinity, address space, RSS and wall time."""
import os,sys,resource,time,json,subprocess,signal,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/'source';COURSE=Path('/root/autodl-tmp/mp3-gpu-lab-20261120/MP3_SQL_student_package')

def main():
 import psutil,importlib.metadata as m
 if len(sys.argv)>1 and sys.argv[1]=='worker':
  sys.path[:0]=[str(SOURCE),str(COURSE)]
  import runpy
  mode=sys.argv[2]
  if mode=='predict':sys.argv=['predict','--split','test','--method',str(SOURCE/'submission.py'),'--workers','1','--output',str(ROOT/'run2/predictions.json')]
  else:sys.argv=['score','--split','test','--predictions',str(ROOT/'run2/predictions.json'),'--workers','4','--output',str(ROOT/'run2/score.json')]
  runpy.run_module('mp3_sql.'+('predict' if mode=='predict' else 'score'),run_name='__main__');return
 out=ROOT/'run2';out.mkdir(exist_ok=False)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MP3_BASE_MODEL='/root/autodl-tmp/mp3-final197-linux-20261122/base/c1899de289a04d12100db370d81485cdf75e47ca',MP3_ADAPTER_DIR=str(ROOT/'adapter'),MP3_LOG_DIR=str(out/'model-logs'))
 cpus=sorted(os.sched_getaffinity(0))[:4]
 def limits():
  os.sched_setaffinity(0,cpus);resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
 meta={'platform':platform.platform(),'python':sys.version,'packages':{x:m.version(x) for x in ['torch','transformers','peft','accelerate','safetensors','sqlglot','psutil']},'cpu_affinity':cpus,'memory_limits':'16GiB RLIMIT_AS per process plus sum-RSS watchdog; cgroup is read-only','gpu_used':False}
 (out/'environment.json').write_text(json.dumps(meta,indent=2))
 for mode,wall in [('predict',7200),('score',180)]:
  start=time.monotonic();peak=0;reason=None;cpu_before=resource.getrusage(resource.RUSAGE_CHILDREN)
  with (out/(mode+'.stdout.log')).open('w') as stdout,(out/(mode+'.stderr.log')).open('w') as stderr:
   p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker',mode],stdout=stdout,stderr=stderr,env=env,preexec_fn=limits,start_new_session=True,cwd=SOURCE)
   proc=psutil.Process(p.pid)
   while p.poll() is None:
    try:rss=sum(q.memory_info().rss for q in [proc]+proc.children(recursive=True) if q.is_running());peak=max(peak,rss)
    except psutil.Error:rss=0
    if rss>16*1024**3 or time.monotonic()-start>wall:
     reason='RSS_LIMIT' if rss>16*1024**3 else 'WALL_LIMIT';os.killpg(p.pid,signal.SIGKILL);break
    time.sleep(.1)
   p.wait()
  cpu_after=resource.getrusage(resource.RUSAGE_CHILDREN)
  result={'exit_code':p.returncode,'seconds':time.monotonic()-start,'process_tree_CPU_seconds':(cpu_after.ru_utime+cpu_after.ru_stime)-(cpu_before.ru_utime+cpu_before.ru_stime),'peak_process_tree_RSS_bytes':peak,'reason':reason,'all_cpu_affinity':cpus}
  (out/(mode+'-cost.json')).write_text(json.dumps(result,indent=2))
  if p.returncode or reason:raise SystemExit(1)
 old=json.loads((SOURCE/'outputs/test.json').read_text());new=json.loads((out/'predictions.json').read_text());score=json.loads((out/'score.json').read_text())
 result={'Linux_correct':score['correct'],'n':score['n'],'same_final_SQL_count':sum(a['sql']==b['sql'] for a,b in zip(old['predictions'],new['predictions'])),'all_predictions_and_traces_equal':old['predictions']==new['predictions'],'score_SHA256':hashlib.sha256((out/'score.json').read_bytes()).hexdigest()}
 (out/'FINAL.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
