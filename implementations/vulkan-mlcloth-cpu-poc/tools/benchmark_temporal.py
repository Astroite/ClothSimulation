"""Sequential ABBA Vulkan measurements; preserves device-load evidence and raw reports."""
from pathlib import Path
import argparse,json,subprocess,sys,threading,time,statistics
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=int,default=180);p.add_argument('--warmup',type=int,default=40)
    p.add_argument('--temporal',type=Path,default=ROOT/'.work/temporal_v4/temporal-init.vthood');p.add_argument('--output',type=Path,default=ROOT/'.work/temporal_v4/benchmark')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);manifest=ROOT/'.work/demo/demo.json';exe=ROOT/'.work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe'
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    runs=[]
    for i,mode in enumerate(('gnn','temporal','temporal','gnn')):
        folder=a.output/f'{i}-{mode}';folder.mkdir(exist_ok=True);loads=[];stop=threading.Event()
        def monitor():
            while not stop.is_set():
                try:
                    r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
                    loads.append(r.stdout.strip())
                except (OSError,subprocess.SubprocessError):loads.append('unavailable')
                stop.wait(2)
        worker=threading.Thread(target=monitor);worker.start()
        args=[str(exe),'--demo-manifest',str(manifest),'--hybrid-algorithm',mode,'--temporal-model',str(a.temporal.resolve()),
              '--fixed-frame-dt',str(1/60),'--capture-metrics','1','--metric-warmup',str(a.warmup),'--frames',str(a.frames)]
        try:
            with (folder/'run.log').open('w') as log:subprocess.run(args,cwd=ROOT/'.work/Vulkan',startupinfo=startup,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        finally:stop.set();worker.join()
        report=json.loads((manifest.parent/'performance.json').read_text());report['device_load_samples']=loads
        (folder/'performance.json').write_text(json.dumps(report,indent=2));runs.append(dict(mode=mode,report=report));print(f'Completed {i}: {mode}',flush=True)
    med={mode:statistics.median(r['report']['gnn_inference_p95_ms'] for r in runs if r['mode']==mode) for mode in ('gnn','temporal')}
    result=dict(runs=runs,median_run_inference_p95_ms=med,relative_inference_overhead=med['temporal']/med['gnn']-1,
        performance_qualified=False,reason='Shared device workload was not controlled; report is diagnostic, not a matched-load acceptance result',
        memory_scope='gnn_buffer_allocation_bytes includes network/history/checkpoint buffers; excludes physics/render allocations and driver overhead')
    (a.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'}))
if __name__=='__main__':main()
