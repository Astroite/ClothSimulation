"""Bounded algorithm smoke run, NOT a full performance/cloth qualification."""
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
assets=ROOT/".work/demo"
manifest=assets/"demo.json"
exe=ROOT/".work/Vulkan/build-mlclothcpu/bin/mlclothcpu.exe"
out=assets/("gnn-smoke-"+datetime.now().strftime("%Y%m%d-%H%M%S"))
out.mkdir()
report={"executable_sha256":hashlib.sha256(exe.read_bytes()).hexdigest(),"scope":"Short fixed-clock playback, final-state diagnostics only; complex clip first five seconds.","performance_qualified":False,"cloth_quality_qualified":False,"cases":[]}
clips=json.loads(manifest.read_text(encoding="utf-8"))["clips"]
startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
for clip in clips:
    seconds=5 if clip["id"]=="complex" else clip["duration"]*(1 if clip["id"]=="idle" else 2)
    folder=out/clip["id"];folder.mkdir()
    command=[str(exe),"--demo-manifest",str(manifest),"--hybrid-algorithm","gnn","--animation",clip["id"],"--frames",str(math.ceil(seconds*60)+2),"--fixed-frame-dt","0.0166666666666667","--playback-speed","1","--dump-physics","1","--screenshot",str(folder/"final.ppm")]
    with (folder/"runtime.log").open("w") as log:
        subprocess.run(command,cwd=ROOT/".work/Vulkan",startupinfo=startup,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=240)
    for name in ("physics-snapshot.json","performance.json"):
        shutil.copy2(assets/name,folder/name)
    snapshot=json.loads((folder/"physics-snapshot.json").read_text());metrics=json.loads((folder/"performance.json").read_text())
    tris=np.array(snapshot["triangles"]).reshape(-1,3);rest=np.array(snapshot["rest"])
    edges=np.unique(np.sort(np.concatenate((tris[:,[0,1]],tris[:,[1,2]],tris[:,[2,0]])),axis=1),axis=0)
    rest_len=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)
    actors=[]
    for actor in snapshot["actors"]:
        x=np.array(actor["cloth"]);ratio=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/np.maximum(rest_len,1e-12)
        actors.append({"finite":bool(np.isfinite(x).all()),"edge_ratio_p95":float(np.percentile(ratio,95)),"edge_ratio_max":float(ratio.max())})
    case={"clip":clip["id"],"time":metrics["simulation_time"],"target_time":seconds,"reached":metrics["simulation_time"]>=seconds-.001 and not metrics["paused"],"actors":actors}
    report["cases"].append(case);(out/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(case),flush=True)
    assert case["reached"] and all(a["finite"] for a in actors), case
print(out,flush=True)
