"""Independent CPU/PyTorch check of the port's features, graph and decoder.

Uses the sibling's reference CODE and external weights; no character assets are
copied. Run with ../vulkan-gnn-poc/.venv/Scripts/python.exe.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "vulkan-gnn-poc"))
from real_scene.tinyhood import load_tinyhood
from real_scene.fine15 import Fine15Weights

def validate(directory, weights, temporal=None):
    torch.set_num_threads(4)
    n, m, e, blocks = map(int, (directory / "shape.txt").read_text().split())
    def read(name, width=1, dtype="<f4"):
        a = np.fromfile(directory / name, dtype=dtype)
        return a.reshape(-1, width) if width > 1 else a
    def tensor(a): return torch.from_numpy(a.copy())
    nodes, mesh = read("nodes.f32", 20), read("mesh.f32", 12)
    direct, inverse = read("direct.f32", 9), read("inverse.f32", 9)
    send, recv = read("senders.u32", dtype="<u4").astype(np.int64), read("receivers.u32", dtype="<u4").astype(np.int64)
    nearest = read("nearest.u32", dtype="<u4").astype(np.int64)
    wc = np.flatnonzero(nearest != 0xffffffff); wo = nearest[wc]
    reverse = read("reverse.u32", dtype="<u4")
    starts, counts = read("begin.u32", dtype="<u4"), read("count.u32", dtype="<u4")
    expected_reverse = wc[np.lexsort((wc, wo))]
    assert np.array_equal(reverse[:len(wc)], expected_reverse)
    for i in range(m):
        expected = wc[wo == i]
        assert counts[i] == len(expected)
        assert np.array_equal(reverse[starts[i]:starts[i]+counts[i]], expected)
    x, v, pins, rest = (read(name, 4) for name in ("x.f32", "velocity.f32", "pins.f32", "rest.f32"))
    body, future, normals = (read(name, 4)[:, :3] for name in ("body.f32", "future.f32", "normals.f32"))
    effective = np.where(x[:, 3:4] == 0, pins[:, :3], x[:, :3])
    # Brute-force oracle independent of the GPU BVH.
    expected_nearest = []
    for p in effective:
        ds = ((body-p)**2).sum(axis=1)
        idx = ds.argmin() if m else 0
        expected_nearest.append(idx if m and ds[idx] <= np.float32(.03)**2 else 0xffffffff)
    assert np.array_equal(nearest, expected_nearest), "nearest-point BVH disagrees with brute force"
    w = Fine15Weights.from_vhood(weights)
    if temporal:
        sys.path.insert(0,str(ROOT))
        from temporal.format import load_model,load_asset
        asset=load_asset(temporal)
        w=Fine15Weights({name:torch.from_numpy(np.frombuffer(view.data,dtype='<f4').copy()).reshape(view.shape) for name,view in asset.tensors.items()})
    triangles = read("triangles.u32", 3, "<u4")
    a, b, c = (effective[triangles[:, k]] for k in range(3))
    tri_normal = np.cross(b-a,c-b)+np.cross(c-b,a-c)+np.cross(a-c,b-a)
    normal = np.zeros((n, 3), np.float32)
    # Ascending triangle order per vertex, matching the definition of area normals.
    for t, ids in enumerate(triangles):
        normal[ids] += tri_normal[t]
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.where(length > 1e-10, normal/np.maximum(length, 1e-30), [0, 1, 0]).astype(np.float32)
    embedding = w.require("model.nodetype_embedding.weight", (9,9)).numpy()
    node_raw = np.concatenate((np.where(x[:,3:4] == 0, effective-x[:,:3], v[:,:3]/30), embedding[np.where(x[:,3] == 0,3,0)], normal,
                               np.full((n,1),1/30,np.float32), np.log(np.maximum(rest[:,3:4],1e-20))), axis=1)
    body_raw = np.concatenate((future-body, np.tile(embedding[1],(m,1)), normals, np.full((m,1),1/30,np.float32), -np.ones((m,1),np.float32)),axis=1)
    mat = np.array([(np.log(3.9625778333333325e-5)-np.log(6.370782056371576e-8))/(np.log(.0013139737991266374)-np.log(6.370782056371576e-8)),
                    (np.log(23600)-np.log(15909))/(np.log(63636)-np.log(15909)),(44400-3535.414406069427)/(93333.73508005822-3535.414406069427)],np.float32)
    expected_nodes = np.concatenate((w.normalize("node",tensor(np.concatenate((node_raw,body_raw)))).numpy(), np.concatenate((np.tile(mat,(n,1)),-np.ones((m,3),np.float32)))),axis=1)
    dr, rr = effective[send]-effective[recv], rest[send,:3]-rest[recv,:3]
    raw_mesh = np.column_stack((dr,np.linalg.norm(dr,axis=1),rr,np.linalg.norm(rr,axis=1),np.full(e,1/30,np.float32)))
    expected_mesh = np.column_stack((w.normalize("mesh_edge",tensor(raw_mesh)).numpy(),np.tile(mat,(e,1))))
    dc, dt = effective[wc]-body[wo], effective[wc]-future[wo]
    raw_world = np.column_stack((dc,np.linalg.norm(dc,axis=1),dt,np.linalg.norm(dt,axis=1),np.full(len(wc),1/30,np.float32)))
    expected_direct = w.normalize("world_edge",tensor(raw_world)).numpy()
    raw_world[:,[0,1,2,4,5,6]] *= -1
    expected_inverse = w.normalize("world_edge",tensor(raw_world)).numpy()
    feature_errors = {name:float(np.max(np.abs(a-b),initial=0)) for name,a,b in [("node",nodes,expected_nodes),("mesh",mesh,expected_mesh),("direct",direct[wc],expected_direct),("inverse",inverse[wc],expected_inverse)]}
    assert max(feature_errors.values()) < .002, feature_errors
    if temporal:
        for a,b in ((nodes,expected_nodes),(mesh,expected_mesh),(direct[wc],expected_direct),(inverse[wc],expected_inverse)):
            np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)
    graph = SimpleNamespace(cloth_nodes=tensor(nodes[:n]),obstacle_nodes=tensor(nodes[n:]),mesh_edges=tensor(mesh),direct_world=tensor(direct[wc]),inverse_world=tensor(inverse[wc]),
                            mesh_senders=tensor(send),mesh_receivers=tensor(recv),world_cloth=tensor(wc),world_obstacle=tensor(wo))
    with torch.inference_mode():
        if temporal:
            history=read('history-input.f32',28)
            stored=read('history.f32',28).reshape(n+m,4,7)
            ticks,current_only=map(int,(directory/'temporal.txt').read_text().split())
            expected_valid=np.array([int(t>=max(0,4-ticks)) for t in range(4)],np.float32)
            np.testing.assert_array_equal(stored[:,:,6],np.tile(expected_valid,(n+m,1)))
            mean,std=(a.numpy().reshape(-1) for a in w.normalizer('node'))
            np.testing.assert_allclose(stored[:n,-1,:3],(v[:,:3]/30-mean[:3])/std[:3],atol=1e-5,rtol=1e-4)
            np.testing.assert_allclose(stored[:,-1,3:6],nodes[:,12:15],atol=1e-5,rtol=1e-4)
            expected_history=np.tile(stored[:,-1:],(1,4,1)) if current_only else stored
            np.testing.assert_array_equal(history,expected_history.reshape(n+m,28))
            # Rebuild ALL four observations from separately copied physical input,
            # host surface correspondence and triangle normals. Never use the
            # temporal shader's own stored history as the numerical oracle.
            capture=np.loadtxt(directory/'history-capture.txt',dtype=np.int64).reshape(4,2)
            hx=read('history-x.f32',4).reshape(4,n,4);hv=read('history-v.f32',4).reshape(4,n,4)
            hp=read('history-pins.f32',3).reshape(4,n,3);hd=read('history-body-delta.f32',3).reshape(4,m,3)
            hn=read('history-body-normal.f32',3).reshape(4,m,3);observations=[]
            for tick in range(max(0,ticks-4),ticks):
                slot=tick%4
                assert capture[slot,0]==tick+1 and capture[slot,1], 'insufficient independent history captures after restore'
                xx=np.where(hx[slot,:,3:]==0,hp[slot],hx[slot,:,:3])
                a,b,c=(xx[triangles[:,k]] for k in range(3))
                tn=np.cross(b-a,c-b)+np.cross(c-b,a-c)+np.cross(a-c,b-a);cn=np.zeros((n,3),np.float32)
                for t,ids in enumerate(triangles):cn[ids]+=tn[t]
                ll=np.linalg.norm(cn,axis=-1,keepdims=True);cn=np.where(ll>1e-10,cn/np.maximum(ll,1e-30),[0,1,0]).astype(np.float32)
                raw=np.column_stack((np.concatenate((hv[slot,:,:3]/30,hd[slot])),np.concatenate((cn,hn[slot])),np.ones(n+m,np.float32)))
                raw[:,:6]=(raw[:,:6]-mean[[0,1,2,12,13,14]])/std[[0,1,2,12,13,14]];observations.append(raw)
            missing=observations[0].copy();missing[:,-1]=0
            independent=np.stack([missing]*(4-len(observations))+observations,axis=1)
            np.testing.assert_allclose(stored,independent,atol=1e-5,rtol=1e-4)
            predicted=w.inverse('output',load_model(temporal).eval()(graph,tensor(history))).numpy()
        else:
            predicted = w.inverse("output",load_tinyhood(weights).eval()(graph)).numpy()
    actual = read("raw.f32",4)[:,:3]
    error = np.abs(actual-predicted)
    if temporal:np.testing.assert_allclose(actual,predicted,atol=1e-5,rtol=1e-4)
    report = {"vertices":n,"body_nodes":m,"directed_mesh_edges":e,"world_edges":len(wc),"blocks":blocks,"feature_max_abs":feature_errors,
              "decoder_max_abs_m":float(error.max()),"decoder_mean_abs_m":float(error.mean()),"stable_reverse_csr":True,"nearest_bvh":True,
              "weight_sha256":hashlib.sha256((temporal or weights).read_bytes()).hexdigest(),"temporal":bool(temporal),"passed":bool(np.isfinite(actual).all() and error.max()<2e-5),"performance_qualified":False}
    (directory/"reference-report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    assert report["passed"], report

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--directory",type=Path,default=ROOT/".work/demo/gnn-validation");p.add_argument("--weights",type=Path,default=ROOT.parent/"vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood");p.add_argument('--temporal',type=Path);a=p.parse_args();validate(a.directory,a.weights,a.temporal)
