"""VTHOOD01 v1: checked tensor container with an explicit history/unit contract."""
from pathlib import Path
import hashlib
import math
import struct
import tempfile
import numpy as np
import torch
from real_scene.formats import TensorAsset, TensorView, TENSOR_HEADER, TENSOR_ENTRY, write_tensor_asset
from real_scene.tinyhood import _export_mlp, _tensor_bytes, _load_mlp
from .model import TemporalTinyHood

MAGIC = b"VTHOOD01"
# version, window, channels, input, width, blocks, Hz, current_only, output unit (dt^2*a)
CONTRACT = (1,4,7,28,32,12,30,0,1)

def load_asset(path):
    blob = Path(path).read_bytes()
    if len(blob) < TENSOR_HEADER.size:
        raise ValueError("truncated temporal model")
    magic,version,count,size,offset,sha,source = TENSOR_HEADER.unpack_from(blob)
    end = TENSOR_HEADER.size + count*TENSOR_ENTRY.size
    if magic != MAGIC or version != 1 or not count or size != len(blob) or offset != (end+15)//16*16 or offset>len(blob):
        raise ValueError("invalid temporal model header")
    if hashlib.sha256(blob[offset:]).digest()!=sha:
        raise ValueError("temporal model checksum mismatch")
    tensors, ranges = {}, []
    for i in range(count):
        name,start,n,rank,*dims = TENSOR_ENTRY.unpack_from(blob,TENSOR_HEADER.size+i*TENSOR_ENTRY.size)
        name=name.split(b'\0')[0].decode('ascii'); shape=tuple(dims[:rank]); stop=start+n*4
        if not name or name in tensors or not 1<=rank<=8 or any(d<=0 for d in shape) or math.prod(shape)!=n or start%16 or start<offset or stop>len(blob):
            raise ValueError("invalid temporal tensor")
        tensors[name]=TensorView(name,shape,start,memoryview(blob)[start:stop]);ranges.append((start,stop))
    ranges.sort()
    if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):
        raise ValueError("overlapping temporal tensors")
    asset=TensorAsset(1,source,sha,blob,tensors)
    contract=np.frombuffer(asset.require('temporal.contract',(9,)).data,dtype='<f4')
    expected=np.array(CONTRACT,dtype=np.float32);expected[7]=contract[7]
    if not np.array_equal(contract,expected) or contract[7] not in (0,1):
        raise ValueError("unsupported temporal feature/history/output contract")
    for view in tensors.values():
        if not np.isfinite(np.frombuffer(view.data,dtype='<f4')).all():
            raise ValueError("nonfinite temporal weights")
    return asset

def export_model(model, teacher, path, checkpoint_sha256='00'*32):
    tensors={}
    _export_mlp(tensors,'model._learned_model.node_encoder',model.node_encoder)
    _export_mlp(tensors,'model._learned_model.edgeset_encoders.mesh',model.mesh_encoder)
    _export_mlp(tensors,'model._learned_model.edgeset_encoders.world',model.world_encoder)
    for i,block in enumerate(model.processor_steps):
        for name in ('mesh_edge_processor','world_edge_processor','node_processor'):
            _export_mlp(tensors,f'model._learned_model.processor_steps.{i}.{name}',getattr(block,name))
    _export_mlp(tensors,'model._learned_model.decoder',model.decoder)
    for name in ('model.nodetype_embedding.weight',):
        tensors[name]=_tensor_bytes(teacher.require(name))
    for name in ('node','mesh_edge','world_edge','output'):
        for suffix in ('_acc_count','_acc_sum','_acc_sum_squared'):
            key=f'model._{name}_normalizer.{suffix}'
            tensors[key]=_tensor_bytes(teacher.require(key))
    for i in (0,2):
        for name in ('weight','bias'):
            tensors[f'temporal.layers.{i}.{name}']=_tensor_bytes(getattr(model.temporal_encoder[i],name))
    contract=list(CONTRACT);contract[7]=int(model.current_only)
    tensors['temporal.contract']=_tensor_bytes(torch.tensor(contract,dtype=torch.float32))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Reuse the legacy tensor writer only for packing; legacy readers reject the final magic.
    with tempfile.TemporaryDirectory(dir=path.parent) as work:
        tmp=Path(work)/'packed.vhood'
        write_tensor_asset(tmp,tensors,checkpoint_sha256=checkpoint_sha256)
        packed=bytearray(tmp.read_bytes());packed[:8]=MAGIC
        tmp.write_bytes(packed);load_asset(tmp);tmp.replace(path)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'contract':contract}

def load_model(path,device='cpu'):
    asset=load_asset(path)
    contract=np.frombuffer(asset.require('temporal.contract').data,dtype='<f4')
    model=TemporalTinyHood(bool(contract[7]))
    for prefix,mlp in [('node_encoder',model.node_encoder),('edgeset_encoders.mesh',model.mesh_encoder),('edgeset_encoders.world',model.world_encoder),('decoder',model.decoder)]:
        _load_mlp(asset,'model._learned_model.'+prefix,mlp)
    for i,block in enumerate(model.processor_steps):
        for name in ('mesh_edge_processor','world_edge_processor','node_processor'):
            _load_mlp(asset,f'model._learned_model.processor_steps.{i}.{name}',getattr(block,name))
    with torch.no_grad():
        for i in (0,2):
            for name in ('weight','bias'):
                parameter=getattr(model.temporal_encoder[i],name)
                view=asset.require(f'temporal.layers.{i}.{name}',parameter.shape)
                parameter.copy_(torch.from_numpy(np.frombuffer(view.data,dtype='<f4').copy()).reshape(parameter.shape))
    return model.to(device)
