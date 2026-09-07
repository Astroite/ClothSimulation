from pathlib import Path
from types import SimpleNamespace
import unittest,sys,tempfile
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from temporal.model import TemporalTinyHood,GnnHistoryState,normalize_history
from temporal.format import export_model,load_model,load_asset
from real_scene.tinyhood import load_tinyhood
from real_scene.fine15 import Fine15Weights

LEGACY=ROOT.parent/'vulkan-gnn-poc/.work/hood_data/student32x12_r1.vhood'
def graph(n=5,m=3):
    gen=torch.Generator().manual_seed(71)
    send=torch.arange(n);recv=(send+1)%n
    return SimpleNamespace(cloth_nodes=torch.randn(n,20,generator=gen),obstacle_nodes=torch.randn(m,20,generator=gen),
        mesh_edges=torch.randn(n,12,generator=gen),mesh_senders=send,mesh_receivers=recv,
        direct_world=torch.randn(n,9,generator=gen),inverse_world=torch.randn(n,9,generator=gen),
        world_cloth=torch.arange(n),world_obstacle=torch.arange(n)%m)

class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_zero_branch_and_variable_nodes(self):
        old=load_tinyhood(LEGACY);new=TemporalTinyHood.from_legacy(LEGACY)
        for n in (3,7,21):
            g=graph(n);h=torch.randn(n+3,28)
            torch.testing.assert_close(old(g),new(g,h),atol=1e-5,rtol=1e-4)
    def test_history_padding_clone_reset_and_gradient(self):
        history=GnnHistoryState();d=torch.randn(5,3,requires_grad=True);n=torch.randn(5,3)
        h=history.push(d,n);self.assertEqual(h[:,:,6].tolist(),[[0,0,0,1]]*5)
        saved=history.clone();history.push(d*2,n);self.assertEqual(saved.ticks,1)
        self.assertEqual(history.samples[:,:,6].tolist(),[[0,0,1,1]]*5)
        history.samples[:,:,:3].sum().backward();self.assertTrue(torch.isfinite(d.grad).all())
        history.reset();self.assertIsNone(history.samples)
    def test_history_changes_output_and_current_ablation(self):
        torch.manual_seed(19);model=TemporalTinyHood();torch.nn.init.normal_(model.temporal_encoder[2].weight,std=.02)
        g=graph();h=torch.randn(8,28);h2=h.clone();h2[:,:21]+=3
        self.assertGreater((model(g,h)-model(g,h2)).abs().max().item(),1e-7)
        model.current_only=True;torch.testing.assert_close(model(g,h),model(g,h2),atol=0,rtol=0)
    def test_permutation_equivariance(self):
        model=TemporalTinyHood.from_legacy(LEGACY);torch.nn.init.normal_(model.temporal_encoder[2].weight,std=.002)
        g=graph();h=torch.randn(8,28);perm=torch.tensor([3,1,4,0,2]);inv=torch.argsort(perm)
        bperm=torch.tensor([2,0,1]);binv=torch.argsort(bperm)
        gp=SimpleNamespace(**vars(g));gp.cloth_nodes=g.cloth_nodes[perm];gp.obstacle_nodes=g.obstacle_nodes[bperm]
        gp.mesh_senders=inv[g.mesh_senders];gp.mesh_receivers=inv[g.mesh_receivers];gp.world_cloth=inv[g.world_cloth];gp.world_obstacle=binv[g.world_obstacle]
        hp=torch.cat((h[:5][perm],h[5:][bperm]));torch.testing.assert_close(model(g,h)[perm],model(gp,hp),atol=1e-5,rtol=1e-4)
    def test_export_roundtrip_and_checksum(self):
        model=TemporalTinyHood.from_legacy(LEGACY);model.current_only=True
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.vthood';export_model(model,Fine15Weights.from_vhood(LEGACY),path)
            loaded=load_model(path);self.assertTrue(loaded.current_only)
            for key,v in model.state_dict().items():self.assertTrue(torch.equal(v,loaded.state_dict()[key]),key)
            data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'checksum'):load_asset(path)
if __name__=='__main__':unittest.main()
