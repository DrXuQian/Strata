"""Host contract checks; these do not execute or certify PPU kernels."""
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from strata_ppu import adapter, jit


class PackageContracts(unittest.TestCase):
    def setUp(self):
        adapter._modules.clear()
        adapter._warmed.clear()
        adapter._layouts.clear()

    def test_flashattention_out_contract(self):
        q = torch.ones((1, 1, 24, 256), dtype=torch.float16)
        k = torch.ones((1, 13, 2, 256), dtype=q.dtype)
        out = torch.empty_like(q)
        def fwd(q_, k_, v_, out_, alibi, dropout, scale, causal, left, right, softcap, softmax, generator):
            self.assertIs(q_, q)
            self.assertIs(k_, k)
            self.assertIs(out_, out)
            self.assertEqual((alibi, dropout, scale, causal, left, right, softcap, softmax, generator),
                             (None, 0.0, 1/16, False, -1, -1, 0.0, False, None))
            out_.fill_(3)
            return out_, None, None, None
        adapter._modules["fa"] = types.SimpleNamespace(fwd=fwd)
        adapter._execute("fa", (q,k,k,out), (), 7, False)
        self.assertTrue(torch.equal(out, torch.full_like(out,3)))

    def test_flashattention_replaced_output_rejected(self):
        q = torch.zeros((1,1,24,256),dtype=torch.float16)
        adapter._modules["fa"] = types.SimpleNamespace(fwd=lambda *args: (q.clone(),))
        with self.assertRaisesRegex(RuntimeError, "caller-owned"):
            adapter._execute("fa",(q,q,q,q),(),0,False)

    def test_deepgemm_signature_and_cold_capture(self):
        a=torch.ones((3,256),dtype=torch.bfloat16)
        w=torch.ones((1,512,256),dtype=a.dtype)
        out=torch.empty((3,512),dtype=a.dtype)
        ids=torch.zeros(3,dtype=torch.int32)
        rows=torch.tensor([3],dtype=torch.int32)
        calls=[]
        def nopad(lhs,rhs,output,m_indices,*,m_rows):
            self.assertIs(m_rows,rows)
            self.assertIs(m_indices,ids)
            self.assertIs(output,out)
            output.fill_(5)
            calls.append(1)
        adapter._modules["moe"]=nopad
        with self.assertRaisesRegex(RuntimeError,"eager"):
            adapter._execute("moe",(a,w,out,ids,rows),(),0,True)
        self.assertFalse(calls)
        adapter._execute("moe",(a,w,out,ids,rows),(),0,False)
        adapter._execute("moe",(a,w,out,ids,rows),(),0,True)
        self.assertEqual(len(calls),2)

    def test_cold_import_never_runs_in_capture(self):
        with patch.object(importlib,"import_module",side_effect=AssertionError("unexpected import")):
            for name in ("fa","moe","gdn","quact:12"):
                with self.assertRaisesRegex(RuntimeError,"before graph capture"):
                    adapter._module(name,True)

    def test_loaded_layout_query_accepts_dynamic_expert_capacity(self):
        def layout(n,k,e):
            return (2,3,4,0,0,64,32,0,0x51344B5034540001,144*n*k//256*e,
                    n*k//2*e,0,n*k//16*e)
        adapter._modules["quact:12"]=types.SimpleNamespace(layout=layout)
        values=adapter._quact_layout(12,1280,2560,8,True)
        self.assertEqual(sum(values[10:]),values[9])

    def test_layout_mismatch_is_fatal(self):
        adapter._modules["quact:12"]=types.SimpleNamespace(layout=lambda *a:
            (2,3,4,0,0,64,32,0,1,100,60,0,39))
        with self.assertRaisesRegex(RuntimeError,"byte-neutral"):
            adapter._quact_layout(12,256,512,1,False)

    def test_upgrade_and_image_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"headers").mkdir()
            header=root/"headers/kernel.cuh"
            header.write_text("kernel version 1")
            before=jit.tree_hash(root,["headers"])
            header.write_text("kernel version 2")
            self.assertNotEqual(before,jit.tree_hash(root,["headers"]))
            image=root/"binding.so";image.write_bytes(b"image 1")
            record={"source":before}
            jit.publish(root,image,record)
            self.assertTrue(jit.valid(root,image,record))
            self.assertFalse(jit.valid(root,image,{"source":"upgrade"}))
            image.write_bytes(b"corrupt")
            self.assertFalse(jit.valid(root,image,record))

    def test_host_wheel_without_sources_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            module=types.SimpleNamespace(__file__=str(Path(directory)/"quactlize/__init__.py"))
            with patch.object(importlib,"import_module",return_value=module):
                with self.assertRaisesRegex(RuntimeError,"pip install --no-build-isolation -e"):
                    jit.source_root("quactlize",["quactlize/packing/word_pack.hpp"])


if __name__=="__main__":
    unittest.main()
