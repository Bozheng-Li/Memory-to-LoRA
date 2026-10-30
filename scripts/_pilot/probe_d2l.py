import os, sys, time, torch
ROOT="/home/libaizheng/Lora-Grpo-memory"
SRC=f"{ROOT}/resources/doc-to-lora/src"
sys.path.insert(0, SRC); os.chdir(f"{ROOT}/resources/doc-to-lora")
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.model_loading import get_tokenizer
t0=time.time()
sd=torch.load(f"{ROOT}/resources/checkpoints/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin", weights_only=False, map_location="cpu")
m=ModulatedPretrainedModel.from_state_dict(sd, train=False, use_flash_attn=False, use_sequence_packing=False)
m.eval(); m.reset()
print(f"load {time.time()-t0:.1f}s | {torch.cuda.memory_allocated()/2**30:.1f} GiB")
tok=get_tokenizer(m.base_model.name_or_path)
doc=("Alice moved to Berlin in 2019. She works as a data engineer at Helios. "
     "Her favourite food is ramen. She adopted a cat named Miso in 2021. ")*12
print("doc tokens", len(tok(doc)["input_ids"]))
t1=time.time(); m.internalize(doc); print(f"internalize {time.time()-t1:.2f}s | peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
q=[{"role":"user","content":"Where does Alice work?"}]
ids=tok.apply_chat_template(q, add_special_tokens=False, return_attention_mask=False, add_generation_prompt=True, return_tensors="pt").to(m.device)
t2=time.time()
out=m.generate(input_ids=ids, max_new_tokens=32, do_sample=False)
print("answer:", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip(), f"| gen {time.time()-t2:.2f}s")
