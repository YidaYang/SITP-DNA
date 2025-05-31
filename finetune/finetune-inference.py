import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math

# 尝试导入 Evo2 和相关组件
try:
    from evo2 import Evo2
    from vortex.model.layers import RMSNorm # 用于猴子补丁
    # from vortex.model.tokenizer import CharLevelTokenizer # 此脚本核心逻辑不直接使用
except ImportError:
    print("Evo2 库未找到。请确保已正确安装。")
    print("您可能需要在 evo2-main 目录下运行: pip install .")
    exit()

# --- 1. LoRA 层定义 (与 finetune.py 对齐) ---
class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # 与 finetune.py 的 LoRA 参数 dtype 逻辑对齐
        model_dtype_for_lora = getattr(original_layer.weight, 'dtype', torch.float32)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8: # Ampere 架构及以上
            # finetune.py 在此条件下将 LoRA 参数设置为 bfloat16
            model_dtype_for_lora = torch.bfloat16

        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank, dtype=model_dtype_for_lora))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features, dtype=model_dtype_for_lora))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B 初始化为零

        self.scaling = (self.alpha / self.rank)
        if model_dtype_for_lora == torch.bfloat16:
            self.scaling = torch.tensor(self.scaling, dtype=torch.bfloat16)

        # 冻结原始层权重
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cloned_weight = self.original_layer.weight.clone()
        cloned_bias = self.original_layer.bias.clone() if self.original_layer.bias is not None else None
        original_output = F.linear(x, cloned_weight, cloned_bias)

        if self.rank > 0:
            current_scaling = self.scaling
            if isinstance(self.scaling, torch.Tensor):
                current_scaling = self.scaling.to(device=x.device, dtype=x.dtype)
            else:
                current_scaling = torch.tensor(self.scaling, dtype=x.dtype, device=x.device)
            
            lora_A_final = self.lora_A.to(x.dtype) # 确保与输入x的dtype匹配
            lora_B_final = self.lora_B.to(x.dtype) # 确保与输入x的dtype匹配
            
            lora_addition = (x @ lora_A_final @ lora_B_final)
            lora_output = lora_addition * current_scaling
            return original_output + lora_output
        
        return original_output

    def extra_repr(self):
        return f'original_in_features={self.in_features}, original_out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}'


# --- 2. 将LoRA应用于模型 (与 finetune.py 对齐) ---
def apply_lora_to_evo2_model(evo2_base_module: nn.Module, lora_rank: int, lora_alpha: int, target_module_patterns: list[str]):
    lora_applied_count = 0
    for name, module in evo2_base_module.named_modules():
        if isinstance(module, nn.Linear):
            for pattern in target_module_patterns:
                if re.fullmatch(pattern, name):
                    try:
                        parent_name, child_name = name.rsplit('.', 1)
                        parent_module = evo2_base_module.get_submodule(parent_name)
                        
                        original_linear_layer = getattr(parent_module, child_name)
                        if not isinstance(original_linear_layer, LoRALayer): # 避免重复应用
                            lora_replacement = LoRALayer(original_linear_layer, lora_rank, lora_alpha)
                            setattr(parent_module, child_name, lora_replacement)
                            lora_applied_count += 1
                            print(f"已将LoRA应用于: {name}")
                        else:
                            print(f"LoRA已存在于: {name}, 跳过。")
                        break # 当前模块已匹配一个模式，处理下一个模块
                    except Exception as e:
                        print(f"为 {name} 应用LoRA失败: {e}")
    
    if lora_applied_count == 0:
        print("警告: 没有层与目标LoRA模式匹配。")
    else:
        print(f"总共将LoRA应用于 {lora_applied_count} 个层。")
    return lora_applied_count


# --- 主推理函数 ---
def main_inference():
    # --- 配置参数 (直接从 finetune.py 获取或设定合理的默认值) ---
    base_model_name = "evo2_7b"
    local_model_path = '/root/autodl-tmp/evo2/models/evo2-7b/evo2_7b.pt' # 与 finetune.py 一致
    
    # output_dir 和 lora_weights_file 的构建方式与 finetune.py 一致
    output_dir_finetune = "evo2_finetuned_enhancer_lora" # finetune.py 中的 output_dir
    lora_weights_file = os.path.join(output_dir_finetune, f"{base_model_name}_enhancer_lora_weights.pt")

    lora_rank = 8
    lora_alpha = 16
    target_module_patterns = [
        r"blocks\.\d+\.mlp\.l1",
        r"blocks\.\d+\.mlp\.l3",
        r"blocks\.\d+\.inner_mha_cls\.Wqkv",
    ]
    
    # 推理特定参数 (可以根据需要修改这些值)
    prompt_seqs_list = ["GATTACA", "TTCCGGAA", "A", "T", "C", "G", ""] # 示例提示序列列表
    num_tokens_to_generate = 100
    output_file = "generated_sequences_inference.txt" # 推理结果的输出文件
    temperature = 0.8
    top_k = 40
    # --- 配置参数结束 ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载基础模型 ---
    print(f"加载基础模型: {base_model_name}...")
    try:
        evo2_wrapper = Evo2(model_name=base_model_name, local_path=local_model_path)
    except Exception as e:
        print(f"加载Evo2模型 '{base_model_name}' 失败: {e}")
        if local_model_path:
            print(f"请检查模型名称和路径 '{local_model_path}'。")
        return
    
    evo2_base_module = evo2_wrapper.model # 这是 StripedHyena nn.Module

    # --- 应用猴子补丁 (与 finetune.py 一致) ---
    # RMSNorm 补丁
    if 'RMSNorm' in globals() and hasattr(RMSNorm, 'forward'):
        print("尝试修补 RMSNorm.forward...")
        def patched_rmsnorm_forward(self_rmsnorm, x_input: torch.Tensor) -> torch.Tensor:
            variance = x_input.float().pow(2).mean(-1, keepdim=True)
            rsqrt_term = torch.rsqrt(variance + self_rmsnorm.eps)
            normalized_and_typed_x = (x_input * rsqrt_term).type_as(x_input)
            if not hasattr(self_rmsnorm, 'scale'):
                print("警告: RMSNorm 实例没有补丁所期望的 'scale' 属性。此补丁可能失败或行为不正确。")
            cloned_scale_param = self_rmsnorm.scale.clone()
            return cloned_scale_param * normalized_and_typed_x

        RMSNorm.forward = patched_rmsnorm_forward
        print("RMSNorm.forward 方法已成功修补。")
    else:
        print("RMSNorm 类未找到或没有 'forward' 属性，跳过 RMSNorm 补丁。")

    # Unembed 补丁
    if hasattr(evo2_base_module, 'unembed') and \
       hasattr(evo2_base_module.unembed, 'func') and \
       hasattr(evo2_base_module.unembed.func, '__self__'):
        print("尝试修补 unembed 函数...")
        embedding_module_instance = evo2_base_module.unembed.func.__self__
        if not hasattr(embedding_module_instance, 'weight'):
            print("警告: 用于 unembed 补丁的目标嵌入模块没有 'weight' 属性。跳过补丁。")
        else:
            def patched_unembed_logic(u_tensor: torch.Tensor) -> torch.Tensor:
                if not hasattr(embedding_module_instance, 'process_group') or embedding_module_instance.process_group is None:
                    cloned_weight = embedding_module_instance.weight.clone()
                    return u_tensor @ cloned_weight.T
                else:
                    print(f"已修补的 unembed 在 {type(embedding_module_instance).__name__} 上调用，并带有 process_group。此情况按原始逻辑引发 NotImplementedError。")
                    raise NotImplementedError("VocabParallelEmbedding 中未实现带 process_group 的 Unembed。")
            
            evo2_base_module.unembed.func = patched_unembed_logic
            print(f"已成功为 {type(embedding_module_instance).__name__} 修补 'unembed.func'。")
    else:
        print("警告: 未能找到 'evo2_base_module.unembed.func' 或其 '__self__' 属性进行修补。跳过 unembed 补丁。")
    # --- 猴子补丁结束 ---

    # --- 冻结原始参数并应用LoRA结构 ---
    print("冻结原始模型参数并应用LoRA结构...")
    for param in evo2_base_module.parameters():
        param.requires_grad = False 
    
    apply_lora_to_evo2_model(evo2_base_module, lora_rank, lora_alpha, target_module_patterns)

    # --- 加载LoRA权重 ---
    print(f"从以下位置加载LoRA权重: {lora_weights_file}")
    if not os.path.exists(lora_weights_file):
        print(f"错误: LoRA权重文件未在 {lora_weights_file} 找到。")
        print(f"请确保 '{output_dir_finetune}' 目录存在，并且包含名为 '{os.path.basename(lora_weights_file)}' 的LoRA权重。")
        print(f"此文件应由 finetune.py 脚本在路径 '{output_dir_finetune}' 中生成。")
        return

    try:
        lora_weights_state_dict = torch.load(lora_weights_file, map_location=device)
        
        filtered_lora_weights = {
            k: v for k, v in lora_weights_state_dict.items() if 'lora_A' in k or 'lora_B' in k
        }
        
        if not filtered_lora_weights:
            print(f"警告: 在 {lora_weights_file} 中未找到包含 'lora_A' 或 'lora_B' 的键。")
            print("请确保LoRA权重文件包含 LoRALayer 实例的参数。")
        else:
            print(f"在权重文件中找到 {len(filtered_lora_weights)} 个LoRA参数。")

        missing_keys, unexpected_keys = evo2_base_module.load_state_dict(filtered_lora_weights, strict=False)
        
        if missing_keys:
             print(f"警告: 加载LoRA权重时丢失键: {missing_keys}")
             print("这可能表明LoRA权重文件不完整或与模型结构不完全匹配。")
        if unexpected_keys:
             print(f"警告: 加载LoRA权重时出现意外键: {unexpected_keys}")
        
        if not missing_keys : 
            print("LoRA权重加载成功。")
        else:
            print("LoRA权重加载完成，但存在一些问题 (见上述警告)。")

    except Exception as e: # 更通用的异常捕获，以防 torch.load 失败等
        print(f"加载LoRA权重时出错: {e}")
        return

    evo2_base_module.to(device)
    evo2_base_module.eval() 

    # --- 生成 ---
    print(f"\n为 {len(prompt_seqs_list)} 个提示序列生成...")

    all_generated_sequences_text = []
    # 确保输出目录存在 (如果 output_file 包含目录)
    output_file_dir = os.path.dirname(output_file)
    if output_file_dir and not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    for i, prompt_sequence in enumerate(prompt_seqs_list):
        print(f"提示 {i+1}/{len(prompt_seqs_list)}: '{prompt_sequence}'")
        print(f"生成 {num_tokens_to_generate} 个 token...")
        
        with torch.no_grad():
            try:
                generated_output = evo2_wrapper.generate(
                    prompt_seqs=[prompt_sequence], 
                    n_tokens=num_tokens_to_generate,
                    temperature=temperature,
                    top_k=top_k,
                    batched=False 
                )
                
                if generated_output and generated_output.sequences:
                    final_generated_sequence = generated_output.sequences[0]
                    print(f"生成的序列 (长度 {len(final_generated_sequence)}):\n{final_generated_sequence}\n")
                    all_generated_sequences_text.append(f">prompt:{prompt_sequence}\n{final_generated_sequence}\n")
                else:
                    print("此提示的生成失败或未产生输出。")
                    all_generated_sequences_text.append(f">prompt:{prompt_sequence}\n[生成失败]\n")

            except Exception as e:
                print(f"为提示 '{prompt_sequence}' 生成时出错: {e}")
                all_generated_sequences_text.append(f">prompt:{prompt_sequence}\n[生成错误: {e}]\n")

    # --- 保存结果 ---
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for text_entry in all_generated_sequences_text:
                f.write(text_entry + "\n") 
        print(f"所有生成的序列已保存至: {output_file}")
    except Exception as e:
        print(f"将生成的序列保存到 {output_file} 时出错: {e}")

    print("\n推理完成。")

if __name__ == "__main__":
    main_inference() 