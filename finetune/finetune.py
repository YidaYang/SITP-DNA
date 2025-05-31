import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm
import re
import math

# 尝试导入Evo2相关组件
try:
    from evo2 import Evo2
    from vortex.model.layers import RMSNorm # ADDED IMPORT for monkey-patching
    # from vortex.model.model import StripedHyena # Evo2内部使用的模型
    # from vortex.model.tokenizer import CharLevelTokenizer # Evo2使用的分词器
except ImportError:
    print("Evo2库未找到。请确保已正确安装。")
    print("您可能需要在evo2-main目录下运行: pip install .")
    exit()

# --- 1. LoRA 层定义 ---
class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # LoRA权重，A的初始化方式参考原始LoRA论文，B初始化为0
        # 确保LoRA权重与模型可能使用的数据类型 (如bfloat16) 兼容
        model_dtype = getattr(original_layer.weight, 'dtype', torch.float32) # 获取原始层权重的数据类型
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8: # Ampere架构及以上支持bfloat16
            # 如果原始模型是bfloat16，或者我们希望强制使用bfloat16
            # 这里我们假设如果模型输入是bfloat16，原始权重也会是或可以转换
            # 为了与错误信息中的 BFloat16 匹配，这里可以更明确地设置
            # 如果原始层就是bfloat16，则model_dtype会是bfloat16
            # 否则，如果x是bfloat16，我们需要确保LoRA层也是。
            # 一个更稳健的方法是传递期望的dtype或从模型配置中获取。
            # 但根据错误，输入x是bfloat16，所以lora层也必须是。
            model_dtype = torch.bfloat16

        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank, dtype=model_dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features, dtype=model_dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 初始化为零，使得初始时LoRA旁路输出为0

        self.scaling = (self.alpha / self.rank)
        if model_dtype == torch.bfloat16:
            self.scaling = torch.tensor(self.scaling, dtype=torch.bfloat16)


        # 冻结原始层权重
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clone weight and bias from the original layer to ensure they are "normal tensors"
        cloned_weight = self.original_layer.weight.clone()
        cloned_bias = self.original_layer.bias.clone() if self.original_layer.bias is not None else None
        
        # Perform the original linear operation using F.linear with cloned tensors
        original_output = F.linear(x, cloned_weight, cloned_bias)

        if self.rank > 0:
            # Handle scaling's dtype and device
            # self.scaling can be a Python float or a bfloat16 tensor (if model_dtype was bfloat16)
            if isinstance(self.scaling, torch.Tensor):
                # If self.scaling is already a tensor, ensure it's on the correct device and matches x's dtype
                current_scaling = self.scaling.to(device=x.device, dtype=x.dtype)
            else:
                # If self.scaling is a Python float, convert it to a tensor on x's device and dtype
                current_scaling = torch.tensor(self.scaling, dtype=x.dtype, device=x.device)
            
            # self.lora_A and self.lora_B are nn.Parameters.
            # They are moved to x.device when the LoRALayer module is moved.
            # Ensure their dtype matches x.dtype for matrix multiplication.
            lora_A_final = self.lora_A.to(x.dtype)
            lora_B_final = self.lora_B.to(x.dtype)
            
            # Perform LoRA path operations
            lora_addition = (x @ lora_A_final @ lora_B_final)
            lora_output = lora_addition * current_scaling
            return original_output + lora_output
        
        return original_output

    def extra_repr(self):
        return f'original_in_features={self.in_features}, original_out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}'


# --- 2. 将LoRA应用于Evo2模型的函数 ---
def apply_lora_to_evo2_model(evo2_base_module: nn.Module, lora_rank: int, lora_alpha: int, target_module_patterns: list[str]):
    """
    将LoRA层应用于Evo2基础模型 (StripedHyena) 的指定模块。
    模型会地修改。
    """
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


# --- 3. 数据集定义 ---
class EnhancerDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, max_length: int = 512, min_length: int = 50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        try:
            df = pd.read_csv(csv_file)
            if 'sequence' not in df.columns:
                raise ValueError("CSV文件必须包含 'sequence' 列。")

            for seq in df['sequence']:
                seq_str = str(seq).upper() # 确保为大写DNA序列
                # 过滤掉非标准DNA字符的序列 (可选，但推荐)
                if not all(c in 'ACGTN' for c in seq_str):
                    print(f"警告: 序列 '{seq_str[:30]}...' 包含非标准DNA字符，已跳过。")
                    continue
                if len(seq_str) >= min_length:
                    self.sequences.append(seq_str)
            
            if not self.sequences:
                raise ValueError(f"未加载任何序列。请检查CSV内容、min_length ({min_length}) 和序列有效性。")
            print(f"从 {csv_file} 加载了 {len(self.sequences)} 条序列 (最小长度 {min_length})。")

        except FileNotFoundError:
            print(f"错误: CSV文件未在 {csv_file} 找到。")
            raise
        except Exception as e:
            print(f"读取或处理CSV文件时出错: {e}")
            raise

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        token_ids = self.tokenizer.tokenize(seq)

        # 截断或填充到max_length
        # 对于因果语言模型，输入和标签通常是相同的（或移位），模型内部处理移位以进行预测
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            padding_length = self.max_length - len(token_ids)
            # Evo2的CharLevelTokenizer的pad_id默认为1
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 1) 
            token_ids.extend([pad_token_id] * padding_length)

        input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        # 标签与输入相同，损失函数将处理移位和忽略填充
        labels_tensor = input_ids_tensor.clone()

        return {"input_ids": input_ids_tensor, "labels": labels_tensor}


# --- 4. 训练周期函数 ---
def finetune_epoch(
    evo2_wrapper: Evo2, 
    data_loader: DataLoader, 
    optimizer: AdamW, 
    device: torch.device, 
    grad_accumulation_steps: int = 1
):
    evo2_wrapper.model.train() # 设置底层模型为训练模式
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(data_loader, desc="微调周期", leave=False)
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Evo2的底层StripedHyena模型返回 (logits, inference_params_dict_out)
        # 我们只关心 logits 进行损失计算
        try:
            # StripedHyena.forward(self, x, inference_params_dict=None, padding_mask=None)
            # 在训练时，通常不需要 inference_params_dict 和 padding_mask (除非有特殊处理)
            output_from_model, _ = evo2_wrapper.model(input_ids) # 直接调用 underlying model
        except Exception as e:
            print(f"模型前向传播失败: {e}")
            # 可以考虑添加更详细的错误处理或回退机制，但目前保持简单
            raise

        # output_from_model 包含 logits
        # logits: (batch, seq_len, vocab_size), labels: (batch, seq_len)
        # CharLevelTokenizer 确实有 pad_id = 1
        pad_token_id = getattr(evo2_wrapper.tokenizer, 'pad_id', 1) # 确保使用 'pad_id'

        loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)(
            output_from_model.view(-1, output_from_model.size(-1)),
            labels.view(-1)
        )

        if loss is not None:
            loss = loss / grad_accumulation_steps
            loss.backward()

            if (i + 1) % grad_accumulation_steps == 0 or i == len(data_loader) - 1:
                torch.nn.utils.clip_grad_norm_([p for p in evo2_wrapper.model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accumulation_steps
            progress_bar.set_postfix(loss=loss.item() * grad_accumulation_steps)
        else:
            print("警告: 未能计算损失。")


    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    return avg_loss


# --- 5. 主微调函数 ---
def finetune_evo2_for_enhancer_generation():
    # --- 配置 ---
    base_model_name = "evo2_7b"  # 从较小模型开始，便于快速迭代
    # 可选: "evo2_7b_base", "evo2_40b_base" (需要更多显存和时间)
    # 如果使用1M上下文长度的模型如 "evo2_7b", 确保有足够显存
    
    # local_model_path = None # 如果模型已下载到本地，请指定 .pt 文件路径
    local_model_path = '/root/autodl-tmp/evo2/models/evo2-7b/evo2_7b.pt' # 示例路径

    csv_file = "data.csv"  # 包含增强子DNA序列的CSV文件路径
                           # 列名应为 'sequence'

    output_dir = "evo2_finetuned_enhancer_lora"
    lora_weights_file = os.path.join(output_dir, f"{base_model_name}_enhancer_lora_weights.pt")

    # LoRA 参数
    lora_rank = 8
    lora_alpha = 16
    # 目标模块 (使用正则表达式) - 参考 evo2_layers.txt
    # 示例: MLP中的线性层, 注意力中的Wqkv层
    target_module_patterns = [
        r"blocks\.\d+\.mlp\.l1",
        r"blocks\.\d+\.mlp\.l3",
        r"blocks\.\d+\.inner_mha_cls\.Wqkv",
        # r"blocks\.\d+\.inner_mha_cls\.out_proj", # 可选
        # r"blocks\.\d+\.filter\.projections\.\d+", # Hyena filter中的线性层 (如果存在且希望微调)
    ]

    # 训练参数
    max_seq_length = 256  # 根据增强子典型长度和GPU显存调整
    min_enhancer_length = 1 # 过滤掉过短的序列
    batch_size = 2        # 根据GPU显存调整 (1B模型在256长度下，batch_size=2可能适合24GB显存)
    epochs = 3            # 微调周期数
    learning_rate = 5e-5  # LoRA参数的学习率
    grad_accumulation_steps = 8 # 有效批次大小 = batch_size * grad_accumulation_steps

    os.makedirs(output_dir, exist_ok=True)

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 加载基础模型和分词器 ---
    print(f"加载基础模型: {base_model_name}...")
    try:
        evo2_wrapper = Evo2(model_name=base_model_name, local_path=local_model_path)
    except Exception as e:
        print(f"加载Evo2模型 '{base_model_name}' 失败: {e}")
        print(f"请检查模型名称和路径 '{local_model_path}' 是否正确。")
        return

    tokenizer = evo2_wrapper.tokenizer
    evo2_base_module = evo2_wrapper.model # 这是 StripedHyena nn.Module

    # --- Monkey-patch RMSNorm to avoid 'Inference tensors cannot be saved for backward' --- START
    if 'RMSNorm' in globals() and hasattr(RMSNorm, 'forward'):
        print("Attempting to patch RMSNorm.forward...")

        def patched_rmsnorm_forward(self_rmsnorm, x_input: torch.Tensor) -> torch.Tensor:
            """
            Patched RMSNorm forward pass.
            This replicates a common robust RMSNorm implementation and uses self_rmsnorm.weight.clone()
            to prevent issues with 'inference tensors' during backward pass when LoRA is active.
            Original RMSNorm in some Evo2/Vortex versions might involve:
            y = (x_input * torch.rsqrt(x_input.float().pow(2).mean(-1, keepdim=True) + self_rmsnorm.eps)).type_as(x_input)
            return self_rmsnorm.weight * y
            This patch changes the multiplication to use self_rmsnorm.weight.clone().
            """
            # Calculate variance and rsqrt_term using float32 for numerical stability
            variance = x_input.float().pow(2).mean(-1, keepdim=True)
            # self_rmsnorm.eps is typically a Python float, e.g., 1e-5 or 1e-6
            rsqrt_term = torch.rsqrt(variance + self_rmsnorm.eps)

            # Normalize x_input and cast back to its original dtype
            # x_input (e.g., bfloat16) * rsqrt_term (float32) results in float32
            # .type_as(x_input) casts it back to the original dtype (e.g., bfloat16)
            normalized_and_typed_x = (x_input * rsqrt_term).type_as(x_input)

            # Get the learnable scaling parameter (standard name is 'weight') and clone it
            cloned_weight = self_rmsnorm.scale.clone()

            return cloned_weight * normalized_and_typed_x

        RMSNorm.forward = patched_rmsnorm_forward
        print("RMSNorm.forward method has been successfully patched.")
    else:
        print("RMSNorm class not found or does not have 'forward' attribute, skipping patch.")
    # --- Monkey-patch RMSNorm --- END

    # --- Monkey-patch for Unembed to fix 'Inference tensors cannot be saved for backward' --- START
    if hasattr(evo2_base_module, 'unembed') and \
       hasattr(evo2_base_module.unembed, 'func') and \
       hasattr(evo2_base_module.unembed.func, '__self__'):

        print("尝试修补 unembed 函数以解决 inference tensor 问题...")
        embedding_module_instance = evo2_base_module.unembed.func.__self__

        if not hasattr(embedding_module_instance, 'weight'):
            print("警告: 用于 unembed 补丁的目标嵌入模块没有 'weight' 属性。跳过补丁。")
        else:
            # original_unembed_bound_method = evo2_base_module.unembed.func

            # 这个新函数将替换 evo2_base_module.unembed.func
            # ForwardAbs 将使用参数 'u' 来调用此函数
            def patched_unembed_logic(u_tensor: torch.Tensor) -> torch.Tensor:
                # embedding_module_instance 从外部作用域捕获
                # 此逻辑基于 VocabParallelEmbedding.unembed
                # 检查 process_group 属性是否存在，否则假定其行为类似于 'None' 的情况
                if not hasattr(embedding_module_instance, 'process_group') or embedding_module_instance.process_group is None:
                    cloned_weight = embedding_module_instance.weight.clone()
                    # 原始错误发生在 u @ self.weight.T
                    return u_tensor @ cloned_weight.T
                else:
                    # 对于 'else' 情况，复制原始行为 (NotImplementedError)
                    # VocabParallelEmbedding.unembed 在这种情况下会引发 NotImplementedError
                    print(f"已修补的 unembed 在 {type(embedding_module_instance).__name__} 上调用，并带有 process_group。此情况按原始逻辑引发 NotImplementedError。")
                    raise NotImplementedError("VocabParallelEmbedding 中未实现带 process_group 的 Unembed。")

            evo2_base_module.unembed.func = patched_unembed_logic
            print(f"已成功为 {type(embedding_module_instance).__name__} 修补 'unembed.func' 以处理 inference tensor。")
    else:
        print("警告: 未能找到 'evo2_base_module.unembed.func' 或其 '__self__' 属性进行修补。跳过 unembed 补丁。")
    # --- Monkey-patch for Unembed --- END

    # --- 冻结所有原始参数并应用LoRA ---
    print("冻结原始模型参数并应用LoRA...")
    for param in evo2_base_module.parameters():
        param.requires_grad = False
    
    apply_lora_to_evo2_model(evo2_base_module, lora_rank, lora_alpha, target_module_patterns)

    # 收集可训练的LoRA参数
    lora_parameters_to_optimize = [
        p for p in evo2_base_module.parameters() if p.requires_grad
    ]
    
    if not lora_parameters_to_optimize:
        print("错误: 未找到可训练的LoRA参数。请检查target_module_patterns。")
        return
    else:
        print(f"找到 {len(lora_parameters_to_optimize)} 组可训练的LoRA参数。")

    evo2_base_module.to(device) # 将修改后的模型移至设备

    # --- 数据集和DataLoader ---
    print("加载数据集...")
    try:
        dataset = EnhancerDataset(csv_file, tokenizer, max_length=max_seq_length, min_length=min_enhancer_length)
        if len(dataset) == 0: return # EnhancerDataset内部会打印错误
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"加载或处理数据集失败: {e}")
        return
    print(f"数据集加载完成: {len(dataset)} 个样本用于微调。")

    # --- 优化器 ---
    optimizer = AdamW(lora_parameters_to_optimize, lr=learning_rate)

    # --- 训练循环 ---
    print("开始LoRA微调...")
    for epoch in range(epochs):
        avg_train_loss = finetune_epoch(
            evo2_wrapper, train_loader, optimizer, device, grad_accumulation_steps
        )
        print(f"周期 {epoch+1}/{epochs}: 平均训练损失 = {avg_train_loss:.4f}")

        # 保存LoRA权重
        lora_state_dict = {
            k: v for k, v in evo2_base_module.state_dict().items() if 'lora_' in k
        }
        if not lora_state_dict:
            print("警告: 未找到LoRA权重进行保存。")
        else:
            torch.save(lora_state_dict, lora_weights_file)
            print(f"LoRA权重已保存至: {lora_weights_file}")

    print("微调完成。")

    # --- 使用微调后的模型进行生成 (示例) ---
    print("\n--- 使用微调后的模型生成示例序列 ---")
    evo2_wrapper.model.eval() # 设置为评估模式

    # 如果需要从头加载并应用LoRA权重:
    # print("重新加载模型并应用LoRA权重进行推理...")
    # new_evo2_wrapper = Evo2(model_name=base_model_name, local_path=local_model_path)
    # for param in new_evo2_wrapper.model.parameters():
    #     param.requires_grad = False # 冻结
    # apply_lora_to_evo2_model(new_evo2_wrapper.model, lora_rank, lora_alpha, target_module_patterns)
    # lora_weights = torch.load(lora_weights_file, map_location=device)
    # new_evo2_wrapper.model.load_state_dict(lora_weights, strict=False) # strict=False只加载LoRA参数
    # new_evo2_wrapper.model.to(device)
    # new_evo2_wrapper.model.eval()
    # current_model_for_generation = new_evo2_wrapper

    current_model_for_generation = evo2_wrapper # 当前模型已包含LoRA层和更新的权重

    prompt_seqs = ["GATTACA"] # 可以是空字符串 "" 或一个短的起始序列
    num_tokens_to_generate = 100

    print(f"使用提示 '{prompt_seqs[0]}' 生成 {num_tokens_to_generate} 个token...")
    with torch.no_grad():
        # Evo2的generate函数在 evo2/models.py 中定义
        generated_output = current_model_for_generation.generate(
            prompt_seqs=prompt_seqs,
            n_tokens=num_tokens_to_generate,
            temperature=0.8, 
            top_k=40,
            # top_p=0.9, # 可选
            batched= (len(prompt_seqs) > 1),
            # device=device # generate函数内部会处理设备
        )
    
    if generated_output and generated_output.sequences:
        generated_sequence = generated_output.sequences[0]
        print(f"生成的增强子序列 (长度 {len(generated_sequence)}):\n{generated_sequence}")
        
        generated_output_file = os.path.join(output_dir, "generated_enhancers_after_finetune.txt")
        with open(generated_output_file, "a", encoding="utf-8") as f:
            f.write(f">prompt:{prompt_seqs[0]}\n{generated_sequence}\n\n")
        print(f"生成的序列已追加到: {generated_output_file}")
    else:
        print("生成失败或未产生输出。")


if __name__ == "__main__":
    # 确保在运行此脚本的目录中有一个名为 data.csv 的文件，
    # 或者在 finetune_evo2_for_enhancer_generation() 函数中更新 csv_file 路径。
    # data.csv 应包含一个名为 'sequence' 的列头，下面是DNA序列，每行一个。
    # 示例 data.csv 内容:
    # sequence
    # ATGCATGCATGCATGCATGCATGCATGCATGC
    # GATCGATCGATCGATCGATCGATCGATCGATC
    # TTTAAATTTAAATTTAAATTTAAATTTAAATT

    # 如果 data.csv 不存在，创建一个虚拟的用于测试
    if not os.path.exists("data.csv"):
        print("创建虚拟 data.csv 用于测试目的。")
        dummy_sequences = [
            'GATTACA' * 20,  # 140 chars
            'TAGCATGC' * 15, # 120 chars
            'ATGCATGC' * 18, # 144 chars
            'CGTACGTA' * 12, # 96 chars
            'AGCTAGCT' * 10, # 80 chars
            'TTTTCCCCGGGGAAAA' * 5, # 80 chars
            'G' * 60,        # 60 chars
            'A' * 30         # 30 chars (将被min_length过滤)
        ]
        pd.DataFrame({'sequence': dummy_sequences}).to_csv("data.csv", index=False)

    finetune_evo2_for_enhancer_generation()