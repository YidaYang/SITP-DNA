
import os
import torch
import torch.nn as nn
# from evo2 import Evo2 # 假设 evo2 库已安装且可导入

# --- 从您的 Notebook 中复制 Evo2ForRegression 类的定义 ---
# 确保 Evo2 库可以被导入，如果Evo2的导入方式有变，请相应修改
try:
    from evo2 import Evo2
except ImportError:
    print("错误：未找到Evo2库。请确保已正确安装，并且在PYTHONPATH中。")
    print("如果Evo2不是一个标准库，您可能需要将其代码放在与此脚本相同的目录或可访问的路径中。")
    # 作为占位符，如果Evo2不可用，后续代码会失败
    class Evo2: # Placeholder
        def __init__(self, model_name, local_path):
            print(f"警告: Evo2 实际的类未加载。使用了占位符。 model_name={model_name}, local_path={local_path}")
            self.tokenizer = self # Placeholder tokenizer
            self.model = nn.Module() # Placeholder model
            self.model.config = type('config', (), {'hidden_size': 4096})() # Placeholder config

        def tokenize(self, text): # Placeholder tokenizer method
            print("警告: 使用了占位符 Evo2.tokenizer.tokenize")
            return [ord(c) for c in text]

        def __call__(self, input_ids, return_embeddings=False, layer_names=None): # Placeholder call
            print("警告: 使用了占位符 Evo2.__call__")
            # 返回一个符合期望结构的模拟输出
            batch_size, seq_len = input_ids.shape
            hidden_size = self.model.config.hidden_size
            # 模拟 embeddings 输出
            mock_embeddings = {
                layer_names[0]: torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
            }
            return None, mock_embeddings # (logits_placeholder, embeddings)


class Evo2ForRegression(nn.Module):
    """
    基于evo2+全连接+回归头的活性预测模型。
    与训练脚本中的定义保持一致。
    """
    def __init__(self, model_name="evo2_7b", evo2_base_model_local_path='/root/autodl-tmp/evo2/models/evo2-7b/evo2_7b.pt', dropout_rate=0.1, intermediate_size=512):
        super().__init__()
        print(f"初始化 Evo2ForRegression，加载 Evo2 基础模型: {model_name} 从 {evo2_base_model_local_path}...")
        try:
            self.evo2_wrapper = Evo2(model_name=model_name, local_path=evo2_base_model_local_path)
        except Exception as e:
            print(f"加载 Evo2 基础模型 '{model_name}' (路径: {evo2_base_model_local_path}) 时出错: {e}")
            print("请确认模型名称是否正确，Evo2 基础模型文件路径是否正确，以及相关依赖是否已安装。")
            raise

        # 在推理时，冻结参数是默认行为，因为我们不计算梯度。
        # 但为了与训练时的模型结构一致性，可以保留这部分逻辑，尽管 param.requires_grad 在 eval() 模式下不起主要作用。
        # print("冻结Evo2基础模型参数...")
        for param in self.evo2_wrapper.model.parameters():
            param.requires_grad = False

        hidden_size = self.evo2_wrapper.model.config.hidden_size

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) # Dropout 在 model.eval() 模式下会自动关闭
        self.regressor = nn.Linear(intermediate_size, 1)

        # 将新层转换为bfloat16以匹配预期的输入数据类型和训练时的设置
        # 如果在不支持bfloat16的CPU上运行，或者模型权重本身不是bfloat16，可能需要调整
        self.fc1.to(torch.bfloat16)
        self.regressor.to(torch.bfloat16)
        print(f"Evo2ForRegression 初始化完成。隐藏层大小={hidden_size}, 全连接层中间大小={intermediate_size}。")

    def forward(self, input_ids):
        # 从指定的层提取DNA增强子序列的embeddings。需要与训练时一致。
        layer_to_embed = "blocks.28.mlp.l3" # 这是您在训练脚本中使用的层

        # Evo2 模型前向传播
        # self.evo2_wrapper 返回 (logits, embeddings_dict)
        _, embeddings = self.evo2_wrapper(input_ids, return_embeddings=True, layer_names=[layer_to_embed])
        
        last_layer_output = embeddings[layer_to_embed] # 形状: [batch_size, sequence_length, hidden_size]
        
        # 平均池化
        pooled_output = last_layer_output.mean(dim=1) # 形状: [batch_size, hidden_size]
        
        # 通过自定义的回归头
        # 注意：如果 pooled_output 是 float32 而 fc1 是 bfloat16, PyTorch 会尝试转换
        # 为确保类型一致，可以显式转换 pooled_output.to(torch.bfloat16)
        x = self.fc1(pooled_output.to(self.fc1.weight.dtype)) # 确保输入类型与fc1层权重类型一致
        x = self.activation(x)
        x = self.dropout(x) # 在eval模式下，dropout不起作用
        prediction = self.regressor(x)

        return prediction.squeeze(1) # 压缩输出维度，例如从 [batch_size, 1] 到 [batch_size]

def predict_enhancer_activity(
    trained_model_pt_path,
    sequences_to_predict,
    evo2_model_name_used_for_training,
    evo2_base_model_file_path, # Evo2 基础模型 .pt 文件的路径
    max_seq_len_for_tokenizer,
    intermediate_fc_size,
    device_type="cuda" # "cuda" 或 "cpu"
):
    """
    使用训练好的 Evo2RegressionModel 进行增强子活性预测。

    参数:
        trained_model_pt_path (str): 训练好的、包含回归头的模型权重文件路径 (.pt)。
                                     (这是您训练脚本中保存的 output_model_file)
        sequences_to_predict (list or str): 一个DNA序列字符串，或一个包含多个DNA序列字符串的列表。
        evo2_model_name_used_for_training (str): 训练时使用的Evo2基础模型的名称 (例如 "evo2_7b")。
        evo2_base_model_file_path (str): Evo2 基础模型文件本身的路径 (例如 '/path/to/evo2_7b.pt')。
                                         这是初始化 Evo2ForRegression 时其内部 Evo2 实例所需的路径。
        max_seq_len_for_tokenizer (int): 分词和处理序列时使用的最大长度 (与训练时一致)。
        intermediate_fc_size (int): 回归头中全连接层的中间大小 (与训练时一致)。
        device_type (str): 指定设备 "cuda" 或 "cpu"。
    返回:
        list: 包含每个输入序列预测活性值的列表。如果发生错误则返回 None。
    """
    # --- 设备设置 ---
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if device_type == "cuda":
            print("警告: 请求使用CUDA，但CUDA不可用。将使用CPU。")
    print(f"推理设备: {device}")

    # --- 1. 初始化模型结构 ---
    # dropout_rate 在推理时不起作用 (因为 model.eval())，但为保持与 __init__ 一致性可以传入训练时的值。
    # 关键是 evo2_base_model_file_path 必须正确指向 Evo2 基础模型文件。
    try:
        model = Evo2ForRegression(
            model_name=evo2_model_name_used_for_training,
            evo2_base_model_local_path=evo2_base_model_file_path, # 传递基础模型路径
            dropout_rate=0.1, # 与训练时一致即可，eval模式下不生效
            intermediate_size=intermediate_fc_size
        )
    except Exception as e:
        print(f"初始化 Evo2ForRegression 模型失败: {e}")
        return None

    # --- 2. 加载训练好的模型权重 (回归头和FC层) ---
    try:
        # map_location 确保模型可以加载到当前选择的 device
        state_dict = torch.load(trained_model_pt_path, map_location=device)
        
        # 如果模型保存时使用了 nn.DataParallel (导致键名有 'module.' 前缀)
        # 而当前加载未使用 DataParallel，则需要移除前缀。
        # 您的训练代码似乎没有使用 DataParallel 包装 Evo2ForRegression 实例。
        # 但为保险起见，可以添加处理逻辑。
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_dataparallel_model = False
        for k in state_dict:
            if k.startswith('module.'):
                is_dataparallel_model = True
                break
        
        if is_dataparallel_model:
            print("检测到模型权重来自 nn.DataParallel 封装，将移除 'module.' 前缀。")
            for k, v in state_dict.items():
                name = k[7:] # 移除 `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
            
        print(f"成功从 {trained_model_pt_path} 加载已训练的回归模型权重。")
    except FileNotFoundError:
        print(f"错误: 未找到训练好的模型权重文件 {trained_model_pt_path}")
        return None
    except Exception as e:
        print(f"加载训练好的模型权重时出错: {e}")
        return None

    model.to(device)
    model.eval() # 设置模型为评估模式 (非常重要，会关闭dropout等)

    # --- 3. 获取分词器 ---
    # 分词器是 Evo2 实例的一部分
    tokenizer = model.evo2_wrapper.tokenizer
    # Evo2 charleveltokenizer 的 pad token id 通常是 1
    pad_token_id = getattr(tokenizer, 'pad_token_id', 1) 

    # --- 4. 预处理输入序列并进行预测 ---
    if isinstance(sequences_to_predict, str): # 处理单个序列输入的情况
        sequences_to_predict = [sequences_to_predict]

    all_predictions = []
    with torch.no_grad(): # 推理时不需要计算梯度
        for dna_sequence in sequences_to_predict:
            if not isinstance(dna_sequence, str):
                print(f"警告: 输入 '{dna_sequence}' 不是一个有效的DNA序列字符串。跳过。")
                all_predictions.append(float('nan')) # 或者 None，或引发错误
                continue
            
            # 分词
            token_ids = tokenizer.tokenize(dna_sequence)

            # 手动进行填充/截断 (与训练时的数据集处理逻辑一致)
            if len(token_ids) > max_seq_len_for_tokenizer:
                token_ids = token_ids[:max_seq_len_for_tokenizer] # 截断
            else:
                padding_length = max_seq_len_for_tokenizer - len(token_ids)
                token_ids.extend([pad_token_id] * padding_length) # 填充

            # 转换为张量并移动到设备
            # 模型期望输入是 [batch_size, sequence_length]
            input_ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

            # 模型前向传播
            try:
                prediction = model(input_ids_tensor)
                all_predictions.append(prediction.item()) # .item() 用于从单元素张量中获取Python数值
            except Exception as e:
                print(f"对序列 '{dna_sequence[:30]}...' 进行预测时出错: {e}")
                all_predictions.append(float('nan')) # 记录错误

    return all_predictions

# --- 主程序入口: 配置参数并调用预测函数 ---
if __name__ == "__main__":
    # --- 用户需要根据实际情况修改以下参数 ---

    # 1. 训练好的回归模型权重文件 (.pt) 的路径
    #    这应该是您训练脚本中 main() 函数里的 `output_model_file` 的最终值。
    #    例如: "evo2_regression_output/evo2_7b_regression.pt"
    TRAINED_REGRESSION_MODEL_PATH = "evo2_regression_output/evo2_7b_regression.pt"  # <--- *** 请务必更新此路径 ***

    # 2. 训练时使用的 Evo2 基础模型的名称
    #    例如: "evo2_1b_base", "evo2_7b", "evo2_7b_base", "evo2_40b", "evo2_40b_base"
    EVO2_MODEL_NAME = "evo2_7b"  # <--- *** 确保与训练时一致 ***

    # 3. Evo2 基础模型文件本身的本地路径
    #    这是初始化 Evo2ForRegression 时，其内部 Evo2 实例所需的 `local_path`。
    #    例如: "/root/autodl-tmp/evo2/models/evo2-7b/evo2_7b.pt" (Linux)
    #    或 "D:/path/to/evo2_models/evo2-7b/evo2_7b.pt" (Windows)
    EVO2_BASE_MODEL_FILE_PATH = "/root/autodl-tmp/evo2/models/evo2-7b/evo2_7b.pt"  # <--- *** 请务必更新此路径 ***
                                                                                # 注意Windows路径格式: "D:\\path\\to\\model.pt"

    # 4. 分词和序列处理时使用的最大长度 (必须与训练时 `max_length` 一致)
    MAX_SEQUENCE_LENGTH = 512 # <--- *** 确保与训练时一致 ***

    # 5. 回归头中全连接层的中间大小 (必须与训练时 `intermediate_hidden_size` 一致)
    INTERMEDIATE_FC_LAYER_SIZE = 512 # <--- *** 确保与训练时一致 ***

    # 6. 要进行活性预测的DNA序列
    #    可以是一个单独的字符串，或一个字符串列表。
    #    确保序列字符是Evo2分词器支持的 (通常是 A, C, G, T, N 等)。
    sequences_for_prediction = [
        "AAGCTAGCTAATTGCTTCTTCAGTTGAAGACCTAAATGAGTTTTAAAGTGAAATGCATATCTCTAAGGGCTAAGTAGCCAACACAATAGGCAATTGAGATAGGAAAGACTAATTTAGAAAAGGTTGTTTTGTTCGTTTTTCTTTTTCCTTCCCTCCCTTCCTGATTTCCCATCTTCTTCCTCCCTCTTCTCTCCCCTCTCCCCCTTCTCCTTTCCGTCCTTCCTCCTTCAGTTCCCTCTTTCCTCTTTTTCACCCTTTTATTTAACATTATAAATACGATGGGATTGTGTCTGCGCTTTTGTTGGTAATTAAATAAATTATTTATACATTTAACACAATCTTGAATTACCAGGTGATCATCTTAGGCACTCAAAAGCATAAGAGCCCTTGAAAGCAATATCTAAGCATAGATATTCCATAGCACGTCTTACAATCTAAATATTGCTTTTAGTGTAATCGAAGCAGCAAGAGTAGTCACAGCAGTTGATGGACTATTTTTCAAATTGATTTCAAAAATGTATTTAAGGGGATGATCTTCTAGTCTAGATTACCTATTGATTTTTAATATGAAAAGCTCATTATGTAAGCAGTAACCGCATATAAAAACCTAGCAAACCTTTGCATAAATCCTTTAATTGAATTTCCAGAGCCTGTGGTTCTACTTTTTTTTTAATTAAATCTATTTCTTTTTTTAAGTGTTACTGTGTAATTTGCATGCTGTGAAGAGGCCCTGTCCCAGATAAAGTGCCATTGATCCTTATTAAACCTCACCTCTGGGCTTGCTTAAAACTAACTGGAAAAATTAAAGTGTTCATGCCGCAATGCACTTATAGCTTGTGTGATAGGATTATGGAAAAAATAATAAAACTAATTTCCAGGGGAGAATTTCTAATGTGAGATTTTATTTTTTTTCAATTTGATAATTAATAGTGAAATCATATCATATATATAAATCATATTTTAGCCTATAAACTGAAATGGCAATTAGGAAAGATAATATATACTTGATGTAAAACCATGTTACGTGCGGATAATCTTTTAGCACTTTAATTTTTTAATTGTAGAAGGAGAGAATTATGAATTCAAGTCAAACACATTAAATGGTGGGTTTCATCCAAAAAATCTGATTCTTTTACTATGTACTGTATTAGTGGATTTATAATATTAGTGGGAGGAAGTATAAAAGATATGGAAAAAGATATTCTGGTTATGTTCGTGCTAAAATGTGTGTATTAGAATTACCAGGGGAAAGAAAAATATAAAAGCTGCAATAGGTTTTTCTATTTTTTAATACCTAACATTTGTTATTTTAAAAGCAATAAAATCCCCTAAAGAAA",
        "TGAGAGAGGGCTCAGAGACAGTACTCGGCCTTGCATTTTTCTCCAGGCCTTGCAGGATGCAAGGTGGACAGTGCTGCAGTCCTGTGGATCTGGTGCCCTAGGCTCTGCAGACACATTGAGGGGACCCAGGGCTCTGAACAAATGGCCAGATGGTAAGAACACATCCCTCGAGTTGTTTCTTCCAGTTGCATTTTTCCCTAATCACTGTCTGCTTTGGGAAAGGTCGATCAACCTAGCAAAAGTACCCCCTGGACATCTTCACTAGGTCCACTGACTGCTCCCCAGAACTCTCAGTTTCCTGAGGATTGTGCCCTTTCCCTTCCTTTCCTGCTGCAGCTTCTCAGCCTAAGCGAACCTAGAGAGAGCAAGGGTGGGGAAGGAGAGCAGGGGTATGTGATTGCGCTGAGAACACCAAAAGCCCATAAAGTCTGAAAGGTTAAGCAAAGACTGAGGCATGAAGAGTGAAAATTCTCCATTCAACAATAATCCTCGCCCCCTGCCACCCCTGACATCTCTTCTTAAAATGGAAAGAAAGGGTGCATGATGACATTAGGCACTTTAAAATATGCAAAATTAGGAGCTGGTGCAGACCTCATTAACACCCGCCTCCAACATCATTAAGATGTTTCCAAGAAAATTAATTGAGAGACTCATTAAAAATAAATTAAGAAAAATGTGTTGCAGAGCGCCTGCACTCAGCTCATAAATCACAGCTCAGTGCTCTGGCCCGCTGCTCCGGCAATTAACTCCAACCTATTTGTTTTATCCTGACTGTGAAAATTAGAAAGCAGACGTGGAGATTAGATTAGGAATGTCTGTCAAGCGGAACTTGGAGTGAATATTTTAGGATACAAAATGGAAACCAGGAAAACAAAGACAGCAGAGATTTCATTCTGAGGAGCTTGTCTTTGAAAAGTGACAGGCAGATGAAGGGTGCTGGAGGGAGAGTATGGGGAAGGAGTACCTACTCTCGGGAAGGATGAAGGGGGAGAGAAAAGAACAGATGTTTAAACTCTTCTGACCAGGAATCGTGCAATACATTTGCATTATAAATATAAACAGTTACACATCTTAGTAAAGCTGGCATTGAGACATAAAATTGATGTTCCTGATGATACTTCATATGTCATACTTTAATGTTTAGACATAGAGTCATTTGGTTGAATCACTCGAAAGAGTTACTGGTCTCTTGCAATATTTTATGTATCAGTGATGACCACATGTTCCTCTATATGAGGTGAAATATGCCAGCCTCTTCCC",
        "CCCCTTCCACCTTTATCCACAATATCCTCGAGTGGACATCACTGGGGTTCCTCTGAATGACTGAGTTGCCTCTTCATTATTCCGCCCAAGATGTCAGCTAAGGCTGTTTACAAATCCAAGGATTCTCTTGCCAAATATACAGTCTGTTCTCCAAGCTTTCATGTTATAACGAAATGAGTAACAAGCAAGACATCTTAATATCCTATTCTGCTAGAAAGTGAGATATTTCCCCTCCCTCGTTCTTAACAGATAAATTAATATCATCAAACATTCTGAAAAGATCTTTTATGAAAACATCTCACTTGCCAAAAAAGAAAAGTTGTATTATAAAACTGGAGAAATTTGTTTCAACTTGTTAAAAGCCCTATTCTCAGCCATGAATTCGGTTCCCGTTTTTTTTCCCCCTTCAATTAATTTCACACTAATCCATTTCTTTATCAGGCGTTGGAGTGAATGCATGTGGATCGAGTGATGAGGATGAGGGGGCAATGGAGGTGTTTGGCTCTGTAATTTCATCCTTGAATTTTGTGATTACTAACAGGACAACTTTTTTAATTTGCTCTTTTGTCTGGATTCCCTGGCTGACAATCTGCTCGGTGAGCTCGGCTTTTTAATCAATCACCTACATAATCAAATGTCACTGGCTATCTGCTCCGTGTAATTACTTTTGCAATTAAAAATCAACCTCAAGTTGCCTCATCTAATTAGAGGGATGGGCAGATTTTCATCTAGATTGATTTTTTAATAAATATTGACTTAAAATGCCATAATCTCATCATATTCTTTCATTTTCTTTGTACCAAAAATCAAACAAATGGAAGAATTAGCAAGCAGAAGGAATCGAGGGACTTCAAAAGCTTCTGCTGGTTCAGACACACAAAACTATGCTGTACAGAAGCCCAGCTTAGCTTGCCTAATAACAACACTCAATAGCTTCCACCTTTTATTGGAAAAAAGAACAAAGCAATTCAATGATTATATTTCACACCAACATTGTTGCAAGCCCACCATTCTAAGAGCTCCTAATTTCATTTATTGTACTGCCAAAGACAATCATTTCTATGAATGATATTATTTCCTTTAAAACAATCCCACACATGCTACTAGAATTTTTTAGCATTATGAGAAAACATATAATGCTTAATGCTGGAACCGCAACCACTGAGTTTTCTTAAACATATGAATGCCACTACAGCCAGATAACTTCCTTGTCTTTGCTGCCGTGTGTCTTTTATGTATTGTAATTAAAACATTGTCAATAACACAGTTCGTTGCTTTTTGGTTG",
    ]
    # 或者单个序列:
    # sequences_for_prediction = "ACGTACGTACGT"

    # 7. 选择推理设备: "cuda" (如果GPU可用) 或 "cpu"
    DEVICE_CHOICE = "cuda"
    # DEVICE_CHOICE = "cpu"

    print("开始使用已训练模型进行增强子活性预测...")
    
    # 调用预测函数
    predicted_activities = predict_enhancer_activity(
        trained_model_pt_path=TRAINED_REGRESSION_MODEL_PATH,
        sequences_to_predict=sequences_for_prediction,
        evo2_model_name_used_for_training=EVO2_MODEL_NAME,
        evo2_base_model_file_path=EVO2_BASE_MODEL_FILE_PATH,
        max_seq_len_for_tokenizer=MAX_SEQUENCE_LENGTH,
        intermediate_fc_size=INTERMEDIATE_FC_LAYER_SIZE,
        device_type=DEVICE_CHOICE
    )

    # 打印预测结果
    if predicted_activities is not None:
        print("\n--- 预测结果 ---")
        for i, seq in enumerate(sequences_for_prediction):
            activity = predicted_activities[i]
            if activity is not None and not (isinstance(activity, float) and activity != activity): # 检查非NaN
                print(f"序列 {i+1} (前30bp: {seq[:30]}...): 预测活性 = {activity:.4f}")
            else:
                print(f"序列 {i+1} (前30bp: {seq[:30]}...): 预测失败或结果为NaN")
    else:
        print("增强子活性预测未能完成。请检查上述错误信息。")