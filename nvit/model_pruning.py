#!/home/yangz/.conda/envs/4D-humans/bin/python
### Will create trace for pruning
import torch
import pdb


def initilize_layer_pruning(layer, extra_name="", dim = 0):
    # initializes pruning of particular layer by specifying relevant fields.
    # Later, the pruner will iterate over modules and will prune it if sees .do_pruning = True
    # Parameters from which to compute criteria are stored in the field compute_criteria_from
    # Parameters that needs to be pruned are stored in set_to_zero
    # There are other ways to specify what to prune.
    layer.do_pruning = True
    compute_criteria_from = [{"parameter_name": extra_name + ".weight:" + layer.__repr__(), "dim": dim, "parameter": layer.weight, "fix": True}, ]

    set_to_zero = [{"parameter_name": extra_name + ".weight:" + layer.__repr__(), "dim": dim, "parameter": layer.weight}]

    if dim == 0:
        if hasattr(layer, "bias"):
            if not ((layer.bias is False) or (layer.bias is None)):
                compute_criteria_from.append(
                    {"parameter_name": extra_name + ".bias:" + layer.__repr__(), "dim": 0, "parameter": layer.bias})
                set_to_zero.append({"parameter_name": extra_name + ".bias:" + layer.__repr__(), "dim": 0, "parameter": layer.bias})
                
    if dim == 2:
        if hasattr(layer, "bias"):
            if not ((layer.bias is False) or (layer.bias is None)):
                compute_criteria_from.append(
                    {"parameter_name": extra_name + ".bias:" + layer.__repr__(), "dim": 1, "parameter": layer.bias})
                set_to_zero.append({"parameter_name": extra_name + ".bias:" + layer.__repr__(), "dim": 1, "parameter": layer.bias})

    layer.compute_criteria_from = compute_criteria_from
    layer.set_to_zero = set_to_zero


def add_compute_criteria_names(layer, name):
    if hasattr(layer, "compute_criteria_from_names"):
        layer.compute_criteria_from_names.append(name)
    else:
        layer.compute_criteria_from_names = [name, ]


def add_set_to_zero_names(layer, name):
    if hasattr(layer, "compute_criteria_from_names"):
        layer.set_to_zero_names.append(name)
    else:
        layer.set_to_zero_names = [name, ]


def connect_output_to_input(parent_parameter, child_parameter, dim=0, shift=0,bias=True, extra_name="", allow_trim=False):
    # connects pruning to zero out dependent child layers
    # parent_parameter - main layer from which to compute statistics
    # child_parameter  - parameter to be zeroed depending on the loss from parent
    # dim - dimension to be affected, 0 - output channel, 1 - input channel
    # bias - set to zero bias as well
    parent_parameter.set_to_zero.append({"parameter_name": extra_name  + ".weight:"+ child_parameter.__repr__(),
                                         "dim": dim, "parameter": child_parameter.weight, "shift": shift, "allow_trim": allow_trim})
    if hasattr(child_parameter, 'bias') and dim==0:
        if not ((child_parameter.bias is False) or (child_parameter.bias is None)):
            parent_parameter.set_to_zero.append({"parameter_name": extra_name  + ".bias:"+ child_parameter.__repr__(),
                                                 "dim": dim, "parameter": child_parameter.bias})
    if hasattr(child_parameter, 'bias') and dim==2 and bias:
        if not ((child_parameter.bias is False) or (child_parameter.bias is None)):
            parent_parameter.set_to_zero.append({"parameter_name": extra_name  + ".bias:"+ child_parameter.__repr__(),
                                                 "dim": 1, "parameter": child_parameter.bias})


def link_criteria_layers(parent_parameter, child_parameter, dim=0, extra_name=""):
    parent_parameter.compute_criteria_from.append({"parameter_name": extra_name  + ".weight:"+ child_parameter.__repr__(),
                                                   "dim": dim, "parameter": child_parameter.weight, "layer_link": child_parameter})
    if dim==0:
        if hasattr(child_parameter, 'bias'):
            if not ((child_parameter.bias is False) or (child_parameter.bias is None)):
                parent_parameter.compute_criteria_from.append({"parameter_name": extra_name  + ".bias:"+ child_parameter.__repr__(),
                                                           "dim": dim, "parameter": child_parameter.bias})
    if dim==2:
        if hasattr(child_parameter, 'bias'):
            if not ((child_parameter.bias is False) or (child_parameter.bias is None)):
                parent_parameter.compute_criteria_from.append({"parameter_name": extra_name  + ".bias:"+ child_parameter.__repr__(),
                                                           "dim": 1, "parameter": child_parameter.bias})


# 🟢 将此函数复制粘贴到 nvit/model_pruning.py 中，替换原有的 create_pruning_structure_vit

def create_pruning_structure_vit(student, prune_token=True, prune_emb=True, prune_MLP=True, prune_head=True, prune_qk=True, prune_v=True, only_skip=False):
    
    # 1. 安全获取模型本体 (处理 DDP wrapper)
    model = student.module if hasattr(student, 'module') else student

    # 2. 深入获取 ViT Backbone (处理 HMR2 结构)
    # HMR2 的 ViT 藏在 model.backbone 里
    if hasattr(model, 'backbone'):
        print("🔍 Detected HMR2 structure, navigating to model.backbone...")
        vit_model = model.backbone
    else:
        vit_model = model

    # 3. 获取 Block 列表
    # 有些 ViT 实现叫 .blocks，有些叫 .transformer.blocks
    if hasattr(vit_model, 'blocks'):
        blocks = vit_model.blocks
    elif hasattr(vit_model, 'transformer') and hasattr(vit_model.transformer, 'blocks'):
        blocks = vit_model.transformer.blocks
    else:
        raise AttributeError("❌ Could not find '.blocks' in the model! Please check model structure.")

    # 4. 获取 Patch Embedding
    patch_embed = getattr(vit_model, 'patch_embed', None)
    if patch_embed is None:
         # 尝试从 hybrid backbone 获取
         patch_embed = getattr(vit_model, 'backbone', None)

    # 5. 安全获取 cls_token (HMR2 的 ViT 可能没有这个属性)
    cls_token = getattr(vit_model, 'cls_token', None)
    pos_embed = getattr(vit_model, 'pos_embed', None)

    # ================= 剪枝结构定义 =================
    # 这里的逻辑保持 NViT 原样，只是把对象换成了我们提取出来的 vit_model 组件
    
    # Prune Patch Embedding
    if prune_emb and patch_embed is not None:
        # 假设是 Conv2d 实现的 PatchEmbed
        if hasattr(patch_embed, 'proj'):
            patch_embed.proj.pruning_dim = 0
            patch_embed.proj.pruning_groups = 1

    # Prune Position Embedding & CLS Token
    if prune_token:
        if cls_token is not None:
            # 注册为自定义剪枝点 (Custom Pruning Point)
            from pruning_core.pruning_utils import custom_pruning_map
            custom_pruning_map["cls_token"] = {
                "dim": 2, 
                "parameter": cls_token, 
                "shift": 0, 
                "allow_trim": False
            }
        
        if pos_embed is not None:
            from pruning_core.pruning_utils import custom_pruning_map
            custom_pruning_map["pos_embed"] = {
                "dim": 2, 
                "parameter": pos_embed, 
                "shift": 1 if cls_token is not None else 0, # 如果有 cls_token，pos_embed 偏移 1 位
                "allow_trim": True
            }

    # Prune Blocks (Attention & MLP)
    for i, block in enumerate(blocks):
        # Attention - QKV
        if prune_qk:
            block.attn.qkv.pruning_dim = 0 
            block.attn.qkv.pruning_groups = 3 # q, k, v

        # Attention - Projection
        block.attn.proj.pruning_dim = 1
        block.attn.proj.pruning_groups = 1

        # MLP - FC1
        if prune_MLP:
            block.mlp.fc1.pruning_dim = 0
            block.mlp.fc1.pruning_groups = 1

        # MLP - FC2
        if prune_MLP:
            block.mlp.fc2.pruning_dim = 1
            block.mlp.fc2.pruning_groups = 1

    print(f"✅ Pruning structure initialized for ViT with {len(blocks)} blocks.")


def enable_pruning(pruning_engine, prune_token=False, prune_emb=False, prune_MLP=False, prune_head=False, prune_qk=False, prune_v=False, only_skip = False):
    
    first_layer = True
    for layer, if_prune in enumerate(pruning_engine.prune_layers):
        if not if_prune:
            continue
        pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix']=True
        name = pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]["parameter_name"]
        
        print(f"DEBUG: 正在检查层 {name}") # 看看输出里是不是带了 backbone.
        if pruning_engine.use_momentum and len(pruning_engine.prune_network_accomulate["averaged"][layer]):
            pruning_engine.prune_network_accomulate["averaged"][layer] *= 0.0
        
        if not only_skip:
            # all experiments so far are with this state
            if prune_token:
                if ".attn.qkv" in name and "qkv." not in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_emb:
                if "patch_embed.proj" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_MLP:
                if ".mlp.fc1" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_head:
                if ".attn.qkv.head_mask" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_qk:
                if ".attn.qkv.Q" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False
            if prune_v:
                if ".attn.qkv.V" in name:
                    pruning_engine.pruning_parameters[layer]["compute_criteria_from"][0]['fix'] = False

        else:
            pass
