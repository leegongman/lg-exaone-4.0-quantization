import torch
import torch.nn as nn
from models.int_exaone4_layer import QuantExaone4DecoderLayer
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")


def _clone_omni_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in omni_state_dict(model).items()
    }


def _load_omni_state(model, state):
    if state:
        model.load_state_dict(state, strict=False)


def _has_non_finite_omni_parameters(model):
    for _, param in model.named_parameters():
        if ("smooth" in _.lower() or "bound_factor" in _.lower()) and not torch.isfinite(param).all():
            return True
    return False


def _stabilize_omni_parameters(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            lower_name = name.lower()
            if "smooth_scale" in lower_name:
                param.data = torch.nan_to_num(param.data, nan=1.0, posinf=1e2, neginf=1e-2).clamp_(1e-4, 1e2)
            elif "smooth_shift" in lower_name:
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=10.0, neginf=-10.0).clamp_(-10.0, 10.0)
            elif "bound_factor" in lower_name:
                param.data = torch.nan_to_num(param.data, nan=4.0, posinf=8.0, neginf=-8.0).clamp_(-8.0, 8.0)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     
def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    is_exaone = False
    supports_let = True
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "exaone" in args.net.lower():
        is_exaone = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
        DecoderLayer = QuantExaone4DecoderLayer
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/exaone now")

    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False
            self.is_exaone = False

        def forward(
            self,
            inp,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            past_key_values=None,
            use_cache=False,
            position_embeddings=None,
            **kwargs,
        ):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = attention_mask
            if self.is_llama or self.is_exaone:
                cache["position_ids"] = position_ids
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    layers[0].is_exaone = is_exaone

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "exaone" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/exaone now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama or is_exaone:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    def get_position_embeddings(hidden_states):
        if not is_exaone:
            return None
        return model.model.rotary_emb(hidden_states, position_ids)

    def decoder_forward(decoder_layer, hidden_states, mask):
        if is_exaone:
            outputs = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=get_position_embeddings(hidden_states),
            )
            return outputs if isinstance(outputs, tuple) else (outputs,)
        if is_llama:
            return decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
            )
        return decoder_layer(
            hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
        )



    if args.resume:
        omni_parameters = torch.load(args.resume, weights_only=False)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = decoder_forward(qlayer, fp_inps[j].unsqueeze(0), attention_mask)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = decoder_forward(qlayer, quant_inps[j].unsqueeze(0), attention_mask)[0]
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or is_exaone or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            if is_exaone:
                q_norm_weight = qlayer.self_attn.q_norm.weight.detach().abs().to(device=dev, dtype=dtype).clamp(min=1e-5)
                k_norm_weight = qlayer.self_attn.k_norm.weight.detach().abs().to(device=dev, dtype=dtype).clamp(min=1e-5)
                qkt_scale = (q_norm_weight / k_norm_weight).sqrt().clamp(min=1e-5)
                qlayer.register_parameter(
                    "qkt_smooth_scale",
                    torch.nn.Parameter(
                        qkt_scale
                    ),
                )

                out_name = f"{layer_name_prefix}.{i}.self_attn.o_proj"
                out_act = act_scales[out_name].to(device=dev, dtype=dtype).clamp(min=1e-5)
                out_act = out_act.view(
                    qlayer.self_attn.num_key_value_heads,
                    qlayer.self_attn.num_key_value_groups,
                    qlayer.self_attn.head_dim,
                ).amax(dim=1).reshape(-1)
                out_weight = qlayer.self_attn.o_proj.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                out_weight = out_weight.view(
                    qlayer.self_attn.num_key_value_heads,
                    qlayer.self_attn.num_key_value_groups,
                    qlayer.self_attn.head_dim,
                ).amax(dim=1).reshape(-1)
                out_scale = (out_act.pow(args.alpha) / out_weight.pow(1 - args.alpha)).clamp(min=1e-5)
                qlayer.register_parameter("out_smooth_scale", torch.nn.Parameter(out_scale))

                up_name = f"{layer_name_prefix}.{i}.mlp.down_proj"
                up_act = act_scales[up_name].to(device=dev, dtype=dtype).clamp(min=1e-5)
                up_weight = qlayer.mlp.down_proj.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                up_scale = (up_act.pow(args.alpha) / up_weight.pow(1 - args.alpha)).clamp(min=1e-5)
                qlayer.register_parameter("up_smooth_scale", torch.nn.Parameter(up_scale))
            else:
                qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
                for name,module in qlayer.named_modules():
                    if isinstance(module, QuantLinear):
                        for key in pairs.keys():
                            if key in name:
                                act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                                weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                                scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                                if use_shift and not is_llama:
                                    shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                                else:
                                    shift = torch.zeros_like(scale)
                                qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                                qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            use_grad_clip = args.wbits <= 4 or args.abits <= 4
            clip_grad = 0.5 if use_grad_clip else None
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            best_state = _clone_omni_state(qlayer)
            best_loss = float("inf")
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        qlayer.smooth_and_quant_temporary()
                        quant_out = decoder_forward(qlayer, quant_inps[index:index+args.batch_size,], attention_mask_batch)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info(
                            f"Layer {i} batch {j}: non-finite loss detected; restoring best parameters and stopping layer optimization."
                        )
                        _load_omni_state(qlayer, best_state)
                        optimizer.zero_grad(set_to_none=True)
                        break
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(
                        loss,
                        optimizer,
                        clip_grad=clip_grad,
                        parameters=get_omni_parameters(qlayer, use_shift),
                    ).cpu()
                    _stabilize_omni_parameters(qlayer)
                    if not torch.isfinite(norm) or _has_non_finite_omni_parameters(qlayer):
                        logger.info(
                            f"Layer {i} batch {j}: non-finite gradient/parameter detected; restoring best parameters and stopping layer optimization."
                        )
                        _load_omni_state(qlayer, best_state)
                        optimizer.zero_grad(set_to_none=True)
                        break
                    norm_list.append(norm.data)
                    batch_loss = float(loss.detach().cpu())
                    if batch_loss < best_loss:
                        best_loss = batch_loss
                        best_state = _clone_omni_state(qlayer)

                loss_mean = torch.stack(loss_list).mean() if loss_list else torch.tensor(float("nan"))
                norm_mean = torch.stack(norm_list).mean() if norm_list else torch.tensor(float("nan"))
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            _load_omni_state(qlayer, best_state)
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half() 
        # real smooth and quantization
        qlayer.smooth_and_quant_inplace()
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = decoder_forward(qlayer, quant_inps[j].unsqueeze(0), attention_mask)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant and getattr(args, "save_format", "gptq") == "gptq":
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
