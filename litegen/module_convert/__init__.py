from typing import Any
from torch import nn
from .op_replace import replace_all_layernorms


class ModuleConverter():
    def __init__(self, config):
        self.sequence_parallel = config.sp_size > 1
        self.fused_layernorm = config.fused_layernorm
        

    def __call__(self, module: nn.Module) -> Any:
        module = self.op_replace(module)

        if self.sequence_parallel:
            module = self.convert_to_sequence_parallel(module)

        return module

    
    def op_replace(self, module):
        if self.fused_layernorm:
            module = replace_all_layernorms(module)
        
        return module

    def convert_to_sequence_parallel(self, model):
        from models.vid_sd3.sparse_attention import Attention, SparseAttnProcessor, SparseAttnProcessorSP

        sequence_parallel_attn_processor_convert_map = {
            Attention: {SparseAttnProcessor: SparseAttnProcessorSP}
            
        }        
        def _convert_attn_processor(module):
            if isinstance(module, tuple(sequence_parallel_attn_processor_convert_map.keys())):
                try:
                    cur_processor = module.get_processor()
                    new_processor = sequence_parallel_attn_processor_convert_map[type(module)][type(cur_processor)]()
                    module.set_processor(new_processor)
                    print(f"Attention processor: {type(cur_processor)} -> {type(new_processor)}")
                except Exception as e:
                    print(f"Failed to convert the attention processor of {type(module)} to sequence parallel, skipped: {e}")
            else:
                return module
            
        for _, module in model.named_modules():
            module = _convert_attn_processor(module)
            
        return model
                        