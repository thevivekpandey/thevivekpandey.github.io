import os


if os.getenv("QLORA_PATCH_VLLM_DEFAULT_WEIGHT_LOADER") == "1":
    try:
        import vllm.model_executor.model_loader.weight_utils as weight_utils

        original = weight_utils.default_weight_loader
        if not getattr(original, "_qlora_patched", False):
            def patched_default_weight_loader(param, loaded_weight, *args, **kwargs):
                return original(param, loaded_weight)

            patched_default_weight_loader._qlora_patched = True
            weight_utils.default_weight_loader = patched_default_weight_loader

            try:
                import vllm.model_executor.models.gpt_oss as gpt_oss

                gpt_oss.default_weight_loader = patched_default_weight_loader
            except Exception:
                pass
    except Exception:
        pass


if os.getenv("QLORA_PATCH_TRANSFORMERS_MXFP4_DTYPE") == "1":
    try:
        import transformers.integrations.mxfp4 as hf_mxfp4

        original_forward = hf_mxfp4.Mxfp4GptOssExperts.forward
        if not getattr(original_forward, "_qlora_patched", False):
            def patched_forward(self, hidden_states, routing_data, gather_idx, scatter_idx):
                FnSpecs, FusedActivation, matmul_ogs = (
                    hf_mxfp4.triton_kernels_hub.matmul_ogs.FnSpecs,
                    hf_mxfp4.triton_kernels_hub.matmul_ogs.FusedActivation,
                    hf_mxfp4.triton_kernels_hub.matmul_ogs.matmul_ogs,
                )
                swiglu_fn = hf_mxfp4.triton_kernels_hub.swiglu.swiglu_fn

                with hf_mxfp4.on_device(hidden_states.device):
                    act = FusedActivation(
                        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")),
                        (self.alpha, self.limit),
                        2,
                    )

                    bias_dtype = hidden_states.dtype
                    intermediate_cache1 = matmul_ogs(
                        hidden_states,
                        self.gate_up_proj,
                        self.gate_up_proj_bias.to(bias_dtype),
                        routing_data,
                        gather_indx=gather_idx,
                        precision_config=self.gate_up_proj_precision_config,
                        gammas=None,
                        fused_activation=act,
                    )

                    if intermediate_cache1.dtype != hidden_states.dtype:
                        intermediate_cache1 = intermediate_cache1.to(hidden_states.dtype)

                    intermediate_cache3 = matmul_ogs(
                        intermediate_cache1,
                        self.down_proj,
                        self.down_proj_bias.to(hidden_states.dtype),
                        routing_data,
                        scatter_indx=scatter_idx,
                        precision_config=self.down_proj_precision_config,
                        gammas=routing_data.gate_scal,
                    )
                return intermediate_cache3

            patched_forward._qlora_patched = True
            hf_mxfp4.Mxfp4GptOssExperts.forward = patched_forward
    except Exception:
        pass
