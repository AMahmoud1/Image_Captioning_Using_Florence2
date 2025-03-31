from transformers import AutoModelForCausalLM, AutoProcessor

if __name__ == "__main__":
    model_id = "microsoft/Florence-2-large"

    # Ensure the runtime is set to GPU in Colab.
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
