import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM(torch.nn.Module):
    def __init__(self, model_name='meta-llama/Llama-3.1-8B-Instruct'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, question, graph_embedding, label=None, desc=None):
        # This is where the GNN embedding is 'fused' with the text
        # For simplicity in STaRK, we use the text description + embedding
        prompt = f"Context: {desc}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # In a real G-Retriever, graph_embedding is used as a soft-prompt here
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss

    def inference(self, question, graph_embedding, desc=None):
        prompt = f"Context: {desc}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_tokens = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)