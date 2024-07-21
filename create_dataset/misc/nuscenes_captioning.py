from transformers import BlipProcessor, BlipForConditionalGeneration


class NuSceneCaptioning:
    def __init__(self, processor_path, model_path, device='cuda'):
        self.processor = BlipProcessor.from_pretrained(processor_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = device
        
    def __call__(self, image):
        
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)
        