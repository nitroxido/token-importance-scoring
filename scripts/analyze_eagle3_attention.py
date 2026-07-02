"""
Task 4: EAGLE-3 Architecture Inspection
Analyzes EAGLE-3 model structure to identify attention integration points.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError as e:
    print(f"Error: Required packages not installed. {e}")
    sys.exit(1)


class EAGLE3Inspector:
    """Inspects EAGLE-3 model architecture for attention layers."""
    
    def __init__(self, model_name: str = "nvidia/EAGLE3-Mistral-7B"):
        """
        Initialize inspector.
        
        Args:
            model_name: HuggingFace model ID for EAGLE-3
        """
        self.model_name = model_name
        self.model = None
        self.attention_layers = []
        self.findings = {}
    
    def load_model(self, device: str = "cpu") -> None:
        """
        Load EAGLE-3 model from HuggingFace.
        
        Args:
            device: Device to load on ("cpu" or "cuda")
        """
        print(f"Loading {self.model_name} on {device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def get_model_architecture(self) -> str:
        """Get full model architecture as string."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return str(self.model)
    
    def find_attention_layers(self) -> Dict[str, Dict]:
        """
        Find all attention layers in the model.
        
        Returns:
            dict: Mapping of layer name to layer info
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.attention_layers = {}
        
        for name, module in self.model.named_modules():
            # Look for attention modules
            if "attention" in name.lower() or "self_attn" in name.lower():
                self.attention_layers[name] = {
                    'type': module.__class__.__name__,
                    'params': sum(p.numel() for p in module.parameters()),
                    'trainable_params': sum(
                        p.numel() for p in module.parameters() if p.requires_grad
                    ),
                }
        
        return self.attention_layers
    
    def test_output_attentions(self) -> bool:
        """
        Test if model supports output_attentions=True.
        
        Returns:
            bool: Whether model can output attention weights
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            print("Testing output_attentions support...")
            
            # Create dummy input
            dummy_input = torch.randint(0, 1000, (1, 128))
            
            # Try forward pass with output_attentions=True
            with torch.no_grad():
                output = self.model(
                    dummy_input,
                    output_attentions=True,
                    return_dict=True,
                )
            
            supports = hasattr(output, 'attentions') and output.attentions is not None
            
            if supports:
                print(f"✓ Model supports output_attentions")
                print(f"  Number of attention layers: {len(output.attentions)}")
                if len(output.attentions) > 0:
                    print(f"  Attention shape: {output.attentions[0].shape}")
                self.findings['output_attentions_supported'] = True
                self.findings['num_attention_layers'] = len(output.attentions)
                if len(output.attentions) > 0:
                    self.findings['attention_shape'] = list(output.attentions[0].shape)
            else:
                print("✗ Model does not support output_attentions")
                self.findings['output_attentions_supported'] = False
            
            return supports
        
        except Exception as e:
            print(f"✗ Error testing output_attentions: {e}")
            self.findings['output_attentions_supported'] = False
            return False
    
    def get_model_config(self) -> Dict:
        """Get model configuration."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        config = self.model.config.to_dict() if hasattr(self.model, 'config') else {}
        
        # Extract key fields
        key_fields = {
            'model_type': config.get('model_type'),
            'hidden_size': config.get('hidden_size'),
            'num_hidden_layers': config.get('num_hidden_layers'),
            'num_attention_heads': config.get('num_attention_heads'),
            'intermediate_size': config.get('intermediate_size'),
            'vocab_size': config.get('vocab_size'),
            'max_position_embeddings': config.get('max_position_embeddings'),
        }
        
        return {k: v for k, v in key_fields.items() if v is not None}
    
    def test_forward_pass(self, seq_len: int = 256) -> bool:
        """
        Test a forward pass to verify model works.
        
        Args:
            seq_len: Sequence length for test
            
        Returns:
            bool: Whether forward pass succeeded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            print(f"Testing forward pass (seq_len={seq_len})...")
            
            dummy_input = torch.randint(0, 1000, (1, seq_len))
            
            with torch.no_grad():
                output = self.model(dummy_input, return_dict=True)
            
            print(f"✓ Forward pass successful")
            print(f"  Output logits shape: {output.logits.shape}")
            
            self.findings['forward_pass_works'] = True
            self.findings['logits_shape'] = list(output.logits.shape)
            
            return True
        
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            self.findings['forward_pass_works'] = False
            return False
    
    def inspect_layer_details(self, layer_idx: int = 0) -> Dict:
        """
        Get detailed information about a specific attention layer.
        
        Args:
            layer_idx: Index of layer to inspect
            
        Returns:
            dict: Layer details
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Find the attention layer
        layer_name = list(self.attention_layers.keys())[layer_idx] if self.attention_layers else None
        
        if layer_name is None:
            print(f"No attention layer at index {layer_idx}")
            return {}
        
        module = dict(self.model.named_modules())[layer_name]
        
        details = {
            'name': layer_name,
            'type': module.__class__.__name__,
            'params': sum(p.numel() for p in module.parameters()),
            'sub_modules': [name for name, _ in module.named_modules() if name != ''],
        }
        
        return details
    
    def generate_report(self, output_file: str = "EAGLE3-ARCHITECTURE.md") -> None:
        """
        Generate a comprehensive architecture report.
        
        Args:
            output_file: Path to save report
        """
        if self.model is None:
            print("Model not loaded. Generate report requires loaded model.")
            return
        
        report = []
        report.append("# EAGLE-3 Architecture Analysis\n")
        report.append(f"Model: {self.model_name}\n")
        report.append(f"Date: {__import__('datetime').datetime.now().isoformat()}\n")
        
        # Configuration
        report.append("## Configuration\n")
        config = self.get_model_config()
        for key, value in config.items():
            report.append(f"- **{key}**: {value}\n")
        
        # Attention Layers
        report.append("## Attention Layers\n")
        if self.attention_layers:
            for name, info in self.attention_layers.items():
                report.append(f"- `{name}`\n")
                report.append(f"  - Type: {info['type']}\n")
                report.append(f"  - Parameters: {info['params']:,}\n")
                report.append(f"  - Trainable: {info['trainable_params']:,}\n")
        else:
            report.append("No attention layers found.\n")
        
        # Findings
        report.append("## Key Findings\n")
        for key, value in self.findings.items():
            report.append(f"- {key}: {value}\n")
        
        # Integration Notes
        report.append("## Integration Notes for TIS-Aware Drafter\n")
        report.append("""
- **Attention Output**: Can extract attention weights for importance-based biasing
- **Attention Hook Points**: Register hooks on identified attention layers
- **Depth Scaling**: Use number of attention layers for depth-aware bias computation
- **Batch Processing**: Attention layers handle batched inputs naturally
- **Gradients**: Attention layers support gradient flow for bias learning (if fine-tuned)
        """)
        
        # Write report
        report_text = "".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to {output_file}")
        
        # Also save JSON findings
        json_file = output_file.replace('.md', '.json')
        with open(json_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'config': config,
                'attention_layers': self.attention_layers,
                'findings': self.findings,
            }, f, indent=2)
        
        print(f"✓ JSON findings saved to {json_file}")


def main():
    """Main inspection routine."""
    print("=" * 60)
    print("EAGLE-3 ARCHITECTURE INSPECTION")
    print("=" * 60)
    
    # Initialize inspector
    inspector = EAGLE3Inspector(model_name="nvidia/EAGLE3-Mistral-7B")
    
    try:
        # Load model (on CPU for quick inspection)
        inspector.load_model(device="cpu")
        
        # Run inspections
        print("\n--- Finding Attention Layers ---")
        attention_layers = inspector.find_attention_layers()
        print(f"Found {len(attention_layers)} attention layers:")
        for name in attention_layers.keys():
            print(f"  - {name}")
        
        print("\n--- Testing Forward Pass ---")
        inspector.test_forward_pass(seq_len=128)
        
        print("\n--- Testing Output Attentions ---")
        inspector.test_output_attentions()
        
        print("\n--- Model Configuration ---")
        config = inspector.get_model_config()
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\n--- Generating Report ---")
        inspector.generate_report(
            output_file=str(Path(__file__).parent.parent / "EAGLE3-ARCHITECTURE.md")
        )
        
        print("\n" + "=" * 60)
        print("INSPECTION COMPLETE")
        print("=" * 60)
    
    except Exception as e:
        print(f"Error during inspection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
