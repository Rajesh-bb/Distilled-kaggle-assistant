# kaggle assistant via knowledge distillation and QLORA fine tuning
Project: Efficient Kaggle Assistant via LLM Knowledge Distillation

Project Description: Engineered a domain-specific, lightweight AI assistant for Kaggle and Python queries by distilling a fine-tuned Llama-3.2-3B (Teacher) into a Llama-3.2-1B (Student). Focused on optimizing an LLM for deployment by balancing high accuracy with significant reductions in computational overhead.

Technical Approach: Fine-tuned the Teacher model using QLoRA and 4-bit quantization on a curated dataset, then built a custom PyTorch distillation trainer using KL Divergence loss. Optimized the training pipeline with gradient checkpointing and mixed precision to efficiently transfer "dark knowledge" to the smaller Student architecture.

Key Outcomes: Successfully reduced model size by 54% (2.1GB to 980MB) and improved inference latency by ~45% (4246ms to 2305ms). Delivered a deployment-ready model that retains strong reasoning capabilities for coding and data science tasks while minimizing resource usage.
