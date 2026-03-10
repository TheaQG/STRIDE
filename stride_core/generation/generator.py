"""
Core orchestration class. The generation pipeline should live here. Decides when and what to generate.

Responsibilities:
- Load checkpoint
- build model
- optionally load EMA weights if saved and requested
- build dataset/dataloader for requested split
- loop through cases
- call sampler
- manage ensemble generation
- call PMM computation if requested
- call output saving helpers
- maybe call plotting helpers
"""