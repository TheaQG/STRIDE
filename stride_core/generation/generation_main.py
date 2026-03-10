"""
Entrypoint. Only job is to parse generation config, initialize a Generator, and call the requested generation mode.

Modes:
1. Single-sample quick generation (smoke tests, debugging, qualitative inspection)
    - Output one generated sample per case
2. Ensemble generation (final inference, uncertainty quantification, PMM computation, evaluation)
    - Outpyt multiple members per case
3. PMM generation product (hydrological/postprocessing use; comparison with deterministic fields)
    - Output PMM field per case, maybe also ensemble mean


- load config
- build Generator(...)
- generator.run()
"""