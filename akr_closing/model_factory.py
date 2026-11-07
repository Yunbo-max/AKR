"""Lazy dispatch for the explicitly supported, pinned audio model families."""
from __future__ import annotations

MODEL_SPECS = {
    'qwen7b': {'id': 'Qwen/Qwen2.5-Omni-7B', 'revision': 'ae9e1690543ffd5c0221dc27f79834d0294cba00',
               'family': 'qwen25', 'conditional': False, 'python': '/root/.venvs/akr/bin/python'},
    'qwen3b': {'id': 'Qwen/Qwen2.5-Omni-3B', 'revision': 'f75b40e3da2003cdd6e1829b1f420ca70797c34e',
               'family': 'qwen25', 'conditional': False, 'python': '/root/.venvs/akr/bin/python'},
    'glm4voice9b': {'id': 'zai-org/glm-4-voice-9b', 'revision': '352e5a64063448d5e46394753fb65bf1dbab975b',
                    'family': 'glm', 'conditional': True, 'python': '/root/.venvs/akr-glm/bin/python'},
    'qwen3omni': {'id': 'Qwen/Qwen3-Omni-30B-A3B-Instruct', 'revision': '26291f793822fb6be9555850f06dfe95f2d7e695',
                  'family': 'qwen3', 'conditional': True, 'python': '/root/.venvs/akr-qwen3/bin/python'},
}


def model_family(model_id: str) -> str:
    for spec in MODEL_SPECS.values():
        if spec['id'] == model_id:
            return spec['family']
    raise ValueError(f'unsupported audio model: {model_id}')


def backend_class(model_id):
    family = model_family(model_id)
    if family == 'qwen25':
        from .backend import OmniBackend
        cls = OmniBackend
    elif family == 'qwen3':
        from .qwen3_backend import Qwen3OmniBackend
        cls = Qwen3OmniBackend
    else:
        from .glm_backend import GLMVoiceBackend
        cls = GLMVoiceBackend
    return cls


def create_backend(model_id, revision, **kwargs):
    return backend_class(model_id)(model_id, revision, **kwargs)


def validate_backend_environment(model_id):
    """Import the actual architecture in its interpreter without loading weights."""
    from importlib.metadata import version
    family = model_family(model_id)
    required = {'qwen25': '4.52.3', 'qwen3': '4.57.3', 'glm': '4.44.1'}[family]
    actual = version('transformers')
    if actual != required:
        raise RuntimeError(f'{family} requires its prepared transformers=={required} environment; found {actual}')
    if family == 'qwen25':
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    elif family == 'qwen3':
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeProcessor
    else:
        from .glm_backend import check_dependencies
        check_dependencies()
    return {p: version(p) for p in ['torch', 'transformers', 'numpy', 'scipy']}
