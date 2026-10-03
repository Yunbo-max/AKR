"""Validate delivered prompt text only; no model, network, or plotting calls."""
from pathlib import Path
import hashlib
import json
import re


def verify(root: Path) -> dict:
    index = json.loads((root / 'INDEX.json').read_text(encoding='utf-8'))
    items = index['items']
    assert len(items) == 18
    lengths = []
    for item in items:
        raw = (root / item['file']).read_bytes()
        text = raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf')
        assert b'\r' not in raw
        assert 4500 <= len(text) <= 5000, (item['id'], len(text))
        assert len(text) == item['characters']
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        assert (root / item['plain_text']).read_bytes() == raw
        assert len(re.findall(r'^## [1-8]\. ', text, re.M)) == 8
        assert not re.search(r'[\u4e00-\u9fff]', text)
        assert item['ratio'] in {'1:1', '16:9'}
        assert item['ratio'] in text
        assert all((root / p).is_file() for p in item['source'])
        assert 'Alternative layouts' not in text
        if item['status'] == 'pending_results':
            assert 'Awaiting complete results' in text or 'missing-field' in text
        lengths.append(len(text))
    blocks = re.findall(r'```text\n(.*?)```', (root / 'FIGURE_PROMPTS_ALL.md').read_text(), re.S)
    assert len(blocks) == 18
    assert blocks == [(root / x['file']).read_text() for x in items]
    assert (root / 'FIGURE_PROMPTS_ALL.md').read_bytes() == (root / 'FIGURE_PROMPTS_ALL.txt').read_bytes()
    return {
        'status': 'passed', 'prompt_count': len(items),
        'existing_figure_prompts': sum(x['status'] == 'existing' for x in items),
        'pending_result_templates': sum(x['status'] == 'pending_results' for x in items),
        'minimum_characters': min(lengths), 'maximum_characters': max(lengths),
        'pipeline_aspect_ratio': '16:9',
        'aspect_ratio_counts': {r: sum(x['ratio'] == r for x in items) for r in ['1:1', '16:9']},
        'copyable_file_and_combined_block_counts_match': True,
        'md_and_txt_identical': True, 'all_source_attachments_present': True,
        'gpu_experiments_run': 0, 'images_generated': 0,
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    result = verify(root)
    (root / 'VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
