"""Evidence and completeness contracts for the editable course presentation."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def content():
    return json.loads((ROOT / 'docs/presentation-content.json').read_text(encoding='utf-8'))


def test_presentation_has_complete_ordered_notes_and_explicit_main_timing():
    spec = content()
    assert [s['id'] for s in spec['slides']] == [f'S{i:02}' for i in range(1, 21)] + [f'A{i:02}' for i in range(1, 5)]
    assert spec['main_slide_count'] == 20 and spec['appendix_slide_count'] == 4
    assert sum(s['seconds'] for s in spec['slides'][:20]) == spec['planned_main_seconds'] == 500
    assert all(len(s['notes']) > 50 and s['sources'] for s in spec['slides'])
    assert spec['slide_size_px'] == [1600, 900]
    assert not any('待填写' in json.dumps(s, ensure_ascii=False) for s in spec['slides'])


def test_presentation_local_sources_and_uncropped_asset_identity_exist():
    spec = content()
    images = []
    for slide in spec['slides']:
        for source in slide['sources']:
            if not source.startswith('https://'):
                assert (ROOT / source.split('#')[0]).is_file()
        if 'image' in slide:
            assert (ROOT / slide['image']).is_file()
            images.append(slide['image'])
        if 'crop' in slide:
            crop = slide['crop']
            assert all(0 <= value < 1 for value in crop.values())
            assert crop['top'] + crop['bottom'] < 1
            assert crop['left'] + crop['right'] < 1
    assert len(images) == len(set(images))


def test_deep_texture_chart_uses_all_four_frozen_real_results():
    spec = content()
    chart = next(s for s in spec['slides'] if s['id'] == 'S13')
    result = json.loads((ROOT / 'docs/task2-deep-texture-summary.json').read_text())
    models = ['learned_joint', 'random_joint', 'learned_rep1_to_rep2', 'learned_rep2_to_rep1']
    assert chart['values'] == [result['models'][name]['mean_fold_roc_auc'] for name in models]
    assert chart['axis_max'] == 1
    assert chart['values'][0] < chart['values'][1]


def test_candidate_table_preserves_main_registry_coordinates_and_five_regions():
    spec = content()
    slide = next(s for s in spec['slides'] if s['id'] == 'S11')
    with (ROOT / 'docs/task2-candidate-registry.csv').open() as handle:
        regions = list(csv.DictReader(handle))
    assert len(slide['rows']) == len(regions) == 5
    for displayed, region in zip(slide['rows'], regions):
        assert displayed[0] == region['candidate_region_id'].split('_')[-1]
        assert displayed[1] == f"{int(region['start']):,}–{int(region['end']):,}"
    case = next(s for s in spec['slides'] if s['id'] == 'A02')
    assert '3,463,680–3,484,160' in case['caption']
    assert '3,458,560–3,486,720' in case['caption']
