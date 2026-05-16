#!/usr/bin/env python3
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path('/home/yu/Project_git/04_decode')
ANALYSIS_DIR = Path('/home/yu/Project_git/04_decode/analysis_20260416_182603')
DATASET_ROOT = Path('/home/yu/Project_git/01_dataset')

FOLDERS = [
    '01_decoded_3way',
    '02_decode_3way_av',
    '03_decode_8way',
    '04_decode_8way_av',
    '05_decode_gnlos_3way',
    '06_decode_gnlos_5way',
    '07_decode_gnlos_8way',
    '08_decode_gnlos_8way_av',
    '09_decode_glos_gnlos_8way_av',
]

PAIRWISE = [
    ('01_decoded_3way', '02_decode_3way_av'),
    ('03_decode_8way', '04_decode_8way_av'),
    ('07_decode_gnlos_8way', '08_decode_gnlos_8way_av'),
]

GNLOS_GRANULARITY = [
    '05_decode_gnlos_3way',
    '06_decode_gnlos_5way',
    '07_decode_gnlos_8way',
]

LABEL_ORDER_8 = [
    'front-left', 'front', 'front-right', 'right',
    'back-right', 'back', 'back-left', 'left'
]
ANGLE_MAP = {
    'front': 0,
    'front-right': 45,
    'right': 90,
    'back-right': 135,
    'back': 180,
    'back-left': 225,
    'left': 270,
    'front-left': 315,
}
FRONT_SECTOR = {'front-left', 'front', 'front-right'}
REAR_SECTOR = {'back-left', 'back', 'back-right'}
LR_CONFUSION_PAIRS = {
    ('left', 'right'), ('right', 'left'),
    ('front-left', 'front-right'), ('front-right', 'front-left'),
    ('back-left', 'back-right'), ('back-right', 'back-left'),
}


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def circular_diff_deg(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def format_pct(x):
    return round(100.0 * x, 2)


def extract_epoch(fp: Path):
    m = re.search(r'epoch_(\d+)_decode\.jsonl$', fp.name)
    return int(m.group(1)) if m else None


def extract_base_sample_id(audio_path: str, fallback_sample_id: str):
    if isinstance(audio_path, str):
        m = re.search(r'/samples/([^/]+)/audio/', audio_path)
        if m:
            return m.group(1)
    if isinstance(fallback_sample_id, str) and fallback_sample_id:
        return fallback_sample_id
    return ''


def normalize_sample_uid(raw_sample_id: str, base_sample_id: str):
    if isinstance(raw_sample_id, str) and raw_sample_id:
        if raw_sample_id != 'mic' and not re.match(r'^sample_\d+$', raw_sample_id):
            return raw_sample_id
    return base_sample_id or raw_sample_id or ''


def parse_scene_mic(base_sample_id: str):
    parts = base_sample_id.split('__') if base_sample_id else []
    scene_id = parts[0] if len(parts) >= 1 else ''
    mic_id = ''
    src_id = ''
    for p in parts[1:]:
        if p.startswith('mic'):
            mic_id = p
        if p.startswith('src'):
            src_id = p
    return scene_id, mic_id, src_id


def label_order(labels):
    labels = list(labels)
    if all(l in LABEL_ORDER_8 for l in labels):
        return [l for l in LABEL_ORDER_8 if l in set(labels)]
    return sorted(labels)


def compute_confusion(rows):
    labels = label_order({r['gt'] for r in rows} | {r['pred'] for r in rows})
    idx = {l: i for i, l in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]
    for r in rows:
        if r['gt'] not in idx or r['pred'] not in idx:
            continue
        mat[idx[r['gt']]][idx[r['pred']]] += 1
    return labels, mat


def compute_metrics(rows):
    n = len(rows)
    if n == 0:
        return {
            'n': 0,
            'accuracy': 0.0,
            'class_accuracy': {},
            'confusion_labels': [],
            'confusion_matrix': [],
            'top_confusions': [],
            'error_patterns': {
                'front_collapse': 0,
                'rear_ambiguity': 0,
                'lr_confusion': 0,
                'adjacent_errors': 0,
                'two_step_errors': 0,
                'large_jump_errors': 0,
                'total_errors': 0,
            },
        }

    acc = sum(1 for r in rows if r['correct']) / n
    by_gt_total = Counter(r['gt'] for r in rows)
    by_gt_correct = Counter(r['gt'] for r in rows if r['correct'])
    class_acc = {
        gt: (by_gt_correct[gt] / by_gt_total[gt] if by_gt_total[gt] else 0.0)
        for gt in label_order(by_gt_total.keys())
    }

    labels, mat = compute_confusion(rows)
    conf_pairs = []
    for i, gt in enumerate(labels):
        for j, pred in enumerate(labels):
            c = mat[i][j]
            if c > 0 and gt != pred:
                conf_pairs.append((gt, pred, c))
    conf_pairs.sort(key=lambda x: (-x[2], x[0], x[1]))

    pattern = {
        'front_collapse': 0,
        'rear_ambiguity': 0,
        'lr_confusion': 0,
        'adjacent_errors': 0,
        'two_step_errors': 0,
        'large_jump_errors': 0,
        'total_errors': 0,
    }

    for r in rows:
        if r['correct']:
            continue
        gt = r['gt']
        pred = r['pred']
        pattern['total_errors'] += 1

        if pred in FRONT_SECTOR and gt not in FRONT_SECTOR:
            pattern['front_collapse'] += 1
        if gt in REAR_SECTOR and pred in REAR_SECTOR and gt != pred:
            pattern['rear_ambiguity'] += 1
        if (gt, pred) in LR_CONFUSION_PAIRS:
            pattern['lr_confusion'] += 1

        if gt in ANGLE_MAP and pred in ANGLE_MAP:
            d = circular_diff_deg(ANGLE_MAP[gt], ANGLE_MAP[pred])
            if d == 45:
                pattern['adjacent_errors'] += 1
            elif d == 90:
                pattern['two_step_errors'] += 1
            elif d >= 135:
                pattern['large_jump_errors'] += 1

    return {
        'n': n,
        'accuracy': acc,
        'class_accuracy': class_acc,
        'confusion_labels': labels,
        'confusion_matrix': mat,
        'top_confusions': [
            {'gt': gt, 'pred': pred, 'count': c}
            for gt, pred, c in conf_pairs[:10]
        ],
        'error_patterns': pattern,
    }


def write_confusion_csv(path: Path, labels, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['gt\\pred'] + labels)
        for gt, row in zip(labels, matrix):
            w.writerow([gt] + row)


def parse_epoch_file(folder: str, fp: Path):
    rows = []
    parse_error = None
    with fp.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                parse_error = f'json_error_line_{line_no}:{e}'
                break

            raw_sample_id = str(obj.get('sample_id', ''))
            audio_path = str(obj.get('audio_path', ''))
            base_sample_id = extract_base_sample_id(audio_path, raw_sample_id)
            sample_uid = normalize_sample_uid(raw_sample_id, base_sample_id)
            pair_key = base_sample_id

            gt = str(obj.get('target_label', '')).strip()
            pred = str(obj.get('pred_label', '')).strip()
            if not gt:
                gt = str(obj.get('target_token', '')).replace('<DIR_H_', '').replace('>', '').lower().replace('_', '-')
            if not pred:
                pred = str(obj.get('pred_token', '')).replace('<DIR_H_', '').replace('>', '').lower().replace('_', '-')

            correct = obj.get('correct', None)
            if isinstance(correct, bool):
                is_correct = correct
            else:
                is_correct = (gt == pred)

            scene_id, mic_id, src_id = parse_scene_mic(base_sample_id)

            rows.append({
                'folder': folder,
                'epoch': extract_epoch(fp),
                'file': str(fp),
                'line_no': line_no,
                'sample_uid': sample_uid,
                'pair_key': pair_key,
                'raw_sample_id': raw_sample_id,
                'base_sample_id': base_sample_id,
                'audio_path': audio_path,
                'gt': gt,
                'pred': pred,
                'correct': bool(is_correct),
                'pred_score': safe_float(obj.get('pred_score', 0.0)),
                'scene_id': scene_id,
                'mic_id': mic_id,
                'src_id': src_id,
            })

    return rows, parse_error


def build_folder_data(folder: str):
    folder_path = ROOT / folder
    files = sorted(folder_path.glob('epoch_*_decode.jsonl'))

    epoch_infos = []
    for fp in files:
        epoch = extract_epoch(fp)
        rows, parse_error = parse_epoch_file(folder, fp)
        epoch_infos.append({
            'epoch': epoch,
            'file': fp,
            'rows': rows,
            'row_count': len(rows),
            'parse_error': parse_error,
        })

    valid_candidates = [e for e in epoch_infos if e['parse_error'] is None and e['row_count'] > 0]
    counts = [e['row_count'] for e in valid_candidates]
    mode_count = None
    if counts:
        freq = Counter(counts)
        mode_count = sorted(freq.items(), key=lambda x: (-x[1], -x[0]))[0][0]

    for e in epoch_infos:
        e['is_complete'] = (e['parse_error'] is None and mode_count is not None and e['row_count'] == mode_count)

    valid_epochs = [e for e in epoch_infos if e['is_complete']]

    metrics_by_epoch = {}
    for e in valid_epochs:
        metrics_by_epoch[e['epoch']] = compute_metrics(e['rows'])

    best_epoch = None
    if metrics_by_epoch:
        best_epoch = sorted(
            metrics_by_epoch.items(),
            key=lambda x: (x[1]['accuracy'], x[0])
        )[-1][0]

    best_rows = []
    best_metrics = None
    if best_epoch is not None:
        for e in valid_epochs:
            if e['epoch'] == best_epoch:
                best_rows = e['rows']
                best_metrics = metrics_by_epoch[best_epoch]
                break

    return {
        'folder': folder,
        'epoch_infos': epoch_infos,
        'mode_count': mode_count,
        'valid_epochs': [e['epoch'] for e in valid_epochs],
        'metrics_by_epoch': metrics_by_epoch,
        'best_epoch': best_epoch,
        'best_rows': best_rows,
        'best_metrics': best_metrics,
    }


def local_audio_path(audio_path: str):
    if not audio_path:
        return None
    p = Path(audio_path)
    if p.exists():
        return p
    marker = '/01_dataset/'
    if marker in audio_path:
        rel = audio_path.split(marker, 1)[1]
        mapped = DATASET_ROOT / rel
        return mapped
    return None


def load_sample_metadata_for_row(row, cache):
    key = row['pair_key'] + '|' + row['audio_path']
    if key in cache:
        return cache[key]

    audio_p = local_audio_path(row['audio_path'])
    md = {}
    if audio_p is not None:
        sample_dir = audio_p.parent.parent
        md_path = sample_dir / 'metadata' / 'sample.json'
        if md_path.exists():
            try:
                md = json.loads(md_path.read_text(encoding='utf-8'))
            except Exception:
                md = {}

    cache[key] = md
    return md


def subset_key_from_meta(meta, row):
    los_raw = meta.get('geometry_los')
    if not los_raw:
        sid = row.get('sample_uid', '')
        sid_l = sid.lower()
        if 'gnlos' in sid_l or sid_l.endswith('__nlos'):
            los_raw = 'gNLOS'
        elif 'glos' in sid_l or sid_l.endswith('__los'):
            los_raw = 'gLOS'

    los_key = None
    if isinstance(los_raw, str):
        if los_raw.lower() in {'glos', 'los'}:
            los_key = 'LOS'
        elif los_raw.lower() in {'gnlos', 'nlos'}:
            los_key = 'NLOS'

    in_fov = meta.get('in_fov', None)
    fov_key = None
    if isinstance(in_fov, bool):
        fov_key = 'FOV' if in_fov else 'OOF'

    if fov_key and los_key:
        return f'{fov_key}+{los_key}'
    return None


def confusion_pair_counter(rows):
    c = Counter()
    for r in rows:
        if r['gt'] != r['pred']:
            c[(r['gt'], r['pred'])] += 1
    return c


def class_acc_on_rows(rows):
    by_gt = defaultdict(list)
    for r in rows:
        by_gt[r['gt']].append(r['correct'])
    out = {}
    for gt in label_order(by_gt.keys()):
        vals = by_gt[gt]
        out[gt] = (sum(1 for v in vals if v) / len(vals)) if vals else 0.0
    return out


def pairwise_compare(name_a, name_b, rows_a, rows_b):
    map_a = {r['pair_key']: r for r in rows_a}
    map_b = {r['pair_key']: r for r in rows_b}
    keys = sorted(set(map_a.keys()) & set(map_b.keys()))

    aligned_a = [map_a[k] for k in keys]
    aligned_b = [map_b[k] for k in keys]

    acc_a = sum(1 for r in aligned_a if r['correct']) / len(aligned_a) if aligned_a else 0.0
    acc_b = sum(1 for r in aligned_b if r['correct']) / len(aligned_b) if aligned_b else 0.0

    class_a = class_acc_on_rows(aligned_a)
    class_b = class_acc_on_rows(aligned_b)
    cls_delta = {}
    for cls in label_order(set(class_a.keys()) | set(class_b.keys())):
        cls_delta[cls] = class_b.get(cls, 0.0) - class_a.get(cls, 0.0)

    conf_a = confusion_pair_counter(aligned_a)
    conf_b = confusion_pair_counter(aligned_b)
    all_pairs = set(conf_a.keys()) | set(conf_b.keys())
    conf_delta = []
    for p in all_pairs:
        da = conf_a.get(p, 0)
        db = conf_b.get(p, 0)
        if da != db:
            conf_delta.append({'gt': p[0], 'pred': p[1], 'audio_only': da, 'av': db, 'delta_av_minus_audio': db - da})
    conf_delta.sort(key=lambda x: (-abs(x['delta_av_minus_audio']), x['gt'], x['pred']))

    ao_correct_av_wrong = []
    ao_wrong_av_correct = []
    for k in keys:
        a = map_a[k]
        b = map_b[k]
        if a['correct'] and (not b['correct']):
            ao_correct_av_wrong.append((k, a, b))
        if (not a['correct']) and b['correct']:
            ao_wrong_av_correct.append((k, a, b))

    return {
        'pair': f'{name_a} vs {name_b}',
        'audio_only_folder': name_a,
        'av_folder': name_b,
        'n_intersection': len(keys),
        'audio_only_accuracy': acc_a,
        'av_accuracy': acc_b,
        'accuracy_delta': acc_b - acc_a,
        'class_accuracy_audio': class_a,
        'class_accuracy_av': class_b,
        'class_accuracy_delta': cls_delta,
        'confusion_delta_top': conf_delta[:12],
        'ao_correct_av_wrong': ao_correct_av_wrong,
        'ao_wrong_av_correct': ao_wrong_av_correct,
        'aligned_audio_rows': aligned_a,
        'aligned_av_rows': aligned_b,
    }


def top_weak_classes(class_acc, k=3):
    items = sorted(class_acc.items(), key=lambda x: (x[1], x[0]))
    return items[:k]


def render_confusion_text(labels, matrix):
    if not labels:
        return 'N/A'
    header = ['gt\\pred'] + labels
    rows = [header]
    for gt, row in zip(labels, matrix):
        rows.append([gt] + row)

    widths = [max(len(str(r[c])) for r in rows) for c in range(len(header))]

    lines = []
    for r in rows:
        line = ' | '.join(str(v).rjust(widths[i]) for i, v in enumerate(r))
        lines.append(line)
    return '\n'.join(lines)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dir(ANALYSIS_DIR)
    ensure_dir(ANALYSIS_DIR / 'confusion_matrices')

    folder_results = {}
    for folder in FOLDERS:
        folder_results[folder] = build_folder_data(folder)

    # write folder summary json
    summary_json = {}
    for folder, fr in folder_results.items():
        available_epochs = sorted([e['epoch'] for e in fr['epoch_infos'] if e['epoch'] is not None])
        skipped = []
        for e in fr['epoch_infos']:
            if not e['is_complete']:
                reason = e['parse_error'] if e['parse_error'] else f'row_count={e["row_count"]}, expected={fr["mode_count"]}'
                skipped.append({'epoch': e['epoch'], 'reason': reason})

        bm = fr['best_metrics'] if fr['best_metrics'] else {}
        summary_json[folder] = {
            'available_epochs': available_epochs,
            'valid_epochs': sorted(fr['valid_epochs']),
            'skipped_epochs': skipped,
            'best_epoch': fr['best_epoch'],
            'best_accuracy': bm.get('accuracy', 0.0),
            'class_accuracy': bm.get('class_accuracy', {}),
            'top_confusions': bm.get('top_confusions', []),
            'error_patterns': bm.get('error_patterns', {}),
            'n_samples': bm.get('n', 0),
        }

        if fr['best_metrics']:
            labels = fr['best_metrics']['confusion_labels']
            matrix = fr['best_metrics']['confusion_matrix']
            cpath = ANALYSIS_DIR / 'confusion_matrices' / f'{folder}__best_epoch_{fr["best_epoch"]:02d}.csv'
            write_confusion_csv(cpath, labels, matrix)

    (ANALYSIS_DIR / 'folder_best_metrics.json').write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding='utf-8')

    # always wrong & unstable per folder
    always_wrong_rows = []
    unstable_rows = []

    for folder, fr in folder_results.items():
        epoch_to_rows = {e['epoch']: e['rows'] for e in fr['epoch_infos'] if e['is_complete']}
        if not epoch_to_rows:
            continue
        sample_hist = defaultdict(list)

        for epoch, rows in epoch_to_rows.items():
            for r in rows:
                sample_hist[r['sample_uid']].append({
                    'epoch': epoch,
                    'correct': r['correct'],
                    'pred': r['pred'],
                    'gt': r['gt'],
                    'pair_key': r['pair_key'],
                    'scene_id': r['scene_id'],
                    'mic_id': r['mic_id'],
                    'audio_path': r['audio_path'],
                })

        for sid, hist in sample_hist.items():
            hist_sorted = sorted(hist, key=lambda x: x['epoch'])
            correct_flags = [h['correct'] for h in hist_sorted]
            preds = [h['pred'] for h in hist_sorted]
            gt_vals = [h['gt'] for h in hist_sorted]

            gt_main = Counter(gt_vals).most_common(1)[0][0] if gt_vals else ''
            pair_key = hist_sorted[0]['pair_key'] if hist_sorted else ''
            scene_id = hist_sorted[0]['scene_id'] if hist_sorted else ''
            mic_id = hist_sorted[0]['mic_id'] if hist_sorted else ''
            audio_path = hist_sorted[0]['audio_path'] if hist_sorted else ''

            if all(not c for c in correct_flags):
                always_wrong_rows.append({
                    'folder': folder,
                    'sample_uid': sid,
                    'pair_key': pair_key,
                    'scene_id': scene_id,
                    'mic_id': mic_id,
                    'gt': gt_main,
                    'num_epochs': len(hist_sorted),
                    'wrong_epochs': [h['epoch'] for h in hist_sorted],
                    'pred_history': [{ 'epoch': h['epoch'], 'pred': h['pred'] } for h in hist_sorted],
                    'audio_path': audio_path,
                })

            if (any(correct_flags) and not all(correct_flags)) or (len(set(preds)) > 1):
                flips = 0
                for i in range(1, len(correct_flags)):
                    if correct_flags[i] != correct_flags[i - 1]:
                        flips += 1
                unstable_rows.append({
                    'folder': folder,
                    'sample_uid': sid,
                    'pair_key': pair_key,
                    'scene_id': scene_id,
                    'mic_id': mic_id,
                    'gt': gt_main,
                    'num_epochs': len(hist_sorted),
                    'correct_ratio': sum(1 for c in correct_flags if c) / len(correct_flags) if correct_flags else 0.0,
                    'correctness_flips': flips,
                    'unique_pred_count': len(set(preds)),
                    'pred_history': [{ 'epoch': h['epoch'], 'pred': h['pred'], 'correct': h['correct'] } for h in hist_sorted],
                    'audio_path': audio_path,
                })

    # pairwise comparisons
    pairwise_results = []
    ao_correct_av_wrong_out = []
    ao_wrong_av_correct_out = []

    for ao_folder, av_folder in PAIRWISE:
        rows_a = folder_results[ao_folder]['best_rows']
        rows_b = folder_results[av_folder]['best_rows']
        cmp_res = pairwise_compare(ao_folder, av_folder, rows_a, rows_b)
        pairwise_results.append(cmp_res)

        for k, a, b in cmp_res['ao_correct_av_wrong']:
            ao_correct_av_wrong_out.append({
                'pair': cmp_res['pair'],
                'sample_key': k,
                'gt': a['gt'],
                'audio_only_pred': a['pred'],
                'av_pred': b['pred'],
                'scene_id': a['scene_id'] or b['scene_id'],
                'mic_id': a['mic_id'] or b['mic_id'],
                'audio_only_folder': ao_folder,
                'av_folder': av_folder,
                'audio_path': a['audio_path'] or b['audio_path'],
            })

        for k, a, b in cmp_res['ao_wrong_av_correct']:
            ao_wrong_av_correct_out.append({
                'pair': cmp_res['pair'],
                'sample_key': k,
                'gt': a['gt'],
                'audio_only_pred': a['pred'],
                'av_pred': b['pred'],
                'scene_id': a['scene_id'] or b['scene_id'],
                'mic_id': a['mic_id'] or b['mic_id'],
                'audio_only_folder': ao_folder,
                'av_folder': av_folder,
                'audio_path': a['audio_path'] or b['audio_path'],
            })

    # large jump errors from best epochs
    large_jump_rows = []
    for folder, fr in folder_results.items():
        be = fr['best_epoch']
        if be is None:
            continue
        for r in fr['best_rows']:
            if r['correct']:
                continue
            gt = r['gt']
            pred = r['pred']
            if gt in ANGLE_MAP and pred in ANGLE_MAP:
                d = circular_diff_deg(ANGLE_MAP[gt], ANGLE_MAP[pred])
                if d >= 135:
                    large_jump_rows.append({
                        'folder': folder,
                        'best_epoch': be,
                        'sample_uid': r['sample_uid'],
                        'pair_key': r['pair_key'],
                        'scene_id': r['scene_id'],
                        'mic_id': r['mic_id'],
                        'gt': gt,
                        'pred': pred,
                        'angle_diff_deg': d,
                        'audio_path': r['audio_path'],
                    })

    # scene/mic error summary
    scene_rows = []
    for folder, fr in folder_results.items():
        be = fr['best_epoch']
        if be is None:
            continue
        grp = defaultdict(list)
        for r in fr['best_rows']:
            grp[(r['scene_id'], r['mic_id'])].append(r)
        for (scene_id, mic_id), rows in grp.items():
            total = len(rows)
            errs = sum(1 for r in rows if not r['correct'])
            conf = confusion_pair_counter(rows)
            top_pair = conf.most_common(1)[0][0] if conf else ('', '')
            scene_rows.append({
                'folder': folder,
                'best_epoch': be,
                'scene_id': scene_id,
                'mic_id': mic_id,
                'total': total,
                'errors': errs,
                'error_rate': round(errs / total, 6) if total else 0.0,
                'top_confusion_gt': top_pair[0],
                'top_confusion_pred': top_pair[1],
            })

    # 09 subset analysis
    subset_report = {}
    md_cache = {}
    fr09 = folder_results['09_decode_glos_gnlos_8way_av']
    rows09 = fr09['best_rows']

    subset_rows = {
        'FOV+LOS': [],
        'OOF+LOS': [],
        'FOV+NLOS': [],
        'OOF+NLOS': [],
    }
    missing_subset = 0

    for r in rows09:
        md = load_sample_metadata_for_row(r, md_cache)
        skey = subset_key_from_meta(md, r)
        if skey in subset_rows:
            subset_rows[skey].append(r)
        else:
            missing_subset += 1

    subset_report['best_epoch'] = fr09['best_epoch']
    subset_report['total_samples'] = len(rows09)
    subset_report['missing_subset_label'] = missing_subset
    subset_report['subsets'] = {}

    for skey, srows in subset_rows.items():
        m = compute_metrics(srows)
        subset_report['subsets'][skey] = m
        cpath = ANALYSIS_DIR / 'confusion_matrices' / f'09_decode_glos_gnlos_8way_av__{skey.replace("+", "_")}.csv'
        write_confusion_csv(cpath, m['confusion_labels'], m['confusion_matrix'])

    (ANALYSIS_DIR / 'mixed_condition_09_subsets.json').write_text(json.dumps(subset_report, indent=2, ensure_ascii=False), encoding='utf-8')

    # gNLOS granularity report
    granularity = {}
    for folder in GNLOS_GRANULARITY:
        fr = folder_results[folder]
        bm = fr['best_metrics'] or {}
        granularity[folder] = {
            'best_epoch': fr['best_epoch'],
            'accuracy': bm.get('accuracy', 0.0),
            'class_accuracy': bm.get('class_accuracy', {}),
            'top_confusions': bm.get('top_confusions', []),
            'error_patterns': bm.get('error_patterns', {}),
            'n_samples': bm.get('n', 0),
            'num_classes': len(bm.get('class_accuracy', {})),
        }
    (ANALYSIS_DIR / 'gnlos_granularity.json').write_text(json.dumps(granularity, indent=2, ensure_ascii=False), encoding='utf-8')

    # summary table csv
    with (ANALYSIS_DIR / 'summary_table_all_folders.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'folder', 'available_epochs', 'valid_epochs', 'best_epoch', 'best_accuracy_pct',
            'weakest_classes', 'top3_confusions'
        ])
        for folder in FOLDERS:
            fr = folder_results[folder]
            available = sorted([e['epoch'] for e in fr['epoch_infos'] if e['epoch'] is not None])
            valid = sorted(fr['valid_epochs'])
            bm = fr['best_metrics'] or {'class_accuracy': {}, 'top_confusions': [], 'accuracy': 0.0}
            weak = top_weak_classes(bm.get('class_accuracy', {}), 3)
            weak_str = '; '.join([f'{k}:{format_pct(v):.2f}%' for k, v in weak])
            top3 = bm.get('top_confusions', [])[:3]
            top3_str = '; '.join([f"{x['gt']}->{x['pred']}:{x['count']}" for x in top3])
            w.writerow([
                folder,
                ','.join(map(str, available)),
                ','.join(map(str, valid)),
                fr['best_epoch'],
                round(format_pct(bm.get('accuracy', 0.0)), 2),
                weak_str,
                top3_str,
            ])

    # jsonl exports
    def write_jsonl(path, rows):
        with path.open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    write_jsonl(ANALYSIS_DIR / 'always_wrong_samples.jsonl', always_wrong_rows)
    write_jsonl(ANALYSIS_DIR / 'unstable_samples.jsonl', unstable_rows)
    write_jsonl(ANALYSIS_DIR / 'audio_only_correct_but_av_wrong.jsonl', ao_correct_av_wrong_out)
    write_jsonl(ANALYSIS_DIR / 'audio_only_wrong_but_av_correct.jsonl', ao_wrong_av_correct_out)
    write_jsonl(ANALYSIS_DIR / 'large_jump_errors.jsonl', large_jump_rows)

    with (ANALYSIS_DIR / 'scenewise_error_summary.csv').open('w', newline='', encoding='utf-8') as f:
        if scene_rows:
            fieldnames = list(scene_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in scene_rows:
                w.writerow(r)
        else:
            f.write('folder,best_epoch,scene_id,mic_id,total,errors,error_rate,top_confusion_gt,top_confusion_pred\n')

    # pairwise json serializable snapshot
    pairwise_json = []
    for p in pairwise_results:
        pairwise_json.append({
            'pair': p['pair'],
            'audio_only_folder': p['audio_only_folder'],
            'av_folder': p['av_folder'],
            'n_intersection': p['n_intersection'],
            'audio_only_accuracy': p['audio_only_accuracy'],
            'av_accuracy': p['av_accuracy'],
            'accuracy_delta': p['accuracy_delta'],
            'class_accuracy_delta': p['class_accuracy_delta'],
            'confusion_delta_top': p['confusion_delta_top'],
            'ao_correct_av_wrong_count': len(p['ao_correct_av_wrong']),
            'ao_wrong_av_correct_count': len(p['ao_wrong_av_correct']),
        })
    (ANALYSIS_DIR / 'pairwise_comparison.json').write_text(json.dumps(pairwise_json, indent=2, ensure_ascii=False), encoding='utf-8')

    # Markdown report
    lines = []
    lines.append('# Decode Analysis Report')
    lines.append('')
    lines.append(f'- Analysis root: `{ANALYSIS_DIR}`')
    lines.append('')

    lines.append('## 1) Summary Table (All Folders)')
    lines.append('')
    lines.append('| folder | available epochs | valid epochs | best epoch | best acc | weakest classes | top-3 confusion pairs |')
    lines.append('|---|---|---|---:|---:|---|---|')
    for folder in FOLDERS:
        fr = folder_results[folder]
        available = sorted([e['epoch'] for e in fr['epoch_infos'] if e['epoch'] is not None])
        valid = sorted(fr['valid_epochs'])
        bm = fr['best_metrics'] or {'class_accuracy': {}, 'top_confusions': [], 'accuracy': 0.0}
        weak = top_weak_classes(bm.get('class_accuracy', {}), 3)
        weak_str = '<br>'.join([f'{k}: {format_pct(v):.2f}%' for k, v in weak]) if weak else '-'
        top3 = bm.get('top_confusions', [])[:3]
        top3_str = '<br>'.join([f"{x['gt']}→{x['pred']}: {x['count']}" for x in top3]) if top3 else '-'
        lines.append(
            f"| {folder} | {available[0] if available else '-'}..{available[-1] if available else '-'} ({len(available)}) | "
            f"{valid[0] if valid else '-'}..{valid[-1] if valid else '-'} ({len(valid)}) | "
            f"{fr['best_epoch']} | {format_pct(bm.get('accuracy', 0.0)):.2f}% | {weak_str} | {top3_str} |"
        )

    lines.append('')
    lines.append('> Note: Incomplete epochs were auto-ignored when row count differed from folder mode count or parse failed.')
    lines.append('')

    lines.append('## 2) Pairwise Comparison')
    lines.append('')

    for p in pairwise_results:
        lines.append(f"### {p['pair']}")
        lines.append('')
        lines.append(f"- Intersection samples: {p['n_intersection']}")
        lines.append(f"- Overall accuracy delta (AV - Audio): {format_pct(p['accuracy_delta']):+.2f}%p ({format_pct(p['audio_only_accuracy']):.2f}% -> {format_pct(p['av_accuracy']):.2f}%)")

        cls_delta = p['class_accuracy_delta']
        inc = sorted(cls_delta.items(), key=lambda x: -x[1])[:5]
        dec = sorted(cls_delta.items(), key=lambda x: x[1])[:5]
        lines.append('- Class-wise accuracy delta (top gains): ' + (', '.join([f'{k} {format_pct(v):+.2f}%p' for k, v in inc]) if inc else '-'))
        lines.append('- Class-wise accuracy delta (top drops): ' + (', '.join([f'{k} {format_pct(v):+.2f}%p' for k, v in dec]) if dec else '-'))

        top_conf_change = p['confusion_delta_top'][:6]
        if top_conf_change:
            lines.append('- Top confusion changes (AV - Audio):')
            for c in top_conf_change:
                lines.append(f"  - {c['gt']}→{c['pred']}: {c['audio_only']} -> {c['av']} (Δ {c['delta_av_minus_audio']:+d})")
        else:
            lines.append('- Top confusion changes: -')

        lines.append(f"- Audio only correct -> AV wrong: {len(p['ao_correct_av_wrong'])}")
        lines.append(f"- Audio only wrong -> AV correct: {len(p['ao_wrong_av_correct'])}")

        # adjacent vs jump profile
        ma = compute_metrics(p['aligned_audio_rows'])['error_patterns']
        mb = compute_metrics(p['aligned_av_rows'])['error_patterns']
        lines.append(f"- Error profile Audio (adj/two-step/large): {ma['adjacent_errors']}/{ma['two_step_errors']}/{ma['large_jump_errors']}")
        lines.append(f"- Error profile AV    (adj/two-step/large): {mb['adjacent_errors']}/{mb['two_step_errors']}/{mb['large_jump_errors']}")
        lines.append(f"- Front collapse Audio -> AV: {ma['front_collapse']} -> {mb['front_collapse']}")
        lines.append(f"- Rear ambiguity Audio -> AV: {ma['rear_ambiguity']} -> {mb['rear_ambiguity']}")
        lines.append(f"- Left/right confusion Audio -> AV: {ma['lr_confusion']} -> {mb['lr_confusion']}")
        lines.append('')

    lines.append('## 3) gNLOS Granularity (3way -> 5way -> 8way)')
    lines.append('')
    for folder in GNLOS_GRANULARITY:
        g = granularity[folder]
        ep = g['error_patterns']
        lines.append(f"- {folder}: best epoch {g['best_epoch']}, acc {format_pct(g['accuracy']):.2f}%, classes {g['num_classes']}, errors(adj/two-step/large)={ep['adjacent_errors']}/{ep['two_step_errors']}/{ep['large_jump_errors']}")
    lines.append('')

    lines.append('## 4) Mixed Condition (09) - 4 Subsets')
    lines.append('')
    lines.append(f"- Best epoch: {subset_report['best_epoch']}")
    lines.append(f"- Total samples: {subset_report['total_samples']}")
    lines.append(f"- Missing subset labels: {subset_report['missing_subset_label']}")
    lines.append('')

    for skey in ['FOV+LOS', 'OOF+LOS', 'FOV+NLOS', 'OOF+NLOS']:
        m = subset_report['subsets'].get(skey, {})
        lines.append(f"### {skey}")
        lines.append('')
        lines.append(f"- N={m.get('n', 0)}")
        lines.append(f"- Accuracy={format_pct(m.get('accuracy', 0.0)):.2f}%")
        ca = m.get('class_accuracy', {})
        if ca:
            lines.append('- Class-wise acc: ' + ', '.join([f'{k}:{format_pct(v):.2f}%' for k, v in ca.items()]))
        else:
            lines.append('- Class-wise acc: -')
        tc = m.get('top_confusions', [])[:5]
        if tc:
            lines.append('- Top confusions: ' + ', '.join([f"{x['gt']}→{x['pred']}:{x['count']}" for x in tc]))
        else:
            lines.append('- Top confusions: -')
        lines.append('')

    # concentrated failures: top scene/mic by error rate with enough support
    lines.append('## 5) Failure Concentration (Scene/Mic)')
    lines.append('')
    scene_sorted = [r for r in scene_rows if r['total'] >= 5]
    scene_sorted.sort(key=lambda x: (-x['error_rate'], -x['errors'], x['folder'], x['scene_id'], x['mic_id']))
    for r in scene_sorted[:15]:
        lines.append(
            f"- {r['folder']} epoch{r['best_epoch']:02d} | {r['scene_id']} {r['mic_id']} | "
            f"err {r['errors']}/{r['total']} ({100*r['error_rate']:.1f}%) | top {r['top_confusion_gt']}→{r['top_confusion_pred']}"
        )
    lines.append('')

    lines.append('## 6) Key Interpretation')
    lines.append('')

    # simple interpretation heuristics from pairwise deltas
    pmap = {p['pair']: p for p in pairwise_results}
    key_0102 = pmap.get('01_decoded_3way vs 02_decode_3way_av')
    key_0304 = pmap.get('03_decode_8way vs 04_decode_8way_av')
    key_0708 = pmap.get('07_decode_gnlos_8way vs 08_decode_gnlos_8way_av')

    def cmp_sentence(p):
        if p is None:
            return 'N/A'
        delta = format_pct(p['accuracy_delta'])
        if delta > 0:
            trend = 'improves'
        elif delta < 0:
            trend = 'degrades'
        else:
            trend = 'ties'
        return f"{p['pair']}: AV {trend} by {delta:+.2f}%p"

    lines.append(f"- {cmp_sentence(key_0102)}")
    lines.append(f"- {cmp_sentence(key_0304)}")
    lines.append(f"- {cmp_sentence(key_0708)}")
    lines.append('')

    # main remaining issue heuristic from 07/08
    if key_0708:
        ma = compute_metrics(key_0708['aligned_audio_rows'])['error_patterns']
        mb = compute_metrics(key_0708['aligned_av_rows'])['error_patterns']
        lines.append('- 07 vs 08 detailed failure mode check:')
        lines.append(f"  - front collapse: {ma['front_collapse']} -> {mb['front_collapse']}")
        lines.append(f"  - rear ambiguity: {ma['rear_ambiguity']} -> {mb['rear_ambiguity']}")
        lines.append(f"  - left/right confusion: {ma['lr_confusion']} -> {mb['lr_confusion']}")
        lines.append(f"  - large jumps: {ma['large_jump_errors']} -> {mb['large_jump_errors']}")
        lines.append('')

    lines.append('- Next step recommendation: target hard subsets where AV hurts (AO correct->AV wrong), and add condition-aware gating (audio-trust vs vision-trust) instead of unconditional fusion.')
    lines.append('')

    report_path = ANALYSIS_DIR / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')

    print(f'Analysis complete. Outputs saved to: {ANALYSIS_DIR}')
    print(f'Report: {report_path}')
    print('Required artifacts:')
    for fn in [
        'always_wrong_samples.jsonl',
        'unstable_samples.jsonl',
        'audio_only_correct_but_av_wrong.jsonl',
        'audio_only_wrong_but_av_correct.jsonl',
        'large_jump_errors.jsonl',
        'scenewise_error_summary.csv',
    ]:
        print('-', ANALYSIS_DIR / fn)


if __name__ == '__main__':
    main()
