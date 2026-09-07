"""Compare controlled terminal snapshots; never certify cloth quality or performance."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--before', type=Path, required=True)
parser.add_argument('--after', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

before, after = read(args.before/'summary.json'), read(args.after/'summary.json')
if before['executable_sha256'] != after['executable_sha256'] or before['manifest'] != after['manifest']:
    raise RuntimeError('Controlled comparison requires the same executable and manifest')
clips = [case['clip'] for case in before['cases']]
if clips != [case['clip'] for case in after['cases']]:
    raise RuntimeError('Animation sets differ')
report = dict(qualification=False, executable_sha256=after['executable_sha256'],
              before=str(args.before.resolve()), after=str(args.after.resolve()),
              controls_repeat_exactly=True, cases=[])
for clip in clips:
    snapshots = [read(directory/clip/'physics-snapshot.json') for directory in (args.before,args.after)]
    quality = [read(directory/clip/'quality.json') for directory in (args.before,args.after)]
    case = dict(clip=clip, actors=[])
    for index in range(3):
        actors = [s['actors'][index] for s in snapshots]
        if actors[0]['time'] != actors[1]['time'] or len(actors[0]['cloth']) != len(actors[1]['cloth']):
            raise RuntimeError('Snapshot times or topology differ')
        difference = max(math.dist(a,b) for a,b in zip(actors[0]['cloth'],actors[1]['cloth']))
        if index < 2:
            report['controls_repeat_exactly'] &= difference == 0
        entry = dict(algorithm=index, max_position_difference_m=difference)
        for name, q in zip(('before','after'),quality):
            a=q['actors'][index]
            entry[name]=dict(edge_ratio_p95=a['edge_ratio_p95'],
                body_plane_p95_mm=a['body']['penetration_p95_m']*1000,
                body_plane_max_mm=a['body']['penetration_max_m']*1000,
                body_solid=a['body'].get('solid'),proxy_solid=a['proxy'].get('solid'))
        case['actors'].append(entry)
    report['cases'].append(case)
report['limitation'] = ('Terminal snapshots only. Body solid classification has ambiguous rays. '
                       'Changes cannot be attributed to the guide switch if the ML and pure XPBD controls differ.')
args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(dict(controls_repeat_exactly=report['controls_repeat_exactly'],
                     cases=[dict(clip=c['clip'],max_position_difference_m=[a['max_position_difference_m'] for a in c['actors']]) for c in report['cases']]),indent=2))
raise SystemExit(0 if report['controls_repeat_exactly'] else 1)
