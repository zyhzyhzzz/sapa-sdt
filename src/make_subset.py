# 按 8 个隐私类别分层抽样（仅 "Both" 位置），固定种子
import json, random, collections
from paths import ANNOT, SUBSET
random.seed(0)
d = json.load(open(ANNOT, encoding='utf-8'))
by = collections.defaultdict(list)
for i, x in enumerate(d):
    if x['Privacy-exposed locations'].startswith('Both'):
        by[x['image_type'].strip().lower()].append(dict(idx=i, image=x['images'], category=x['image_type'].strip().lower(),
            level=x['conversations1']['security_type'].strip().capitalize(), instruction=x['conversations1']['instructions'].strip()))
sub = []
for k in sorted(by):
    sub += random.sample(by[k], 40)
json.dump(sub, open(SUBSET, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(sub), collections.Counter(s['level'] for s in sub))
