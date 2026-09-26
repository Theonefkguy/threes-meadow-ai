from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[2]
selection=json.loads((root/'training/v2/selection.json').read_text())
eval=json.loads((root/'training/v2/evaluation.json').read_text())
v1,v2=eval['v1'],eval['v2'];delta=eval['paired']['difference']
def pct(n):return f'{n*100:.1f}%'
def interval(values):return '–'.join(pct(x) for x in values)
lines=['# 第二版实测结果','',
'以下结果来自本项目实际运行，成功条件为正常开局首次出现 3072 或更大的牌。','',
'## 200 局候选筛选','',
'| 策略 | 达成次数 | 达成率 |','|---|---:|---:|']
labels={'v1':'第一版','a':'A：多步分阶段训练（追加 40 万轮）','b':'B：得分辅助成功率（10 万轮）','c':'C：得分辅助成功率＋公开来牌特征（10 万轮）'}
for name,r in selection['screen'].items():lines.append(f"| {labels[name]} | {r['successes']}/{r['games']} | {pct(r['rate'])} |")
lines+=['',f"筛选胜出：**{labels[selection['winner']]}**。在读取独立测试结果前固定权重和校验和。",'',
'## 每种策略 1,000 局独立测试','',
'| 策略 | 达成次数 | 达成率 | 95% Wilson 区间 |','|---|---:|---:|---|',
f"| 第一版 | {v1['successes']}/1000 | {pct(v1['rate'])} | {interval(v1['wilson95'])} |",
f"| 第二版 | {v2['successes']}/1000 | {pct(v2['rate'])} | {interval(v2['wilson95'])} |",'',
f"达成率变化：**{delta*100:+.1f} 个百分点**；相对变化 {delta/v1['rate']*100:+.1f}%。配对差值近似 95% 区间：{interval(eval['paired']['pairedNormal95'])}。双侧精确 McNemar p={eval['paired']['mcnemarExactTwoSidedP']:.6g}。",'',
f"只在第一版成功的种子 {eval['paired']['v1Only']} 个，只在第二版成功的种子 {eval['paired']['v2Only']} 个。全部测试均到达目标或真实终局，没有步数上限截断。",'',
'先选定候选，再读取独立测试；未根据独立测试修改模型。筛选和独立测试分别使用 910001–910200、1910001–1911000，训练另用不同种子。', '',
'两者使用相同规则、三步搜索和 24,000 节点上限。离线评测不设墙钟时间限制，网页另有 160 ms 软预算；低性能手机可能搜索更浅。这是本项目重建规则下的结果，不能当作原版 Threes 应用或任意手机的保证。','',
'每局达到 3072 就停止，因此这些记录不用于声称平均终局得分、6144 达成率或总体最优策略有所提高。','',
'完整方法、训练量、失败尝试和复现命令见 [RL-V2.md](RL-V2.md)。原始逐局数据与冻结记录在 training/v2 中。','']
(root/'RL-V2-RESULTS.md').write_text('\n'.join(lines))
p=root/'RL-V2.md';s=p.read_text();s=s.replace('第二版以提高正常开局合出 3072 的比例为主要评测目标。','第二版以提高正常开局合出 3072 的比例为主要评测目标。实际结果见 [RL-V2-RESULTS.md](RL-V2-RESULTS.md)。');p.write_text(s)
if selection['winner']=='a':
 model=root/'dist/models/ntuple-v2.bin';a=model.read_bytes();old=(root/'dist/models/ntuple-v1.bin').read_bytes();assert a[16+2*8*65536*4:]==old[16+2*8*65536*4:]
 metadata={'version':2,'kind':'score','format':'NTD1','sha256':hashlib.sha256(a).hexdigest(),'bytes':len(a),'baseSha256':selection['v1Hash'],'extraTraining':json.loads((root/'training/v2/run-a/config.json').read_text()),'cumulativeTrainingEpisodes':1090000+400000,'post3072Stage':'V1 frozen unchanged','search':{'maxDepth':3,'maxNodes':24000,'browserSoftBudgetMs':160},'holdout':eval,'limitations':'Offline success test has no wall-time cutoff; 3072 stopping is not final-score evaluation.'}
 (root/'dist/models/ntuple-v2.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('\n'.join(lines[:26]))
