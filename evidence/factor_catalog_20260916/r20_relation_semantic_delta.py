"""Final relation recipe correction. No engine edits; explicit economic definitions."""
import ast,copy,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"evidence/factor_catalog_20260915"))
E=ROOT/"evidence/factor_catalog_20260916"
def parse(s):return ast.parse(s,mode="eval").body
def ref(field,rank):return f"source_col('StockTopTenShareholder', '{field}', 'ShareholderRank', {rank})"
def total(xs):
    result=xs[0]
    for x in xs[1:]:result=f"add({result}, {x})"
    return result
def replace_call(formula,name,replacement):
    class Replace(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id==name:
                self.count+=1;return copy.deepcopy(replacement)
            return self.generic_visit(node)
    visitor=Replace();tree=visitor.visit(ast.parse(formula,mode="eval"))
    if visitor.count!=1:raise ValueError((name,visitor.count))
    return ast.unparse(ast.fix_missing_locations(tree))
def main():
    from compile_catalog import build_runtime
    from smoke_catalog import bind_fields
    from factor_engine.api.factor import Factor
    parser,engine=build_runtime()
    rows=list(map(json.loads,gzip.open(E/"r20-merged-final.jsonl.gz","rt")))
    overall=total([ref("ShareRatio",r) for r in range(1,11)])
    group=ref("ShareholderId",1);weight=ref("ShareRatio",1)
    bands=[total([ref("ShareRatio",r),ref("ShareRatio",r+1)]) for r in range(1,11,2)]
    band_js="holder_class_js_shift("+", ".join(bands+[f"delay({x}, 1)" for x in bands])+")"
    category=f"safe_div_null({overall}, group_sum({overall}, {group}, fallback_policy='nan'))"
    peer=f"safe_div_null(safe_div_null(subtract(group_sum(multiply({weight}, {weight}), {group}, fallback_policy='nan'), multiply({weight}, {weight})), subtract(group_sum({weight}, {group}, fallback_policy='nan'), {weight})), gt({weight}, 0))"
    out=[]
    for base in rows:
        ident=base["id"]; formula=base["current_formula"]; note=None
        if ident in {f"R66_{n:05d}" for n in range(1289,1295)}:
            formula=replace_call(formula,"relation_hhi",parse(f"relation_category_share({overall}, {group})"))
            note="R20_CURRENT_DEFINITION: 原草稿缺少类别参数；明确以第一大股东ShareholderId定义同一持有人股票组，以各股票前十持股比例之和作为非负value，计算该value占同组总和的比例。恢复relation_category_share，不再替换为HHI。这是已明确分组与value的研究重定义，非原十个位置参数的等价展开；横截面只含本次研究股票池。"
        elif ident in {f"R66_{n:05d}" for n in range(1307,1313)}:
            formula=replace_call(formula,"relation_hhi",parse(f"relation_peer_weighted_mean_ex_self({overall}, {weight}, {group})"))
            note="R20_CURRENT_DEFINITION: 原草稿缺value/weight/group；明确同一第一大股东ID股票组，value=前十持股比例之和，weight=第一大股东持股比例，计算剔除本股票后的同组加权均值。保留peer/ex-self算子，不再换成HHI；单股票组无peer返回缺失，不填0；这是明确口径的研究重定义，横截面仅本次股票池。"
        elif ident in {f"R65X_{n:05d}" for n in range(1102,1108)}|{"R65X_01138"}:
            formula=replace_call(formula,"holder_top10_daily_id_churn",parse(band_js))
            note="R20_CURRENT_DEFINITION: 原草稿没有五类股东分类聚合槽及上一有效披露快照。明确改为五个固定名次组(1-2/3-4/5-6/7-8/9-10)的持股比例分布，与前一交易日同名次组分布计算JS距离，保留分布漂移数学含义；不是股东自然人/机构类别漂移，也不是上一报告期或ID churn。各组必须两名都有观测，缺失不填0。原类别版仍需额外分类聚合与历史PIT物化；此处是非等价、明确标注的可编译研究替代。"
        elif ident in {f"R66_{n:05d}" for n in range(1320,1339,2)}:
            note="R20_DATA_REQUIREMENT: report_revision_magnitude只在来源保留逐日PIT历史版本且同报告期的前日值确为上一可见版本时可解释为修订；最新快照覆写数据不能复原历史修订。编译通过不解除此数据准入条件，不得当作真实历史修订已验证。"
        if note is None:continue
        row=dict(base);row.update(current_formula=formula,changes=list(base["changes"])+[note],current_definition=note,semantic_redesign=True)
        expr=parser.parse(formula);bindings,failures=bind_fields(expr)
        if failures:raise ValueError((ident,failures))
        engine.compile(Factor(name=ident,expr=expr,source_expr=formula,surface="compat_research"))
        row.update(status="COMPILED",error="",bindings=bindings)
        out.append(row)
    if len(out)!=29:raise ValueError(len(out))
    dst=E/"r20_relation_semantic_delta_proposals.jsonl"
    with dst.open("x") as f:
        for row in out:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps({"rows":len(out),"compiled":len(out),"changed":sum(x["current_formula"]!=next(r["current_formula"] for r in rows if r["id"]==x["id"]) for x in out)},ensure_ascii=False))
if __name__=="__main__":main()
