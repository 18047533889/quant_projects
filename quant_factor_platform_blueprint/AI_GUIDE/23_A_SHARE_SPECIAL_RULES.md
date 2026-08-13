# 23 — A-share-specific Rules for This Platform

本页用于防止通用开源框架语义直接搬到 A 股。

## Data/Timing

- 财报类特征严格使用实际可知/公告时间语义；不要用报告期末日期替代 availability。
- 单季度/TTM 必须由当时可知累计值因果构造。
- UpdateTime/文件更新时间不能替代经济意义上的 publication time。
- 分钟数据只能使用真实已有 bar；没有集合竞价/Level2 就不虚构。

## Tradability

QE 需要支持 context：

- suspend
- ST/status
- actual high/low limit
- can_buy / can_sell / can_hold
- board / IPO age

不要把 “raw factor IC” 和 “可交易 IC” 混在一起。

## T+1 / Label

Factor at EOD t 的完整日信号，一般不能假设按同日收盘前已知并成交。EvaluationContext 必须明确 decision/execution/label timing。

LabelProvider/ExecutionReturnProvider 由调用者/DA/执行假设提供；QE 不自行 `shift` 猜测。

## Liquidity / Microcap

A 股核心 robustness：

- size/free-float cap buckets
- turnover/liquidity buckets
- exclude bottom microcap slices
- benchmark constituent slices
- tradable vs raw

## Neutralization

行业表可能有多个来源时，context 构造端先选定明确行业体系；FP 不对重复行业行做隐式处理。

## Current Known Unsupported Production Domains

如果服务器数据没有新增，则不把以下当核心生产数据：

- Level2/order book
- analyst consensus/revisions
- full news/text sentiment
- true northbound flow

若未来 DataAccess 新增，先更新 semantic catalog/schema，再开放 Optimizer grammar。
