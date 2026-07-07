#!/usr/bin/env python3
"""
异步报告生成器
生成更新报告，支持异步处理以避免阻塞主流程
"""

import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class AsyncReporter:
    """异步报告生成器"""
    
    def __init__(self, report_dir: Optional[Path] = None, enable_async: bool = True):
        self.enable_async = enable_async
        self.report_dir = report_dir or Path("/home/yluel/share/projects/quantsociety_backend_project/user_workspace/lfl_workspace/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # 报告文件路径
        self.daily_report_file = self.report_dir / f"daily_update_report_{datetime.now().strftime('%Y%m%d')}.json"
        self.summary_file = self.report_dir / "update_summary.jsonl"
        
        # 异步任务队列
        self._task_queue = asyncio.Queue() if enable_async else None
        self._worker_task = None
        
        if enable_async:
            # 启动异步工作线程
            self._worker_task = asyncio.create_task(self._report_worker())
    
    async def report_source_update(self, result) -> None:
        """报告数据源更新结果"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'source_name': result.source_name,
            'status': 'success' if result.success else 'failed',
            'records_processed': result.records_cleaned,
            'quality_score': result.quality_score,
            'processing_time_seconds': result.processing_time,
            'error_message': result.error_message,
            'warnings': result.warnings
        }
        
        if self.enable_async:
            await self._task_queue.put(report_data)
        else:
            await self._write_single_report(report_data)
    
    async def generate_report(self, source_name: str, status: str, 
                            records_processed: int, quality_score: float,
                            processing_time: float) -> None:
        """生成单个数据源的更新报告"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'source_name': source_name,
            'status': status,
            'records_processed': records_processed,
            'quality_score': quality_score,
            'processing_time_seconds': processing_time
        }
        
        if self.enable_async:
            await self._task_queue.put(report_data)
        else:
            await self._write_single_report(report_data)
    
    async def report_error(self, source_name: str, error_message: str) -> None:
        """报告错误信息"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'source_name': source_name,
            'status': 'failed',
            'records_processed': 0,
            'quality_score': 0.0,
            'processing_time_seconds': 0.0,
            'error_message': error_message
        }
        
        if self.enable_async:
            await self._task_queue.put(report_data)
        else:
            await self._write_single_report(report_data)
    
    async def generate_final_report(self, summary: Dict[str, Any]) -> None:
        """生成最终汇总报告"""
        if self.enable_async:
            # 发送结束信号
            await self._task_queue.put({'type': 'final_report', 'data': summary})
        else:
            await self._write_final_report(summary)
    
    async def _report_worker(self):
        """异步报告工作线程"""
        while True:
            try:
                task_data = await self._task_queue.get()
                
                if task_data.get('type') == 'final_report':
                    await self._write_final_report(task_data['data'])
                    break
                else:
                    await self._write_single_report(task_data)
                    
            except Exception as e:
                logger.error(f"报告工作线程异常: {e}")
    
    async def _write_single_report(self, report_data: Dict[str, Any]) -> None:
        """写入单个报告"""
        try:
            # 追加到汇总文件
            with open(self.summary_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(report_data, ensure_ascii=False) + '\n')
            
            logger.info(f"报告生成: {report_data['source_name']} - "
                       f"状态: {report_data['status']}, "
                       f"记录数: {report_data['records_processed']}, "
                       f"质量分数: {report_data['quality_score']:.2f}")
            
        except Exception as e:
            logger.error(f"写入报告失败: {e}")
    
    async def _write_final_report(self, summary: Dict[str, Any]) -> None:
        """写入最终汇总报告"""
        try:
            # 创建可JSON序列化的摘要副本
            summary_copy = {}
            for key, value in summary.items():
                if key == 'start_time' and value:
                    summary_copy[key] = value.isoformat()
                elif key == 'end_time' and value:
                    summary_copy[key] = value.isoformat()
                elif (key == 'results' or key == 'details') and value:
                    # 兼容处理：支持字典格式结果和UpdateResult对象
                    summary_copy[key] = []
                    for result in value:
                        if isinstance(result, dict):
                            # 处理字典格式的结果（主控器返回的格式）
                            summary_copy[key].append({
                                'source_name': result.get('source', 'unknown'),
                                'success': result.get('status') == 'success',
                                'skipped': result.get('status') == 'skipped_by_ok',
                                'status': result.get('status', 'unknown'),
                                'records_fetched': result.get('records', 0),
                                'records_cleaned': result.get('records', 0),
                                'quality_score': result.get('quality_score', 0.0),
                                'processing_time': result.get('duration_seconds', 0.0),
                                'error_message': result.get('error', None),
                                'warnings': []
                            })
                        else:
                            # 处理UpdateResult对象（原有逻辑）
                            summary_copy[key].append({
                                'source_name': getattr(result, 'source_name', 'unknown'),
                                'success': getattr(result, 'success', False),
                                'skipped': getattr(result, 'skipped', False),
                                'status': getattr(result, 'status', 'unknown'),
                                'records_fetched': getattr(result, 'records_fetched', 0),
                                'records_cleaned': getattr(result, 'records_cleaned', 0),
                                'quality_score': getattr(result, 'quality_score', 0.0),
                                'processing_time': getattr(result, 'processing_time', 0.0),
                                'error_message': getattr(result, 'error_message', None),
                                'warnings': getattr(result, 'warnings', [])
                            })
                else:
                    summary_copy[key] = value
            
            # 生成详细的最终报告
            final_report = {
                'report_generated_at': datetime.now().isoformat(),
                'update_summary': summary_copy,
                'status_overview': {
                'total_sources': summary['total_sources'],
                'successful': summary['successful_updates'],
                'failed': summary['failed_updates'],
                'skipped': summary['skipped_updates'],
                'success_rate': summary['successful_updates'] / summary['total_sources'] if summary['total_sources'] > 0 else 0
            },
                'performance_metrics': {
                    'total_processing_time': summary['total_processing_time'],
                    'total_records_processed': summary['total_records'],
                    'average_processing_time_per_source': summary['total_processing_time'] / summary['total_sources'] if summary['total_sources'] > 0 else 0
                }
            }
            
            # 保存到每日报告文件
            with open(self.daily_report_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            
            # 同时生成人类可读的报告
            await self._generate_human_readable_report(final_report)
            
            logger.info(f"最终报告生成完成: {self.daily_report_file}")
            
        except Exception as e:
            logger.error(f"生成最终报告失败: {e}")
    
    async def _generate_human_readable_report(self, final_report: Dict[str, Any]) -> None:
        """生成人类可读的报告"""
        summary = final_report['update_summary']
        
        start_time = summary.get('start_time')
        end_time = summary.get('end_time')
        readable_report = f"""
=====================================
每日增量数据更新报告
生成时间: {final_report['report_generated_at']}
=====================================

"""
        if start_time and end_time:
            if isinstance(start_time, datetime) and isinstance(end_time, datetime):
                duration = (end_time - start_time).total_seconds()
                readable_report += f"⏱️  处理时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                readable_report += f"⌛ 总耗时: {duration:.2f}秒\n\n"
            elif isinstance(start_time, str) and isinstance(end_time, str):
                try:
                    s_time = datetime.fromisoformat(start_time)
                    e_time = datetime.fromisoformat(end_time)
                    duration = (e_time - s_time).total_seconds()
                    readable_report += f"⏱️  处理时间: {s_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {e_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    readable_report += f"⌛ 总耗时: {duration:.2f}秒\n\n"
                except:
                    pass
        
        readable_report += f"""📊 更新概况:
- 总数据源数量: {summary['total_sources']}
- 成功更新: {summary['successful_updates']} ✅
- 失败更新: {summary['failed_updates']} ❌
- 跳过更新: {summary['skipped_updates']} ⏭️
- 成功率: {final_report['status_overview']['success_rate']:.1%}

📈 性能指标:
- 总处理时间: {summary['total_processing_time']:.2f} 秒
- 总处理记录数: {summary['total_records']:,}
- 平均每个数据源处理时间: {final_report['performance_metrics']['average_processing_time_per_source']:.2f} 秒

📋 详细更新情况:
"""
        
        # 预处理详情数据，统一字典格式
        details_to_use = []
        source_list = None
        if 'details' in summary and summary['details']:
            source_list = summary['details']
        elif 'results' in summary and summary['results']:
            source_list = summary['results']
        
        if source_list:
            for item in source_list:
                if isinstance(item, dict):
                    # 如果已经是字典，直接处理
                    if 'source' in item:
                        details_to_use.append({
                            'source_name': item.get('source', 'unknown'),
                            'status': item.get('status', 'unknown'),
                            'records': item.get('records', 0),
                            'quality_score': item.get('quality_score', 0.0),
                            'error': item.get('error', None),
                            'warnings': item.get('warnings', [])
                        })
                    else:
                        # 已经是兼容处理后的格式
                        details_to_use.append(item)
                else:
                    # 处理UpdateResult对象
                    details_to_use.append({
                        'source_name': getattr(item, 'source_name', 'unknown'),
                        'status': getattr(item, 'status', 'unknown'),
                        'records': getattr(item, 'records_fetched', 0),
                        'quality_score': getattr(item, 'quality_score', 0.0),
                        'error': getattr(item, 'error_message', None),
                        'warnings': getattr(item, 'warnings', [])
                    })
        
        if details_to_use:
            for detail in details_to_use:
                # 根据状态选择正确的emoji
                if detail['status'] == 'success':
                    status_emoji = "✅"
                elif detail['status'] == 'skipped_by_ok':
                    status_emoji = "⏭️"
                elif detail['status'] == 'no_data':
                    status_emoji = "ℹ️"
                elif detail['status'] == 'massive_not_ready':
                    status_emoji = "⏳"
                elif detail['status'] == 'quality_failed':
                    status_emoji = "📉"
                elif detail['status'] in ['failed', 'exception']:
                    status_emoji = "❌"
                else:
                    status_emoji = "❓"
                readable_report += f"\n{status_emoji} {detail['source_name']}: {detail['status']}"
                  
                if detail['status'] == 'success':
                    readable_report += f" (记录数: {detail['records_fetched']}, 质量分数: {detail['quality_score']:.2f})"
                elif detail['status'] in ['failed', 'no_data', 'exception']:
                    readable_report += f" (错误: {detail.get('error_message', '未知错误')})"
                
                if detail.get('warnings'):
                    readable_report += f"\n   ⚠️  警告: {', '.join(detail['warnings'])}"
        elif 'results' in summary and summary['results']:
            for result in summary['results']:
                status_emoji = "✅" if result['success'] else "❌"
                readable_report += f"\n{status_emoji} {result['source_name']}: {'success' if result['success'] else 'failed'}"
                
                if result['success']:
                    readable_report += f" (记录数: {result['records_cleaned']}, 质量分数: {result['quality_score']:.2f})"
                else:
                    readable_report += f" (错误: {result['error_message']})"
                
                if result['warnings']:
                    readable_report += f"\n   ⚠️  警告: {', '.join(result['warnings'])}"
        
        readable_report += "\n=====================================\n"
        
        # 保存人类可读报告
        readable_file = self.daily_report_file.with_suffix('.txt')
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(readable_report)
        
        logger.info(f"人类可读报告生成: {readable_file}")
    
    def get_recent_reports(self, days: int = 7) -> List[Dict[str, Any]]:
        """获取最近几天的报告"""
        reports = []
        
        for i in range(days):
            report_date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            report_file = self.report_dir / f"daily_update_report_{report_date}.json"
            
            if report_file.exists():
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        reports.append(json.load(f))
                except Exception as e:
                    logger.error(f"读取历史报告失败 {report_file}: {e}")
        
        return reports
    
    def generate_trend_report(self, days: int = 30) -> Dict[str, Any]:
        """生成趋势报告"""
        recent_reports = self.get_recent_reports(days)
        
        if not recent_reports:
            return {'error': '没有足够的历史报告数据'}
        
        # 提取趋势数据
        trend_data = {
            'dates': [],
            'success_rates': [],
            'total_records': [],
            'processing_times': [],
            'failed_sources': []
        }
        
        for report in reversed(recent_reports):  # 从早到晚
            summary = report['update_summary']
            trend_data['dates'].append(summary['timestamp'][:10])  # 只取日期部分
            trend_data['success_rates'].append(report['status_overview']['success_rate'])
            trend_data['total_records'].append(summary['total_records'])
            trend_data['processing_times'].append(summary['total_processing_time'])
            
            # 统计失败的数据源
            failed_sources = [d['source'] for d in summary['details'] if d['status'] == 'failed']
            trend_data['failed_sources'].append(failed_sources)
        
        # 计算趋势指标
        import numpy as np
        
        trend_report = {
            'period_days': days,
            'average_success_rate': np.mean(trend_data['success_rates']),
            'success_rate_trend': 'improving' if trend_data['success_rates'][-1] > trend_data['success_rates'][0] else 'declining',
            'average_daily_records': np.mean(trend_data['total_records']),
            'average_processing_time': np.mean(trend_data['processing_times']),
            'most_problematic_sources': self._identify_problematic_sources(trend_data['failed_sources']),
            'recommendations': self._generate_recommendations(trend_data)
        }
        
        return trend_report
    
    def _identify_problematic_sources(self, failed_sources_list: List[List[str]]) -> List[str]:
        """识别问题最多的数据源"""
        source_failure_count = {}
        
        for failed_sources in failed_sources_list:
            for source in failed_sources:
                source_failure_count[source] = source_failure_count.get(source, 0) + 1
        
        # 按失败次数排序
        problematic_sources = sorted(source_failure_count.items(), key=lambda x: x[1], reverse=True)
        return [source for source, count in problematic_sources[:5]]  # 返回前5个问题源
    
    def _generate_recommendations(self, trend_data: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 检查成功率趋势
        recent_success_rate = trend_data['success_rates'][-1]
        if recent_success_rate < 0.8:  # 成功率低于80%
            recommendations.append("成功率较低，建议检查失败数据源的日志和配置")
        
        # 检查处理时间趋势
        recent_time = trend_data['processing_times'][-1]
        average_time = sum(trend_data['processing_times']) / len(trend_data['processing_times'])
        if recent_time > average_time * 1.5:  # 最近处理时间明显高于平均
            recommendations.append("处理时间增加，可能需要优化性能或增加资源")
        
        # 检查数据量变化
        recent_records = trend_data['total_records'][-1]
        average_records = sum(trend_data['total_records']) / len(trend_data['total_records'])
        if recent_records > average_records * 2:  # 数据量突然增加
            recommendations.append("数据量显著增加，建议检查是否有异常数据或扩大存储容量")
        
        if not recommendations:
            recommendations.append("系统运行正常，继续保持当前配置")
        
        return recommendations
    
    async def close(self):
        """关闭异步报告器"""
        if self.enable_async and self._worker_task:
            # 等待工作线程完成
            await self._task_queue.put({'type': 'shutdown'})
            await self._worker_task
    
    def __del__(self):
        """析构函数"""
        if self.enable_async and self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()