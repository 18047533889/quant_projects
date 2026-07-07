# 📅 数据更新定时任务脚本

> **⚠️ TASK-ENG-004（P0）— 当前已熔断**  
> 存在 `../CRON_PAUSED` 时：**禁止**安装 cron、**禁止**跑日更脚本。  
> 解除条件：TASK-DATA-002/003 PASS + TASK-ENG-001/002 + FEAT-008。  
> 验收：`bash verify_eng004.sh` → `PASS TASK-ENG-004`

本目录包含用于自动化每日增量数据更新的定时任务脚本（修复完成前默认关闭）。

## 🚀 快速开始

### 1. 安装定时任务
```bash
./install_cron_jobs.sh   # 熔断期间会 exit 1
```

### 2. 手动运行更新
```bash
./daily_update_scheduler.sh   # 熔断期间会 exit 1
./minute_update_scheduler.sh  # 熔断期间会 exit 1
```

## 📋 脚本说明

### daily_update_scheduler.sh
- **功能**：执行每日数据更新任务
- **运行时间**：建议 5:00 AM EST（北京时间 17:00/18:00）
- **更新内容**：
  - 日线数据（daily_market_summary）
  - 基本面数据（财务报表、股息等）
  - 公司行为数据（分红、拆股等）
  - 交易所信息

### minute_update_scheduler.sh
- **功能**：执行分钟级数据更新任务
- **运行时间**：交易时间内每15分钟
- **更新内容**：
  - 分钟级行情数据
  - 实时交易数据

### install_cron_jobs.sh
- **功能**：安装和管理定时任务
- **选项**：
  1. 安装每日更新任务
  2. 安装分钟级更新任务
  3. 安装所有任务
  4. 查看定时任务说明
  5. 退出

## ⏰ 定时任务配置

### 默认配置
```bash
# 每日数据更新 - 5:00 AM EST
0 5 * * * /home/yluel/share/projects/quantsociety_backend_project/raw_data_layer/data_daily_update/scripts/daily_update_scheduler.sh >> /home/yluel/share/projects/quantsociety_backend_project/raw_data_layer/data_daily_update/scripts/cron_daily.log 2>&1

# 分钟级数据更新 - 交易时间内每15分钟
*/15 9-16 * * 1-5 /home/yluel/share/projects/quantsociety_backend_project/raw_data_layer/data_daily_update/scripts/minute_update_scheduler.sh >> /home/yluel/share/projects/quantsociety_backend_project/raw_data_layer/data_daily_update/scripts/cron_minute.log 2>&1
```

### 时区说明
- **EST (Eastern Standard Time)**: 美国东部标准时间
- **北京时间转换**：
  - 夏令时（3月-11月）：EST + 12小时
  - 冬令时（11月-3月）：EST + 13小时

### 交易时间
- **美股交易时间**：9:30 AM - 4:00 PM EST
- **北京时间对应**：
  - 夏令时：21:30 - 04:00
  - 冬令时：22:30 - 05:00

## 📁 文件结构
```
scripts/
├── daily_update_scheduler.sh      # 每日更新脚本
├── minute_update_scheduler.sh     # 分钟级更新脚本
├── install_cron_jobs.sh          # 定时任务安装脚本
├── cron_daily.log                # 每日更新日志（自动生成）
├── cron_minute.log               # 分钟级更新日志（自动生成）
└── README.md                     # 本说明文档
```

## 📝 日志文件

### 定时任务日志
- **每日更新**：`cron_daily.log`
- **分钟级更新**：`cron_minute.log`

### 详细日志
- **位置**：`../logs/`
- **文件格式**：
  - `incremental_update_YYYYMMDD_HHMMSS.log`（详细更新日志）
  - `scheduler_YYYYMMDD_HHMMSS.log`（调度器日志）

### 报告文件
- **位置**：`../reports/`
- **文件格式**：
  - `daily_update_report_YYYYMMDD.json`（JSON格式报告）
  - `daily_update_report_YYYYMMDD.txt`（文本格式报告）

## ⚙️ 配置选项

### 环境变量
脚本会自动检测以下环境变量：
```bash
PYTHON_CMD        # Python命令（默认：python3）
WORKSPACE_DIR     # 工作目录（默认：脚本上级目录）
LOG_DIR           # 日志目录（默认：../logs/）
REPORT_DIR        # 报告目录（默认：../reports/）
```

### 自定义配置
您可以在脚本开头修改以下参数：
```bash
# Python命令
PYTHON_CMD="python3"

# 工作目录
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 日志保留天数
LOG_RETENTION_DAYS=30
```

## 🔧 故障排除

### 常见问题

#### 1. 权限问题
```bash
# 给脚本添加执行权限
chmod +x *.sh
```

#### 2. Python环境问题
```bash
# 检查Python版本
python3 --version

# 检查依赖包
pip3 list | grep -E "pandas|pyarrow|requests"
```

#### 3. 定时任务不执行
```bash
# 检查cron服务状态
sudo systemctl status cron

# 查看当前用户的定时任务
crontab -l

# 检查脚本路径是否正确
realpath daily_update_scheduler.sh
```

#### 4. 日志文件权限
```bash
# 创建日志目录
mkdir -p ../logs ../reports

# 设置正确的权限
chmod 755 ../logs ../reports
```

### 手动调试
```bash
# 手动运行并查看详细输出
bash -x daily_update_scheduler.sh

# 检查配置文件
ls -la ../configs/
cat ../configs/incremental_update_config.yaml
```

## 📊 监控和告警

### 状态检查
```bash
# 查看最近的更新状态
tail -n 50 cron_daily.log
tail -n 50 cron_minute.log

# 检查今日报告
ls -la ../reports/daily_update_report_$(date +%Y%m%d).*
```

### 性能监控
脚本会自动记录：
- ✅ 更新成功/失败状态
- ⏱️ 更新耗时
- 📈 数据源统计
- 🔄 重试次数
- ⚠️ 错误信息

### 告警配置（可选）
在脚本中可以配置邮件告警：
```bash
# 在 daily_update_scheduler.sh 中取消注释
mail -s "数据更新失败告警" admin@example.com < "$LOG_FILE"
```

## 🔄 更新和维护

### 更新脚本
```bash
# 备份当前配置
cp crontab -l > cron_backup_$(date +%Y%m%d).txt

# 重新安装定时任务
./install_cron_jobs.sh
```

### 清理旧日志
```bash
# 手动清理30天前的日志
find ../logs -name "*.log" -mtime +30 -delete
find . -name "cron_*.log" -mtime +30 -delete
```

### 停用定时任务
```bash
# 移除所有定时任务
crontab -r

# 或者编辑定时任务
crontab -e
```

## 📚 相关文档

- [增量更新系统README](../README.md)
- [配置文件说明](../configs/)
- [API文档](../../raw_data_fetching/rest_api_doc/)
- [测试报告](../../user_workspace/lfl_workspace/comprehensive_test_report.md)
