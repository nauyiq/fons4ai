# BUG 修复报告

> Bug: `skill-frontmatter-bom-not-detected`
> Status: Verified
> Completed: 2026-06-22

## 基本信息

- 模块/功能：Fons4AI 全局技能与仓库技能识别
- 严重级别：中
- 严重级别依据：SDD 主流程技能无法被 Codex 扫描展示，影响需求、设计、任务、实现类工作流入口。
- 影响范围：`fons4ai-sdd-change`、`fons4ai-sdd-design`、`fons4ai-sdd-implement`、`fons4ai-sdd-requirements`、`fons4ai-sdd-tasks`
- 报告人：Codex

## 问题描述

- 期望结果：Codex 能扫描到全局与仓库内全部 Fons4AI 技能。
- 实际结果：界面只能显示 4 个本地/个人技能，5 个 `fons4ai-sdd-*` 技能未显示。
- 首次发现时间：2026-06-22
- 触发频率：必现

## 复现步骤

1. 打开 Codex 技能列表。
2. 查看 Fons4AI 技能展示结果。
3. 观察到仅显示非 SDD 的 4 个技能，未显示 5 个 `fons4ai-sdd-*` 技能。

## 复现环境

- 环境/版本：Codex Desktop，本机 Windows 工作区
- 账号/角色/权限：本机用户技能目录 `C:\Users\chuang_ying_h\.agents\skills`
- 配置/依赖/外部条件：技能入口文件为各技能目录下的 `SKILL.md`
- 日志/截图/报错信息：用户截图显示仅有 4 个 Fons4AI 技能；字节级检查显示未识别文件以 `EF BB BF` 开头。

## 根因分析

- 关键线索：已识别的 4 个技能 `SKILL.md` 第一字节为 `2D 2D 2D`，即 front matter 分隔符 `---`；未识别的 5 个技能第一字节为 `EF BB BF 2D 2D 2D`。
- 排查路径：对比全局目录 `C:\Users\chuang_ying_h\.agents\skills` 与仓库目录 `skills` 下 9 个技能的 `SKILL.md` 文件头。
- 根因说明：5 个 `fons4ai-sdd-*` 技能文件带 UTF-8 BOM，导致 Codex 技能扫描器无法从文件起始位置识别 YAML front matter。
- 是否属于需求变更：否

## 修复方案

- 修复策略：移除 5 个目标技能 `SKILL.md` 开头的 UTF-8 BOM 三字节。
- 最小改动说明：只删除文件开头 `EF BB BF`，保留后续全部字节不变。
- 影响评估：技能正文、名称、描述、换行和工作流内容不变；只影响扫描器识别入口。
- 风险点：需要重启或刷新 Codex 技能扫描后，界面才可能重新加载最新技能列表。
- 回滚方案：如需回滚，可从 Git 历史或备份恢复对应 `SKILL.md` 文件；不涉及数据库、依赖或业务逻辑。

## 变更文件

- `C:\Users\chuang_ying_h\.agents\skills\fons4ai-sdd-change\SKILL.md`：移除 UTF-8 BOM。
- `C:\Users\chuang_ying_h\.agents\skills\fons4ai-sdd-design\SKILL.md`：移除 UTF-8 BOM。
- `C:\Users\chuang_ying_h\.agents\skills\fons4ai-sdd-implement\SKILL.md`：移除 UTF-8 BOM。
- `C:\Users\chuang_ying_h\.agents\skills\fons4ai-sdd-requirements\SKILL.md`：移除 UTF-8 BOM。
- `C:\Users\chuang_ying_h\.agents\skills\fons4ai-sdd-tasks\SKILL.md`：移除 UTF-8 BOM。
- `skills/fons4ai-sdd-change/SKILL.md`：同步移除 UTF-8 BOM。
- `skills/fons4ai-sdd-design/SKILL.md`：同步移除 UTF-8 BOM。
- `skills/fons4ai-sdd-implement/SKILL.md`：同步移除 UTF-8 BOM。
- `skills/fons4ai-sdd-requirements/SKILL.md`：同步移除 UTF-8 BOM。
- `skills/fons4ai-sdd-tasks/SKILL.md`：同步移除 UTF-8 BOM。
- `specs/bugfixes/skill-frontmatter-bom-not-detected/bugfix-report.md`：记录本次修复。

## 自动化测试

- RED 证据：修复前字节级检查显示 5 个 `fons4ai-sdd-*` 技能文件 `HasBom=True`，对应 Codex 技能列表未显示这些技能。
- 新增/更新测试：无；本问题属于文件编码头元数据修复。
- 测试命令：
  - `Get-ChildItem C:\Users\chuang_ying_h\.agents\skills -Directory | ... | Format-Table -AutoSize`
  - `Get-ChildItem skills -Directory | ... | Format-Table -AutoSize`
- 测试结果：全局目录与仓库目录下 9 个技能 `HasBom=False`，5 个 SDD 技能文件第一字节已从 `EF BB BF` 变为 `2D 2D 2D`。
- 若无法自动化测试，原因：Codex 技能 UI 的重新扫描结果需要客户端刷新或重启后人工确认。

## 手动验证

1. 刷新或重启 Codex Desktop。
2. 打开技能列表。
3. 搜索或查看 Fons4AI 技能。

预期结果：除原本可见的 4 个技能外，`fons4ai-sdd-change`、`fons4ai-sdd-design`、`fons4ai-sdd-implement`、`fons4ai-sdd-requirements`、`fons4ai-sdd-tasks` 也能被扫描展示。

## 回归验证

- 回归范围：全部 9 个 Fons4AI 技能的 `SKILL.md` 文件头。
- 验证命令或步骤：分别检查全局目录与仓库目录下所有 `SKILL.md` 是否仍以 `---` 开头且不含 BOM。
- 验证结果：已通过字节级检查，全部 `HasBom=False`。

## 知识库同步

- Knowledge Sync Needed: no
- 影响的真理源：无
- SQL DDL files: no
- DDL grouping: 不适用
- Suggested follow-up: none

## 后续事项

- 刷新或重启 Codex Desktop，确认技能列表重新展示全部 9 个 Fons4AI 技能。
