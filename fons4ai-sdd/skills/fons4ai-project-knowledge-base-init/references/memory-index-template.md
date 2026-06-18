# 项目知识库索引

> 项目名称：<project-name>
> 文档状态：初稿 | 已评审 | 待补充
> 更新日期：YYYY-MM-DD

## 1. 项目概览

- 系统定位：
- 核心业务线：
- 核心领域：
- 主要使用方：
- 关键约束：

## 2. 领域索引

| 领域 | 说明 | 业务文档 | 技术文档 | 数据文档 | SQL/DDL 参考 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| <domain-slug> | <领域说明> | `.specify/memory/domains/<domain-slug>/业务架构.md` | `.specify/memory/domains/<domain-slug>/技术架构.md` | `.specify/memory/domains/<domain-slug>/数据架构.md` | <已有 SQL 路径/MCP DDL 已确认不落盘/待确认/无> | 已确认/推断/待确认 |

## 3. 核心能力索引

| 能力 | 所属领域 | 说明 | 关键场景 | 关联卡片 |
| --- | --- | --- | --- | --- |
| <能力名称> | <domain-slug> | <能力说明> | BS-xxx | KC-BIZ-xxx |

## 4. 跨领域协作

| 协作链路 | 参与领域 | 触发场景 | 关键规则 | 状态 |
| --- | --- | --- | --- | --- |
| <链路名称> | <domain-a>/<domain-b> | <场景> | <规则> | 已确认/推断/待确认 |

## 5. 知识卡片索引

| 卡片编号 | 类型 | 标题 | 所属领域 | 状态 | 路径 |
| --- | --- | --- | --- | --- | --- |
| KC-BIZ-001 | 业务场景 | <标题> | <domain-slug> | 已确认/推断/待确认/已废弃 | `.specify/memory/domains/<domain-slug>/cards/KC-BIZ-001-<slug>.md` |

## 6. SQL/DDL 参考索引

| 数据库/服务 | 业务模型 | 归属领域 | 参考来源 | 处理状态 |
| --- | --- | --- | --- | --- |
| <database_or_service> | <business_model> | <domain-slug> | <已有 SQL 路径/MCP DDL 已确认不落盘/待确认/无> | 已确认/推断/待确认 |

## 7. 待确认事项

| 编号 | 问题 | 影响范围 | 建议处理 |
| --- | --- | --- | --- |
| Q-001 | <问题> | <领域/文档/SQL/DDL> | <处理方式> |
