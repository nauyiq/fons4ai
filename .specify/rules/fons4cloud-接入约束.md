# fons4cloud 接入约束

> 适用范围：下游业务仓接入 fons4cloud 框架
> 规则状态：已确认
> 更新日期：2026-08-13
> 依据来源：用户确认、fons4cloud 仓源码、`.specify/rules/代码编写规范.md`、`AGENTS.md`

## 适用说明与规则优先级

- 本规则约束下游业务仓对 fons4cloud 能力的复用、分层、校验、领域对象和技术支撑边界，不替代业务仓自身更具体的业务规则。
- 下游引入 fons4cloud 后，将本文件复制到业务仓 `.specify/rules/fons4cloud-接入约束.md`；版本升级时同步更新。
- 框架类名和包路径以业务仓实际引入版本为准，不得编造接口、字段或配置。
- 新增代码和发生实质修改的相关代码必须遵循本规则；历史代码不要求立即迁移。
- 冲突优先级：用户已确认规则 → 本接入约束 → 下游更严格且不绕过框架能力的规则 → 历史源码习惯和旧示例。

## 核心架构原则

统一采用：

> 声明式入口校验 + 行为型实体 + 同域聚合根 + 领域服务决策 + 应用层业务编排 + 基础设施技术运行编排。

- 框架已提供的能力必须复用，禁止平行实现。
- 鉴权、网关、Token、锁、MQ、限流、文件等关键链路不得绕过框架机制。
- 入口层校验输入结构，Application 编排业务用例，Domain 判断业务合法性，Infrastructure 编排技术能力。
- Entity 知道自身数据如何合法变化，但不知道数据从哪里来、如何持久化。
- Aggregate 只维护同一领域和同一一致性边界内的对象，不跨领域拥有可变实体。
- 引入新依赖前确认与框架技术栈兼容，并优先复用现有依赖链。

## 必须复用的框架能力

### 统一返回、错误码与异常

- HTTP/RPC 返回使用 `com.fons.cloud.common.result.R`，禁止新增平行返回体系。
- 通用错误码复用 `ResultCode`；业务错误码实现 `Result`，实现 `getCode()`/`getMessage()`，遵循 `前缀 + 错误类型(0-9) + 序号(00001-99999)`。
- 可预期业务失败使用 `BizException` 或项目领域异常；业务运行时异常使用 `BusinessRuntimeException`，系统内部异常使用 `SystemIntervalException` 并保留 cause。
- 禁止裸 `throw new RuntimeException`，禁止散落字符串错误或吞掉影响一致性的异常。

### 对象转换、锁、限流、文件与消息

- 新增 5 个以上字段映射或重复同类字段搬运时，优先使用 MapStruct、`CommonConverter` 或现有转换器；不使用必须说明原因。
- 分布式锁使用 `DistributeLock` 或 `LockService`；限流使用 `Limiter`；文件使用 `FileService`。
- 消息能力复用 fons4cloud MQ/Stream 抽象，不在业务模块平行实现中间件接入和事务消息骨架。

## 包结构与依赖边界

采用横向技术分层，可按复杂度增加领域二级包：

```text
com.fons.cloud.{业务标识}
├─ controller
├─ application
├─ common
│  ├─ constants
│  ├─ model
│  ├─ request
│  ├─ response
│  ├─ dto
│  ├─ vo
│  └─ util
├─ domain
│  ├─ entity
│  │  └─ user
│  │     ├─ User
│  │     ├─ UserRole
│  │     └─ UserAggregate
│  ├─ mapper
│  │  └─ user
│  └─ service
│     └─ user
└─ infrastructure
   ├─ config
   ├─ converter
   ├─ adapter
   ├─ client
   └─ support
```

- `controller → application → domain`。
- `application → infrastructure`，Application 使用技术能力完成业务用例。
- `domain.service → domain.mapper`，Domain Service 只能直接访问本领域 Mapper。
- Controller 负责协议转换、声明式参数校验和调用 Application，不编排业务。
- Application 负责跨领域协作、事务、幂等、补偿和业务执行顺序。
- Domain Entity 负责行为与不变量，Domain Service 查询本领域实体、装配聚合根和执行复杂领域判断。
- Infrastructure 负责框架集成、外部适配和技术运行编排，不反向决定领域规则。
- Common 放跨层模型、契约、常量和无状态纯工具，不承载领域状态流转。

## 声明式参数校验

### 外部入口

- 所有外部入口必须在进入 Application 前优先完成 Jakarta Validation 校验。
- HTTP Request 字段使用 `@NotNull`、`@NotBlank`、`@Size`、`@Pattern`、数值范围等约束，Controller 参数使用 `@Valid`。
- `PathVariable`、`RequestParam` 和方法参数约束由 Controller 配合 `@Validated` 触发。
- RPC/Facade Request 复用 Jakarta Validation 约束，由 RPC Filter 或统一 `BeanValidator` 触发。
- MQ、任务和事件消费由入口 Adapter 在调用 Application 前触发校验。
- 嵌套对象和集合元素使用级联 `@Valid`；配置对象使用 `@Validated` 和字段约束。
- 参数校验异常统一交给 fons4cloud 全局异常处理，Controller 不重复捕获。

### 校验职责

| 校验类型 | 负责位置 |
| --- | --- |
| 必填、判空、长度、格式、范围、集合数量、嵌套结构 | Request 约束和入口层 |
| 依赖数据库、权限、幂等或外部状态的业务前置条件 | Application 或 Domain Service |
| 实体不变量、聚合一致性、状态能否迁移 | Entity 或 Aggregate |

- Application 不得重复编写已由 Request 注解覆盖的简单判空和格式校验。
- 纯请求结构的跨字段约束可以使用类级自定义 Validator。
- 数据唯一性、权限、状态迁移、聚合一致性、外部资源可用性等规则不得伪装成普通参数注解。
- 创建、修改、查询语义明显不同时，优先拆分 `XxxCreateRequest`、`XxxUpdateRequest`、`XxxQueryRequest`；字段与约束高度一致时才使用 Validation Groups。
- Request 不得兼任 Response、Entity 或持久化对象；错误响应不得泄露堆栈、内部类名和敏感值。

## 行为型实体与聚合根

### Entity

- 实体类不使用 `Entity` 后缀，例如 `User`、`UserRole`、`InvestigationReport`。
- 实体可以封装初始化、恢复后校验、数据变更、局部业务判断、状态判断、状态流转和自身不变量。
- 实体不得注入 Bean，不得查询 Mapper，不得调用数据库、缓存、MQ、Agent、文件或远程服务，不得管理事务或编排跨领域流程。
- 外部不得通过 Setter 绕过实体行为直接拼装复杂状态。

### Aggregate

- 聚合根使用 `Aggregate` 后缀，例如 `UserAggregate`、`OrderAggregate`。
- 聚合根管理同一领域、同一事务一致性边界内的多个实体，对外提供统一入口，协调内部实体行为并维护跨实体不变量。
- 内部实体继续负责局部规则；聚合根调用 `user.disable()`、`userRole.revoke()` 等行为，不越过实体直接修改字段。
- 外部不得绕过聚合根修改受聚合一致性保护的内部实体。
- 聚合根不得包含其他领域的可变实体，不得查询 Mapper、调用外部能力、承担跨领域编排或无限扩大边界。
- 跨领域只传递标识、不可变 Snapshot、Context、Result 或领域事件。

## 持久化策略

- 简单实体可以携带最小 MyBatis 映射注解，并由 `domain.mapper` 查询和持久化。
- 映射注解只能表达表和字段；实体不得出现 Mapper、Wrapper、SQL 构造、事务或基础设施调用。
- 复杂聚合在 `domain.entity.<domain>` 中放实体与 `XxxAggregate`；Mapper 查询实体，Domain Service 装配聚合根。
- 聚合根完成行为后，Domain Service 使用本领域 Mapper 保存变化实体；聚合根不要求直接映射到单表。
- 不强制所有场景完全分离领域对象与持久化对象；确需分离时必须保留明确转换边界。
- Request、Response、DTO、VO 不得直接作为持久化实体。

## Domain、Application 与 Infrastructure

### Domain Service

- 可以注入本领域 Mapper，查询和保存本领域实体，装配或恢复聚合根，执行同领域复杂判断、策略选择并调用聚合行为。
- 不得查询其他领域 Mapper，不得调用远程服务、MQ、Redis、Agent或文件服务，不得编排跨领域顺序或重复实体/聚合根规则。

### Application Service

- 负责跨领域业务编排、多个 Domain Service 协作、Infrastructure 调用、事务、幂等、补偿和失败协调。
- 负责将 Infrastructure 技术结果转换成领域输入。
- 不得重复 Request 注解已覆盖的校验，不得通过 Setter 修改实体状态，不得复制状态机或堆积本应属于 Domain 的复杂判断。

### Infrastructure

- 可以编排缓存、锁、MQ、远程调用、文件、搜索、Agent运行、工具调用、上下文压缩、状态恢复、线程调度、重试、超时和熔断等技术步骤。
- 不得判断业务状态是否合法，不得作出审批、发布、完成等业务结论，不得编排跨领域业务顺序或直接修改实体状态。
- Infrastructure 编排技术能力，Application 编排业务能力，Domain 判断业务合法性。

## Common 包与工具类

### Common 包

- `common` 允许 `constants`、`model`、`request`、`response`、`dto`、`vo`、`util` 等语义明确的子包，不作机械限制。
- 内部跨层结构优先 `common.model`，HTTP 输入优先 `common.request`，HTTP 输出优先 `common.response`。
- 复杂传输、RPC 或集成对象可以使用 DTO；页面视图对象可以使用 VO。
- 状态、类型、原因码和固定语义放 `common.constants.<domain>`；状态枚举只表达稳定值、编码和说明，不执行状态迁移、查询或外部调用。

### 工具类

- 无 Bean 注入、无外部资源访问、无业务状态的纯工具类放 `common.util`。
- 依赖 Spring Bean、配置、数据库、Redis、MQ、文件、远程接口或包含领域判断/状态迁移的能力不得放入 `common.util`。
- 工具优先级：Apache Commons 常见操作 → fons4cloud 现有工具/组件 → Hutool 等专项能力 → 确认无复用能力后新增工具类。
- 字符串判空优先 `StringUtils.isBlank/isNotBlank`，集合判空优先 `CollectionUtils.isEmpty/isNotEmpty`，Map 判空优先 `MapUtils.isEmpty/isNotEmpty`，数组操作优先 `ArrayUtils`。
- 构造器不变量允许使用 `Objects.requireNonNull`。
- Hutool 可用于日期、转换、缓存、JSON 等既有专项场景，不全面禁止；同一模块同类功能避免多套工具混用。
- 不为单次调用强行引入新依赖；需要的 Commons 模块应由依赖链提供或在模块 POM 明确声明。

## 鉴权、数据与安全约束

- 业务按实际选型遵循 `fons4cloud-auth-core`、Sa-Token 或 Spring Security 对应契约。
- 接入 auth-core 时，认证用户使用 `AuthUserHeaderConstants.AUTH_USER` 传递，客户端不得伪造；当前用户上下文使用 `AuthUtils`。
- 权限资源使用 `@AuthenticationResource`，预认证使用 `@PreAuthentication`；Token 生命周期由选定认证方案负责。
- 数据访问优先复用 `fons4cloud-common-db` 的 MyBatis-Plus 配置、公共 Mapper 和自动填充能力。
- 事务边界放在 Application 或明确事务服务；跨数据库、MQ、Redis 和外部服务必须说明补偿与失败处理。
- 更新、删除和批量操作必须有条件约束，不得绕过全表保护。
- 日志不得输出完整请求体、Token、密码、clientSecret、证件号、手机号、邮箱、连接串等敏感数据。
- 敏感数据加密与脱敏遵循框架已有 TypeHandler 和脱敏模式。

## 依赖与禁止事项

- 通过 `fons4cloud-starter` 或 `fons4cloud-common-*` 按需引入能力；鉴权、网关和 MQ 复用对应模块。
- 禁止新增与 `R`、`Result`、框架异常、`CommonConverter` 平行的体系。
- 禁止绕过框架鉴权、锁、限流、消息和文件能力。
- 禁止在 Application 重复声明式校验已覆盖的简单判断。
- 禁止让 Infrastructure 决定业务状态、结论或跨领域顺序。
- 禁止 Aggregate 包含其他领域可变实体或直接调用外部能力。
- 禁止将包含业务规则或外部访问的 Service 伪装成 `common.util`。
- 禁止引入冲突的重量级依赖，禁止提交真实密钥、Token、账号、连接串和敏感数据。

## 下游使用说明

- 下游代码规范可以补充更严格规则，但不能放宽本文件的框架复用和安全约束。
- Fons4AI SDD、变更和实现流程应加载本规则作为设计与实现证据。
- 历史代码按需渐进对齐，不为形式统一发起无业务收益的大范围重构。
