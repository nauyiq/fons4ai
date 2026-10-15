package com.fons.cloud.ai.agent.infrastructure.middleware;

import com.fons.cloud.ai.agent.infrastructure.utils.AgentScopeDelegationResult;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeApprovalState;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeApprovalState.Branch;
import com.fons.cloud.ai.agent.model.runtime.AgentScopeApprovalState.Delegation;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.ReActAgent;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.ToolCallState;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolResultState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.middleware.AgentInput;
import io.agentscope.core.middleware.ActingInput;
import io.agentscope.core.middleware.MiddlewareBase;
import io.agentscope.core.permission.PermissionContextState;
import io.agentscope.core.permission.PermissionDecision;
import io.agentscope.core.permission.PermissionRule;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.ToolContextState;
import io.agentscope.core.state.ToolContextState.SpawnEntry;
import io.agentscope.core.tool.ToolBase;
import io.agentscope.core.tool.ToolCallParam;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.harness.agent.HarnessAgent;
import io.agentscope.harness.agent.middleware.DynamicSubagentsMiddleware;
import io.agentscope.harness.agent.subagent.DefaultAgentManager;
import io.agentscope.harness.agent.tool.AgentSpawnTool;
import org.apache.commons.lang3.StringUtils;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * 将本地子Agent的审批等待转换为父委派工具的原生挂起结果。
 *
 * <p>只在真实工具返回点识别子状态，不修改只用于观察的过程事件。
 * SDK先执行完当前工具批次，再以TOOL_SUSPENDED结束父阶段，已派发的兄弟任务继续执行。
 * 阶段完整结束后保存可信父调用与子会话关联，正式HITL由框架在安全边界统一发布。</p>
 *
 * <p>嵌套委派时，每个原生父Agent安装自己的中间件并保存自己的Branch。
 * 恢复器沿Branch恢复子会话，再一次性回填原始父工具调用；普通消息不能覆盖挂起分支。</p>
 *
 * @author hongqy
 */
public final class AgentScopeSubagentHitlMiddleware implements MiddlewareBase {

    /**
     * SDK原生会话状态键；审批依据来自持久化状态，不依赖事件source寻址。
     */
    private static final String NATIVE_STATE_KEY = "agent_state";

    /**
     * 只包装同步委派工具，不接管业务工具或后台任务管理工具。
     */
    private static final List<String> DELEGATION_TOOLS = List.of("agent_spawn", "agent_send");

    /**
     * 包装器安装使用稳定的私有锁；只保护工具查找与注册，不覆盖任何子Agent执行。
     * 不同中间件实例也可能接触同一Toolkit，因此安装过程共用这一把锁。
     */
    private static final Lock TOOLKIT_ADAPTATION_LOCK = new ReentrantLock();

    /**
     * 不绑定Agent实例的共享适配器；运行状态只放入每次调用自己的RuntimeContext。
     */
    private static final AgentScopeSubagentHitlMiddleware INSTANCE = new AgentScopeSubagentHitlMiddleware();

    /**
     * 对应的SDK Agent提供者；为null时适配当前Agent，便于子Builder直接注册。
     */
    private final Supplier<? extends Agent> owner;

    /**
     * 创建不绑定实例的适配器，适用于SDK子Agent Builder。
     */
    public AgentScopeSubagentHitlMiddleware() {
        this.owner = null;
    }

    /**
     * 创建仅接管指定delegate的适配器，适用于框架Harness的顶层生命周期。
     */
    public AgentScopeSubagentHitlMiddleware(Supplier<? extends Agent> owner) {
        this.owner = Objects.requireNonNull(owner, "owner");
    }

    /**
     * 取得无共享运行状态的适配器，可复用到多个本地子Agent。
     */
    public static AgentScopeSubagentHitlMiddleware getInstance() {
        return INSTANCE;
    }

    @Override
    public Flux<AgentEvent> onAgent(Agent agent, RuntimeContext context, AgentInput input,
                                    Function<AgentInput, Flux<AgentEvent>> next) {
        if (owner != null && agent != nativeAgent(owner.get())) {
            return next.apply(input);
        }
        return Flux.defer(() -> {
            ReActAgent nativeOwner = nativeAgent(agent);
            validateRegistryPatch();
            AgentStateStore store = requireStateStore(nativeOwner);
            validateResumeInput(store, context, input);
            // Harness已经复制Toolkit，此时安装到真正参与原生执行的工具实例上。
            adaptToolkit(nativeOwner.getToolkit());
            context.put(AgentSpawnTool.CTX_FORCE_SYNC, true);
            DelegationRun run = new DelegationRun();
            context.put(DelegationRun.class, run);
            // 等原生父状态保存且整批工具结束后再保存Branch。持久化失败不能发布可恢复HITL。
            return next.apply(input).concatWith(Mono.fromRunnable(() -> {
                        Branch branch = new Branch();
                        branch.setDelegations(run.snapshot());
                        store.save(context.getUserId(), context.getSessionId(),
                                AgentScopeApprovalState.BRANCH_KEY, branch);
                    }).subscribeOn(Schedulers.boundedElastic()).thenMany(Flux.<AgentEvent>empty()))
                    .doFinally(signal -> context.put(DelegationRun.class, null));
        });
    }

    /**
     * 派发前按模型给出的顺序识别同目标调用；独立会话并行，有依赖的后续调用明确拒绝。
     * CHM只保护登记，不允许两个任务同时改写同一个子会话。
     */
    @Override
    public Flux<AgentEvent> onActing(Agent agent, RuntimeContext context, ActingInput input,
                                     Function<ActingInput, Flux<AgentEvent>> next) {
        DelegationRun run = context.get(DelegationRun.class);
        if (run != null && (owner == null || agent == nativeAgent(owner.get()))) {
            run.blockedCalls = Set.of();
            if (input.toolCalls().stream().noneMatch(call -> call.getName() != null
                    && DELEGATION_TOOLS.contains(call.getName()))) {
                // 叶子节点可以关闭subagents；普通业务工具不需要委派管理器。
                return next.apply(input);
            }
            DefaultAgentManager manager = nativeManager(agent, context);
            AgentState state = RuntimeContext.resolveAgentState(context, agent);
            if (state == null) {
                throw SystemIntervalException.of("Delegation batch has no native parent state");
            }
            Map<String, String> targets = new HashMap<>();
            Set<String> blocked = new HashSet<>();
            Map<String, SpawnEntry> registry = state.getToolContext().getSpawnRegistry();
            for (ToolUseBlock call : input.toolCalls()) {
                String target = delegationTarget(call, manager, registry);
                if (target != null && targets.putIfAbsent(target, call.getId()) != null) {
                    blocked.add(call.getId());
                }
            }
            // 一次acting的工具全集先确定，再并行订阅；工具只读取不可变的冲突集合。
            run.blockedCalls = Set.copyOf(blocked);
        }
        return next.apply(input);
    }

    /**
     * 静态子Agent模式优先使用SDK写入当前请求的管理器快照。
     * 动态模式不写入该上下文属性，工具直接使用中间件持有的管理器；此处遵循同一路径，
     * 不创建替代管理器，也不把缺少上下文属性误判为委派能力缺失。
     */
    private static DefaultAgentManager nativeManager(Agent agent, RuntimeContext context) {
        DefaultAgentManager contextual = context.get(AgentSpawnTool.CTX_AGENT_MANAGER);
        if (contextual != null) {
            return contextual;
        }
        for (MiddlewareBase middleware : nativeAgent(agent).getMiddlewares()) {
            if (middleware instanceof DynamicSubagentsMiddleware dynamic && dynamic.getAgentManager() != null) {
                return dynamic.getAgentManager();
            }
        }
        throw SystemIntervalException.of("Delegation has no native DefaultAgentManager");
    }

    /**
     * 将key/label别名和持久spawn归一到同一目标；无label的非持久spawn各建独立会话。
     */
    private String delegationTarget(ToolUseBlock call, DefaultAgentManager manager,
                                    Map<String, SpawnEntry> registry) {
        // 流式工具响应允许暂缺名称；不完整片段不参与路由，继续交由SDK原生链路处理。
        // List.of的contains不接受null，不能让观察性的路由检查先于工具校验抛出空指针。
        if (call.getName() == null || !DELEGATION_TOOLS.contains(call.getName())) {
            return null;
        }
        Map<String, Object> input = call.getInput();
        String label = input.get("label") instanceof String text ? StringUtils.trimToNull(text) : null;
        if (label != null) {
            return "label:" + label.toLowerCase(Locale.ROOT);
        }
        if ("agent_spawn".equals(call.getName())) {
            String agentId = input.get("agent_id") instanceof String text ? text : null;
            return agentId != null && manager.getDeclaration(agentId)
                    .map(declaration -> declaration.isPersistSession()).orElse(false) ? "agent:" + agentId : null;
        }
        String key = input.get("agent_key") instanceof String text ? StringUtils.trimToNull(text) : null;
        SpawnEntry entry = registry.get(key == null ? "" : key);
        if (entry == null) {
            return key == null ? null : "key:" + key;
        }
        if (StringUtils.isNotBlank(entry.label())) {
            return "label:" + entry.label().trim().toLowerCase(Locale.ROOT);
        }
        return manager.getDeclaration(entry.agentId()).map(declaration -> declaration.isPersistSession()).orElse(false)
                ? "agent:" + entry.agentId() : "session:" + entry.sessionId();
    }

    /**
     * 挂起分支只接受完整父工具结果，普通对话不能清空已有恢复路由。
     */
    private void validateResumeInput(AgentStateStore store, RuntimeContext context, AgentInput input) {
        Branch previous = store.get(context.getUserId(), context.getSessionId(),
                AgentScopeApprovalState.BRANCH_KEY, Branch.class).orElse(null);
        if (previous == null || previous.getDelegations().isEmpty()) {
            return;
        }
        Set<String> suppliedIds = input.msgs().stream()
                .filter(message -> message.getRole() == MsgRole.TOOL)
                .flatMap(message -> message.getContentBlocks(ToolResultBlock.class).stream())
                .map(ToolResultBlock::getId).collect(Collectors.toSet());
        if (!previous.getDelegations().stream()
                .allMatch(delegation -> suppliedIds.contains(delegation.toolCall().getId()))) {
            throw SystemIntervalException.of("Pending subagent approvals require complete delegated tool results");
        }
    }

    /**
     * 每个Toolkit只安装一次；保留原工具Schema、权限和并发属性。
     */
    private void adaptToolkit(Toolkit toolkit) {
        TOOLKIT_ADAPTATION_LOCK.lock();
        try {
            List<DelegationTool> wrappers = new ArrayList<>();
            for (String name : DELEGATION_TOOLS) {
                var tool = toolkit.getTool(name);
                if (tool == null || tool instanceof DelegationTool) {
                    continue;
                }
                if (!(tool instanceof ToolBase delegate)) {
                    throw SystemIntervalException.of("Subagent HITL requires native ToolBase: " + name);
                }
                wrappers.add(new DelegationTool(delegate));
            }
            wrappers.forEach(toolkit::registerAgentTool);
        } finally {
            TOOLKIT_ADAPTATION_LOCK.unlock();
        }
    }

    /**
     * 启动及恢复前确认兼容补丁实际被加载，防止依赖顺序让原SDK类抢先生效。
     * 不改变SDK公开API；两类均来自本starter时才允许进入受管执行。
     */
    public static void validateRegistryPatch() {
        var patchSource = ToolContextState.class.getProtectionDomain().getCodeSource();
        var frameworkSource = AgentScopeSubagentHitlMiddleware.class.getProtectionDomain().getCodeSource();
        if (patchSource == null || frameworkSource == null
                || !Objects.equals(patchSource.getLocation(), frameworkSource.getLocation())) {
            throw SystemIntervalException.of("本地子审批并行需要加载Fons的ToolContextState兼容补丁，请检查类路径顺序");
        }
    }

    /**
     * 支持SDK Harness与原生ReActAgent，其他执行协议不属于本地收口能力。
     */
    private static ReActAgent nativeAgent(Agent agent) {
        if (agent instanceof HarnessAgent harness) {
            return harness.getDelegate();
        }
        if (agent instanceof ReActAgent react) {
            return react;
        }
        throw SystemIntervalException.of("Subagent HITL requires local AgentScope ReActAgent");
    }

    private static AgentStateStore requireStateStore(ReActAgent agent) {
        if (agent.getStateStore() == null) {
            throw SystemIntervalException.of("Subagent HITL requires AgentStateStore");
        }
        return agent.getStateStore();
    }

    /**
     * 单次原生调用的委派关联，不持有Agent或异步执行权柄。
     */
    private static final class DelegationRun {

        /**
         * 按原父工具调用ID原子去重，各子结果返回时不争用整批互斥锁。
         */
        private final ConcurrentMap<String, Delegation> delegations = new ConcurrentHashMap<>();

        /**
         * 本次acting中依赖同一目标的后续调用；并行执行开始前一次发布。
         */
        private volatile Set<String> blockedCalls = Set.of();

        private void record(Delegation delegation) {
            Delegation existing = delegations.putIfAbsent(delegation.toolCall().getId(), delegation);
            if (existing != null && !existing.equals(delegation)) {
                throw SystemIntervalException.of("Delegated tool call has conflicting child routing");
            }
        }

        private List<Delegation> snapshot() {
            // 只在整批工具结束后读取，排序使持久化顺序不受线程完成先后影响。
            return delegations.values().stream()
                    .sorted(Comparator.comparing(delegation -> delegation.toolCall().getId()))
                    .toList();
        }
    }

    /**
     * 在真实callAsync返回值上挂起原工具，避免事件状态与SDK保存状态不一致。
     */
    private static final class DelegationTool extends ToolBase {

        /**
         * 原执行器，保留SDK事件发送、取消传播和参数注入链路。
         */
        private final ToolBase delegate;

        private DelegationTool(ToolBase delegate) {
            // 会话登记表由最小SDK兼容补丁保证并发安全，包装器保留原工具的并发声明。
            super(delegate.getName(), delegate.getDescription(), delegate.getParameters(), delegate.isReadOnly(),
                    delegate.isConcurrencySafe(), delegate.isMcp(), delegate.getMcpName(),
                    delegate.isExternalTool(), delegate.isStateInjected());
            this.delegate = delegate;
        }

        @Override
        public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
            DelegationRun run = param.getRuntimeContext().get(DelegationRun.class);
            if (run != null && run.blockedCalls.contains(param.getToolUseBlock().getId())) {
                return Mono.just(ToolResultBlock.error(
                                "Delegations to the same child target must be submitted in separate tool batches")
                        .withIdAndName(param.getToolUseBlock().getId(), param.getToolUseBlock().getName()));
            }
            return Mono.fromCallable(() -> findWaitingBefore(param))
                    .subscribeOn(Schedulers.boundedElastic())
                    .flatMap(waiting -> {
                        if (waiting.isPresent()) {
                            // 当前新任务尚未执行，不能把旧子任务的审批恢复结果当作本次委派结果。
                            // 原来已执行并挂起的父调用继续等待，此调用只保存明确拒绝结果，不新增恢复路由。
                            return Mono.just(ToolResultBlock.error(
                                            "Child session is waiting for approval; ordinary delegation cannot replace its pending task")
                                    .withIdAndName(param.getToolUseBlock().getId(), param.getToolUseBlock().getName()));
                        }
                        // 原生同步调用已将子状态保存。只解析SDK传输头，子业务文本不参与寻址。
                        return delegate.callAsync(param).flatMap(result -> Mono.fromCallable(() -> {
                            // SDK在工具Mono完成后才补SUCCESS。原始结果默认RUNNING，此处只归一化解析视图，
                            // 原执行结果仍由ToolExecutor决定终态；已失败或suspended结果不能据此伪装成功。
                            ToolResultBlock completedView = result.getState() == ToolResultState.RUNNING && !result.isSuspended()
                                    ? result.withState(ToolResultState.SUCCESS) : result;
                            AgentScopeDelegationResult transport = AgentScopeDelegationResult
                                    .parse(param.getToolUseBlock(), completedView);
                            if (transport == null) {
                                return result;
                            }
                            SpawnEntry entry = parentState(param).getToolContext().getSpawnRegistry()
                                    .get(transport.agentKey());
                            if (entry == null || !entry.agentId().equals(transport.agentId())) {
                                throw SystemIntervalException.of("Delegation result has no matching native spawn entry");
                            }
                            return isWaiting(param, entry) ? suspend(param, entry) : result;
                        }).subscribeOn(Schedulers.boundedElastic()));
                    });
        }

        /**
         * 对已登记会话先检查审批，防止send或持久会话spawn用普通消息覆盖原确认。
         */
        private Optional<SpawnEntry> findWaitingBefore(ToolCallParam param) {
            AgentState parent = parentState(param);
            Map<String, Object> input = param.getInput();
            DefaultAgentManager manager = nativeManager(param.getAgent(), param.getRuntimeContext());
            Object requestedAgent = input.get("agent_id");
            if ("agent_spawn".equals(getName()) && requestedAgent instanceof String agentId) {
                var declaration = manager.getDeclaration(agentId);
                if (declaration.isPresent() && declaration.get().isRemote()) {
                    throw SystemIntervalException.of("Subagent HITL supports local delegation only");
                }
                // 非持久spawn创建新会话，不按同类型子Agent阻断正常兄弟任务。
                if (declaration.isEmpty() || !declaration.get().isPersistSession()) {
                    return Optional.empty();
                }
            }
            List<SpawnEntry> matches = parent.getToolContext().getSpawnRegistry().values().stream()
                    .filter(entry -> matchesExisting(entry, input)).toList();
            if (matches.size() > 1) {
                throw SystemIntervalException.of("Delegation target resolves to multiple native sessions");
            }
            if (matches.isEmpty()) {
                return Optional.empty();
            }
            SpawnEntry entry = matches.getFirst();
            return isWaiting(param, entry) ? Optional.of(entry) : Optional.empty();
        }

        /**
         * send遵守SDK的key/label互斥规则；持久spawn按类型与label定位原会话。
         */
        private boolean matchesExisting(SpawnEntry entry, Map<String, Object> input) {
            if ("agent_spawn".equals(getName())) {
                String label = input.get("label") instanceof String text && !text.isBlank() ? text.trim() : null;
                return entry.agentId().equals(input.get("agent_id")) && Objects.equals(entry.label(), label);
            }
            String key = input.get("agent_key") instanceof String text ? text.trim() : null;
            String label = input.get("label") instanceof String text ? text.trim() : null;
            if (StringUtils.isNotBlank(key) && StringUtils.isNotBlank(label)) {
                return false;
            }
            return StringUtils.isNotBlank(key) ? entry.key().equals(key)
                    : StringUtils.isNotBlank(label) && entry.label() != null && entry.label().equalsIgnoreCase(label);
        }

        /**
         * 直接读取共享存储，既识别本节点ASKING，也识别其已收口的嵌套委派。
         */
        private boolean isWaiting(ToolCallParam param, SpawnEntry entry) {
            RuntimeContext context = param.getRuntimeContext();
            // 父子节点须共用Store；检查状态不应调用factory重新构建完整子Agent实例。
            AgentStateStore store = requireStateStore(nativeAgent(param.getAgent()));
            AgentState state = store.get(context.getUserId(), entry.sessionId(), NATIVE_STATE_KEY, AgentState.class)
                    .orElseThrow(() -> SystemIntervalException.of(
                            "Native child state is unavailable in the shared parent-child store: " + entry.sessionId()));
            if (hasPendingApproval(state)) {
                return true;
            }
            return store.get(context.getUserId(), entry.sessionId(), AgentScopeApprovalState.BRANCH_KEY, Branch.class)
                    .map(branch -> !branch.getDelegations().isEmpty()).orElse(false);
        }

        /**
         * 仅当前assistant尚无结果的ASKING有效，历史审批不能重复挂起。
         */
        private boolean hasPendingApproval(AgentState state) {
            List<Msg> messages = state.getContext();
            Set<String> resultIds = messages.stream()
                    .flatMap(message -> message.getContentBlocks(ToolResultBlock.class).stream())
                    .filter(result -> !result.isSuspended())
                    .map(ToolResultBlock::getId).collect(Collectors.toSet());
            for (int index = messages.size() - 1; index >= 0; index--) {
                Msg message = messages.get(index);
                if (message.getRole() == MsgRole.ASSISTANT) {
                    return message.getContentBlocks(ToolUseBlock.class).stream()
                            .anyMatch(tool -> tool.getState() == ToolCallState.ASKING && !resultIds.contains(tool.getId()));
                }
            }
            return false;
        }

        /**
         * 记录原调用后返回原生suspended；SDK仍会收集本批其余工具结果。
         */
        private ToolResultBlock suspend(ToolCallParam param, SpawnEntry entry) {
            RuntimeContext context = param.getRuntimeContext();
            DelegationRun run = context.get(DelegationRun.class);
            if (run == null || param.getToolUseBlock() == null) {
                throw SystemIntervalException.of("Delegation suspension has no active native run or original tool call");
            }
            run.record(new Delegation(context.getSessionId(), param.getToolUseBlock(),
                    entry.key(), entry.agentId(), entry.sessionId()));
            return ToolResultBlock.suspended(param.getToolUseBlock());
        }

        private AgentState parentState(ToolCallParam param) {
            AgentState state = RuntimeContext.resolveAgentState(param.getRuntimeContext(), param.getAgent());
            if (param.getRuntimeContext() == null || state == null || param.getToolUseBlock() == null) {
                throw SystemIntervalException.of("Delegation has no native parent context or tool call");
            }
            return state;
        }

        @Override
        public Mono<PermissionDecision> checkPermissions(Map<String, Object> input, PermissionContextState context) {
            return delegate.checkPermissions(input, context);
        }

        @Override
        public boolean matchRule(String ruleContent, Map<String, Object> input) {
            return delegate.matchRule(ruleContent, input);
        }

        @Override
        public List<PermissionRule> generateSuggestions(Map<String, Object> input) {
            return delegate.generateSuggestions(input);
        }

        @Override
        public Boolean getStrict() {
            return delegate.getStrict();
        }

        @Override
        public Map<String, Object> getOutputSchema() {
            return delegate.getOutputSchema();
        }
    }
}
