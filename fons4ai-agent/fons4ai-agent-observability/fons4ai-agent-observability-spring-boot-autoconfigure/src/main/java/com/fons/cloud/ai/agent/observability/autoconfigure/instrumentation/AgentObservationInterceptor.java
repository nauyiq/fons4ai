package com.fons.cloud.ai.agent.observability.autoconfigure.instrumentation;

import com.fons.cloud.ai.agent.observability.annotation.AgentObserved;
import com.fons.cloud.ai.agent.observability.api.ObservationConvention;
import lombok.extern.slf4j.Slf4j;
import org.aopalliance.intercept.MethodInterceptor;
import org.aopalliance.intercept.MethodInvocation;
import org.springframework.aop.support.AopUtils;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.core.annotation.AnnotatedElementUtils;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.lang.reflect.Method;
import java.util.List;

/**
 * {@link AgentObserved} 声明的统一拦截入口。
 *
 * <p>拦截器处理 RUN 与 STEP 两种语义：RUN 在方法执行期维护创建期绑定栈，把采集描述
 * 交给工厂创建的 Run；STEP 在每次真实订阅时判定是否采集。业务方法恰好调用一次，
 * 同步异常原样抛出，采集装配失败只关闭该入口的观测。</p>
 *
 * <p>注解解析使用真实目标类和最具体方法，方法注解优先于类型注解；类型声明只对可拦截
 * 的响应式方法生效，同步方法与 Object 方法不采集。Convention 通过延迟查找获得，
 * 不在代理装配期实例化业务 Bean。</p>
 *
 * @author hongqy
 */
@Slf4j
public class AgentObservationInterceptor implements MethodInterceptor {

    /**
     * 通用观测生命周期支持。
     */
    private final ReactiveObservationSupport support;

    /**
     * 业务侧 Convention 提供器；首次实际调用前不触发业务 Bean 实例化。
     */
    private final ObjectProvider<ObservationConvention> conventionProvider;

    /**
     * Convention 列表的单例快照；单例约定集合在容器启动后不再变化。
     */
    private volatile List<ObservationConvention> cachedConventions;

    /**
     * 创建统一拦截入口。
     *
     * @param support 通用观测生命周期支持
     * @param conventionProvider Convention 延迟提供器
     */
    public AgentObservationInterceptor(ReactiveObservationSupport support,
                                       ObjectProvider<ObservationConvention> conventionProvider) {
        this.support = support;
        this.conventionProvider = conventionProvider;
    }

    @Override
    public Object invoke(MethodInvocation invocation) throws Throwable {
        Object target = invocation.getThis();
        Class<?> targetClass = target == null
                ? invocation.getMethod().getDeclaringClass()
                : AopUtils.getTargetClass(target);
        Method specificMethod = AopUtils.getMostSpecificMethod(invocation.getMethod(), targetClass);
        AgentObserved declaration = findDeclaration(specificMethod, targetClass);
        if (declaration == null) {
            // 代理已按注解筛选目标 Bean；此处仅防御非声明方法的意外进入。
            return invocation.proceed();
        }
        ObservationConvention.Invocation observed = new ObservationConvention.Invocation(
                target, specificMethod, invocation.getArguments(), declaration);
        ObservationConvention convention = resolveConvention(observed);
        if (convention == null) {
            return invocation.proceed();
        }
        return switch (declaration.kind()) {
            case RUN -> invokeRun(invocation, observed, convention);
            case STEP -> invokeStep(invocation, observed, convention);
        };
    }

    /**
     * RUN 语义：创建期绑定一次，业务方法恰好执行一次。
     *
     * <p>进入方法前压入创建描述，返回或抛错的 finally 中弹出并冻结有效性；
     * 真正的根观测由工厂捕获的专属实例在首次订阅时建立。</p>
     */
    private Object invokeRun(MethodInvocation invocation, ObservationConvention.Invocation observed,
                             ObservationConvention convention) throws Throwable {
        ReactiveObservationSupport.RunCreation creation = support.beginCreation(observed, convention);
        Object result = null;
        try {
            result = invocation.proceed();
            return result;
        } finally {
            // 方法抛错时 result 为 null，绑定描述按无效处理并诊断。
            support.endCreation(creation, result);
        }
    }

    /**
     * STEP 语义：每次真实订阅时判定是否采集。
     *
     * <p>proceed 只调用一次；返回的 Publisher 以 defer 包装，订阅时读取
     * shouldObserveStep，false 只关闭本次采集。非响应式返回值不采集。</p>
     */
    private Object invokeStep(MethodInvocation invocation, ObservationConvention.Invocation observed,
                              ObservationConvention convention) throws Throwable {
        Object result = invocation.proceed();
        if (result instanceof Mono<?> mono) {
            return Mono.defer(() -> observeStepMono(observed, convention, mono));
        }
        if (result instanceof Flux<?> flux) {
            return Flux.defer(() -> observeStepFlux(observed, convention, flux));
        }
        // 类型声明遇到同步方法：不采集，原值返回。
        return result;
    }

    @SuppressWarnings("unchecked")
    private Mono<?> observeStepMono(ObservationConvention.Invocation observed,
                                    ObservationConvention convention, Mono<?> mono) {
        if (!shouldObserve(observed, convention)) {
            return mono;
        }
        try {
            return support.observeStep(convention.stepName(observed), (Mono<Object>) mono,
                    value -> convention.stepOutput(observed, value));
        } catch (Throwable error) {
            log.warn("AgentObserved STEP observation preparation failed, observation skipped, method:{}",
                    observed.method().getName(), error);
            return mono;
        }
    }

    @SuppressWarnings("unchecked")
    private Flux<?> observeStepFlux(ObservationConvention.Invocation observed,
                                    ObservationConvention convention, Flux<?> flux) {
        if (!shouldObserve(observed, convention)) {
            return flux;
        }
        try {
            return support.observeStep(convention.stepName(observed), (Flux<Object>) flux,
                    () -> convention.stepOutput(observed, null));
        } catch (Throwable error) {
            log.warn("AgentObserved STEP observation preparation failed, observation skipped, method:{}",
                    observed.method().getName(), error);
            return flux;
        }
    }

    /**
     * 订阅时判定本次是否采集；判定异常按不采集处理，不影响业务执行。
     */
    private boolean shouldObserve(ObservationConvention.Invocation observed, ObservationConvention convention) {
        try {
            return convention.shouldObserveStep(observed);
        } catch (Throwable error) {
            log.warn("AgentObserved STEP shouldObserveStep failed, observation skipped, method:{}",
                    observed.method().getName(), error);
            return false;
        }
    }

    /**
     * 解析当前声明唯一匹配的 Convention。
     *
     * <p>无匹配或多个匹配且无法唯一确定时，记录诊断并关闭该入口采集；
     * 不得擅自使用首个 Bean。</p>
     */
    private ObservationConvention resolveConvention(ObservationConvention.Invocation observed) {
        ObservationConvention matched = null;
        for (ObservationConvention candidate : conventions()) {
            boolean supported;
            try {
                supported = candidate.supports(observed);
            } catch (Throwable error) {
                log.warn("ObservationConvention.supports failed, candidate:{}", candidate.getClass().getName(), error);
                continue;
            }
            if (supported) {
                if (matched != null) {
                    log.warn("AgentObserved declaration matched multiple conventions, observation disabled, "
                                    + "target:{}, method:{}, first:{}, second:{}",
                            observed.target().getClass().getName(), observed.method().getName(),
                            matched.getClass().getName(), candidate.getClass().getName());
                    return null;
                }
                matched = candidate;
            }
        }
        if (matched == null) {
            log.warn("AgentObserved declaration has no matching ObservationConvention, "
                            + "observation disabled, target:{}, method:{}",
                    observed.target().getClass().getName(), observed.method().getName());
        }
        return matched;
    }

    /**
     * 获取容器中全部 Convention 的快照。
     */
    private List<ObservationConvention> conventions() {
        List<ObservationConvention> current = cachedConventions;
        if (current == null) {
            current = conventionProvider.orderedStream().toList();
            cachedConventions = current;
        }
        return current;
    }

    /**
     * 解析观测声明：方法注解优先，其次目标类型注解（支持父类与 CGLIB 用户类）。
     */
    private AgentObserved findDeclaration(Method method, Class<?> targetClass) {
        AgentObserved declaration = method.getAnnotation(AgentObserved.class);
        if (declaration != null) {
            return declaration;
        }
        return AnnotatedElementUtils.findMergedAnnotation(targetClass, AgentObserved.class);
    }
}
