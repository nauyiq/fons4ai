package com.fons.cloud.ai.agent.observability.autoconfigure.instrumentation;

import com.fons.cloud.ai.agent.observability.annotation.AgentObserved;
import lombok.extern.slf4j.Slf4j;
import org.springframework.aop.Advisor;
import org.springframework.aop.Pointcut;
import org.springframework.aop.framework.autoproxy.AbstractAutoProxyCreator;
import org.springframework.aop.support.DefaultPointcutAdvisor;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.AnnotatedElementUtils;
import org.springframework.util.ClassUtils;
import org.springframework.util.ReflectionUtils;

import java.lang.reflect.Method;

/**
 * {@link AgentObserved} 目标 Bean 的自动代理装配。
 *
 * <p>创建器只负责两件事：筛选带观测声明的 Bean，并为其挂载唯一的观测 Advisor；
 * 注解解析、Convention 选择与采集语义全部由 {@link AgentObservationInterceptor} 执行。
 * 未声明的 Bean 不被代理，全应用代理策略不受影响。</p>
 *
 * <p>代理类型策略：接口可代理时优先 JDK 动态代理，保证目标类的 final 模板方法
 * （如通过接口调用的 {@code handle}）可被拦截；无接口或接口不满足时回退 CGLIB。
 * 本创建器独立于应用级 {@code spring.aop.proxy-target-class} 配置，不修改它。</p>
 *
 * <p>拦截器通过延迟提供器在首次实际装配时取得，避免代理创建器注册阶段
 * 提前实例化观测基础设施和业务 Bean。</p>
 *
 * @author hongqy
 */
@Slf4j
public class AgentObservationAutoProxyCreator extends AbstractAutoProxyCreator {

    /**
     * 观测拦截器延迟提供器。
     */
    private final ObjectProvider<AgentObservationInterceptor> interceptorProvider;

    /**
     * 唯一观测 Advisor 的懒加载缓存。
     */
    private volatile Advisor observationAdvisor;

    /**
     * 创建自动代理装配器。
     *
     * @param interceptorProvider 观测拦截器延迟提供器
     */
    public AgentObservationAutoProxyCreator(ObjectProvider<AgentObservationInterceptor> interceptorProvider) {
        this.interceptorProvider = interceptorProvider;
        // 不修改全应用代理策略；本创建器独立选择 JDK 优先，保证 final 方法可拦截。
        setProxyTargetClass(false);
        setOrder(Ordered.LOWEST_PRECEDENCE);
    }

    @Override
    protected Object[] getAdvicesAndAdvisorsForBean(Class<?> beanClass, String beanName,
                                                    org.springframework.aop.TargetSource customTargetSource) {
        // 先筛选注解目标再解析拦截器，避免基础设施 Bean 初始化期触发观测依赖链。
        if (!isAnnotatedBean(beanClass)) {
            return DO_NOT_PROXY;
        }
        Advisor advisor = observationAdvisor();
        return advisor == null ? DO_NOT_PROXY : new Object[]{advisor};
    }

    /**
     * 获取唯一的观测 Advisor；拦截器不可用时记录诊断并放弃代理。
     */
    private Advisor observationAdvisor() {
        Advisor current = observationAdvisor;
        if (current == null) {
            AgentObservationInterceptor interceptor = interceptorProvider.getIfAvailable();
            if (interceptor == null) {
                log.warn("AgentObservationInterceptor unavailable, annotated beans will not be proxied");
                return null;
            }
            // 代理目标 Bean 已按注解筛选；方法级声明与语义判断由拦截器执行。
            current = new DefaultPointcutAdvisor(Pointcut.TRUE, interceptor);
            observationAdvisor = current;
        }
        return current;
    }

    /**
     * 判断 Bean 是否携带观测声明：类型注解（含父类层级）或任一声明方法注解。
     *
     * <p>CGLIB 代理类先还原为用户类再判断。方法注解不继承，逐方法直接检查；
     * Object 继承的方法没有声明，直接跳过。</p>
     */
    private boolean isAnnotatedBean(Class<?> beanClass) {
        Class<?> userClass = ClassUtils.getUserClass(beanClass);
        if (AnnotatedElementUtils.findMergedAnnotation(userClass, AgentObserved.class) != null) {
            return true;
        }
        for (Method method : ReflectionUtils.getAllDeclaredMethods(userClass)) {
            if (method.isAnnotationPresent(AgentObserved.class)) {
                return true;
            }
        }
        return false;
    }
}
