package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.infrastructure.session.ActiveAgentSessionStore;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.extensions.jdbc.JdbcDistributedStore;
import io.agentscope.extensions.jdbc.dialect.AbstractJdbcDialect;
import io.agentscope.extensions.jdbc.state.JdbcAgentStateStore;
import io.agentscope.harness.agent.DistributedStore;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;

/**
 * @author hongqy
 */
@Configuration
public class AgentScopeStateStoreAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnBean(DataSource.class)
    public AgentStateStore agentStateStore(DataSource dataSource) {
        AbstractJdbcDialect dialect = AbstractJdbcDialect.from(dataSource).build();
        // 不自动建表
        return new JdbcAgentStateStore(dataSource, dialect, false);
    }

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnBean(DataSource.class)
    public DistributedStore distributedStore(DataSource dataSource) {
        return JdbcDistributedStore.create(dataSource);
    }


    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnBean(DataSource.class)
    public ActiveAgentSessionStore activeAgentSessionStore(AgentStateStore agentStateStore) {
        return new ActiveAgentSessionStore(agentStateStore);
    }

}
