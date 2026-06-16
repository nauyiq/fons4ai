package com.fons.cloud.ai.agent.standard.websearch;

import com.fons.cloud.ai.agent.common.constants.AgentType;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.prompt.AgentSystemPrompt;
import com.fons.cloud.ai.agent.prompt.WebSearchReactAgentSystemPromptBuilder;
import com.fons.cloud.ai.agent.standard.react.ReactAgent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Sinks;

import java.util.List;

/**
 * 基于网络搜索的ReactAgent
 * @author hongqy
 */
@Slf4j
public class WebSearchReactAgent extends ReactAgent {

    protected WebSearchReactAgent(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
        super(AgentType.WEB_SEARCH, tools, chatModel, agentTaskManager);
    }

    /**
     * 新增工具输出, websearch工具调用之前可以输出一些提示语
     * @param sink
     * @param toolCall
     * @param context
     */
    @Override
    protected void beforeToolCall(Sinks.Many<String> sink, AssistantMessage.ToolCall toolCall, ReactExecutionContext context) {
//        if (toolCall.name())
    }




    public static class Builder {
        private final List<ToolCallback> tools;
        private final ChatModel chatModel;
        private final AgentTaskManager agentTaskManager;

        private List<Advisor> advisors;
        private AgentSystemPrompt systemPrompt;
        private int maxRounds = 5;
        private boolean useChatMemory;
        private int maxMemoryMessages;

        public Builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager) {
            this.tools = tools;
            this.chatModel = chatModel;
            this.agentTaskManager = agentTaskManager;
        }

        public Builder advisors(List<Advisor> advisors) {
            this.advisors = advisors;
            return this;
        }

        public Builder systemPrompt(AgentSystemPrompt systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            this.maxRounds = maxRounds;
            return this;
        }

        public Builder useChatMemory(boolean useChatMemory) {
            this.useChatMemory = useChatMemory;
            return this;
        }

        public Builder maxMemoryMessages(int maxMemoryMessages) {
            this.maxMemoryMessages = maxMemoryMessages;
            return this;
        }

        public WebSearchReactAgent build() {
            WebSearchReactAgent reactAgent = new WebSearchReactAgent(tools, chatModel, agentTaskManager);
            if (this.systemPrompt == null) {
                // 这里赋予新的系统提示语， 把角色定义为专门用于网络搜索的agent
                this.systemPrompt = WebSearchReactAgentSystemPromptBuilder.build();
            }
            reactAgent.systemPrompt = this.systemPrompt;
            reactAgent.advisors = this.advisors;
            reactAgent.maxRounds = this.maxRounds;
            reactAgent.maxMemoryMessages = this.maxMemoryMessages;
            // 初始化
            reactAgent.init(this.useChatMemory);
            return reactAgent;
        }
    }

}
