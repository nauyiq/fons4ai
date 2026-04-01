package com.fons.cloud.ai.agent.standard;

import com.fons.cloud.ai.agent.advisor.ReflectionAdvisor;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 基于ReAct的基础上， 进行自我思考，批判
 * @author hongqy
 * @date 2026/3/20
 */
public class ReflectionAgent {

    private final SimpleReActAgent delegate;

    private ReflectionAgent(SimpleReActAgent delegate) {
        this.delegate = delegate;
    }

    public String call(String question) {
        return delegate.call(question);
    }

    public String call(String conversationId, String question) {
        return delegate.call(conversationId, question);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private String name = "reflection-react-agent";
        private ChatModel chatModel;
        private List<ToolCallback> tools = new ArrayList<>();
        private int maxRounds;
        private String systemPrompt = "";
        private List<Advisor> advisors = new ArrayList<>();
        private int maxReflectionRounds = 1;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder chatModel(ChatModel chatModel) {
            this.chatModel = chatModel;
            return this;
        }

        public Builder tools(ToolCallback... tools) {
            this.tools = Arrays.asList(tools);
            return this;
        }

        public Builder tools(List<ToolCallback> tools) {
            this.tools = tools;
            return this;
        }

        public Builder advisors(Advisor... advisors) {
            this.advisors.addAll(Arrays.asList(advisors));
            return this;
        }

        public Builder systemPrompt(String systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder maxReflectionRounds(int maxReflectionRounds) {
            this.maxReflectionRounds = maxReflectionRounds;
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            this.maxRounds = maxRounds;
            return this;
        }

        public ReflectionAgent build() {

            if (chatModel == null) {
                throw new IllegalArgumentException("chatModel 不能为空");
            }

            ReflectionAdvisor reflectionAdvisor = new ReflectionAdvisor(chatModel);

            List<Advisor> finalAdvisors = new ArrayList<>(advisors);
            finalAdvisors.add(reflectionAdvisor);

            ChatMemory chatMemory = MessageWindowChatMemory.builder().maxMessages(20).build();

            SimpleReActAgent reactAgent = SimpleReActAgent.builder(chatModel, tools)
                    .maxRounds(maxRounds)
                    .systemPrompt(systemPrompt)
                    .chatMemory(chatMemory)
                    .maxReflectionRounds(maxReflectionRounds)
                    .advisors(finalAdvisors)
                    .build();

            return new ReflectionAgent(reactAgent);
        }
    }


}
