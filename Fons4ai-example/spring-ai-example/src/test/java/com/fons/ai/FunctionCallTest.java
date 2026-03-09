package com.fons.ai;

import com.fons.ai.function.TimeTools;
import jakarta.annotation.PostConstruct;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;

/**
 * @author hongqy
 * @date 2026/3/5
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class FunctionCallTest {

    @Autowired
    private OpenAiChatModel openAiChatModel;

    private ChatClient chatClient;

    @PostConstruct
    public void init() {
        ChatMemory chatMemory = MessageWindowChatMemory.builder().maxMessages(10).build();
        this.chatClient = ChatClient.builder(openAiChatModel)
                .defaultAdvisors(MessageChatMemoryAdvisor.builder(chatMemory).build())
                .build();
    }

    @Test
    public void testFunctionCall1() throws InterruptedException {
        // 通过bean name获取工具名
        String content = chatClient.prompt().toolNames("getTimeFunction")
                .user("广东现在几点了?")
                .call().content();
        System.out.println(content);
    }

    @Test
    public void testFunctionCall2() {
        // 指定工具调用
        String content = chatClient.prompt().tools(new TimeTools())
                .user("广东现在几点了?")
                .call().content();
        System.out.println(content);
    }
}
