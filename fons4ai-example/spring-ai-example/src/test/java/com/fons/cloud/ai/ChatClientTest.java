package com.fons.cloud.ai;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import jakarta.annotation.PostConstruct;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;

/**
 * 初识ChatClient，相当于chatModel的门面客户端， 并且拥有可定制化，可扩展性的能力与其他功能
 * @author hongqy
 * @date 2026/2/28
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class ChatClientTest {

    @Autowired
    private DashScopeChatModel chatModel;

    private ChatClient chatClient;

    @PostConstruct
    public void init() {
        this.chatClient = ChatClient.builder(chatModel)
                .defaultAdvisors(new SimpleLoggerAdvisor())
                // 默认系统提示词
                .defaultSystem("请用英文回答问题")
                // 默认操作
                .defaultOptions(DashScopeChatOptions.builder()
                        .model("deepseek-v3")
                        .topP(0.7)
                        .temperature(0.7)
                        .build())
                .build();

    }


    @Test
    public void testPrompt() {
        String userPromptContent = "你是谁？";

        System.out.println("简单调用:");
        ChatClient.ChatClientRequestSpec spec = this.chatClient.prompt(userPromptContent);
        System.out.println(spec.call().content());

        System.out.println();

        System.out.println("系统提示词覆盖调用:");
        System.out.println(this.chatClient.prompt(userPromptContent).system("请用中文回答问题").call().content());

        System.out.println();

        System.out.println("系统提示词追加调用:");
        Prompt prompt = Prompt.builder()
                .messages(new SystemMessage("请再用日文再回答一次"), new UserMessage(userPromptContent))
                .build();
        System.out.println(this.chatClient.prompt(prompt).call().content());
    }






}
