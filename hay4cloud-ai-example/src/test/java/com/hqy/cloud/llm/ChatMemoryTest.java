package com.hqy.cloud.llm;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import jakarta.annotation.PostConstruct;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import org.springframework.web.client.ResourceAccessException;

import java.util.ArrayList;
import java.util.List;

/**
 * 对话记忆
 * <pre>
 *     1. 短期记忆
 *     2. 长期记忆
 * </pre>
 * @author hongqy
 * @date 2026/3/2
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class ChatMemoryTest {

    @Autowired
    private ChatModel chatModel;

    private ChatClient chatClient;

    @PostConstruct
    public void init() {
        this.chatClient = ChatClient.builder(chatModel)

                .defaultAdvisors(new SimpleLoggerAdvisor())
                // 默认操作
                .defaultOptions(DashScopeChatOptions.builder()
                        .model("deepseek-v3")
                        .temperature(0.7)
                        .build())
                .defaultSystem("你是一个旅游专家")
                .build();
    }

    @Test
    public void testShortMemory() {
        try {
            System.out.println("开始测试短期记忆对话...");
            List<Message> messages = new ArrayList<>();
            Message message2 = new UserMessage("我想去新疆旅游, 有什么简单的方案吗?");
            messages.add(message2);

            System.out.println("发送第一轮对话请求...");
            AssistantMessage message3 = chatClient.prompt(Prompt.builder()
                    .messages(messages).build()).call().chatResponse().getResult().getOutput();
            messages.add(message3);
            System.out.println("第一轮回复: " + message3.getText());

            Message message4 = new UserMessage("如果我计划元旦期间去，并且旅游时间为7天，有什么更好的方案吗，请简单出一个攻略");
            messages.add(message4);

            System.out.println("发送第二轮对话请求...");
            String content = chatClient.prompt(Prompt.builder().messages(messages).build()).call().content();
            System.out.println("第二轮回复: " + content);
            
        } catch (ResourceAccessException e) {
            System.err.println("网络连接超时，请检查网络连接或稍后重试: " + e.getMessage());
            System.err.println("建议: 1. 检查网络连接 2. 稍后再试 3. 检查API密钥是否正确");
        } catch (Exception e) {
            System.err.println("请求失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Test
    public void chat() {

        List<Message> messages = new ArrayList<>();

        //第一轮对话
        messages.add(new SystemMessage("你是一个游戏设计师"));
        messages.add(new UserMessage("我想设计一个回合制游戏"));
        ChatResponse chatResponse = chatModel.call(new Prompt(messages));
        String content = chatResponse.getResult().getOutput().getText();
        System.out.println(content);
        System.out.println("======");

        messages.add(new AssistantMessage(content));

        //第二轮对话
        messages.add(new UserMessage("能帮我结合一些二次元的元素吗?"));
        chatResponse = chatModel.call(new Prompt(messages));
        content = chatResponse.getResult().getOutput().getText();
        System.out.println(content);
        System.out.println("======");

        messages.add(new AssistantMessage(content));

        //第三轮对话
        messages.add(new UserMessage("那如果主要是针对女性玩家的游戏呢?有什么需要改进的？"));
        chatResponse = chatModel.call(new Prompt(messages));
        content = chatResponse.getResult().getOutput().getText();
        System.out.println(content);
        System.out.println("======");

    }



}
