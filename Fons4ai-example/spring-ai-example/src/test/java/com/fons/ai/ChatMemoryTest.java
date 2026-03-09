package com.fons.ai;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import jakarta.annotation.PostConstruct;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

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
        // 会话记忆
        ChatMemory chatMemory = MessageWindowChatMemory.builder().maxMessages(2).build();

        this.chatClient = ChatClient.builder(chatModel)

                .defaultAdvisors(MessageChatMemoryAdvisor.builder(chatMemory).build())
                // 默认操作
                .defaultOptions(DashScopeChatOptions.builder()
                        .temperature(0.7)
                        .topP(0.7)
                        .build())
                .build();
    }

    @Test
    public void testShortMemory() {
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
    }

    @Test
    public void testShortMemoryBy_chat_memory_conversation_id() throws InterruptedException {
        CountDownLatch countDownLatch = new CountDownLatch(1);
        String chatId = "1111";
        chatClient.prompt().user("我叫hongqiyuan?").advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId)).stream()
                .content()
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(countDownLatch::countDown)
                        .subscribe();

        countDownLatch.await();
        System.out.println("第一次对话结束");

        countDownLatch = new CountDownLatch(1);
        chatClient.prompt().user("今天天气真好?").advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId)).stream()
                .content()
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(countDownLatch::countDown)
                .subscribe();

        countDownLatch.await();
        System.out.println("第二次对话结束");

        countDownLatch = new CountDownLatch(1);
        chatClient.prompt().user("你还记得我是谁吗?").advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, chatId)).stream()
                .content()
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(countDownLatch::countDown)
                .subscribe();

        countDownLatch.await();
        System.out.println("第三次对话结束");


    }




}
