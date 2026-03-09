package com.fons.cloud.ai;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import reactor.core.publisher.Flux;

import java.time.Duration;

/**
 * 阿里云百炼-灵积 测试类
 * DashScope实现了 org.springframework.ai.chat.model.
 * ChatModel就是专门和对话模型对接的一套接口。定义了与支持对话功能的语言模型交互的统一方式。
 * @author hongqy
 * @date 2026/2/28
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class ChatModelTest {

    @Autowired
    private DashScopeChatModel dashScopeChatModel;


    @Test
    public void testCall() {
        String called = dashScopeChatModel.call("你是谁?");
        System.out.println(called);

        Message sysMessage = new SystemMessage("你是一个翻译工具，请把用户的消息翻译成英文");
        Message userMessage = new UserMessage("什么样的生活才是幸福的？");
        System.out.println(dashScopeChatModel.call(sysMessage, userMessage));
    }

    @Test
    public void testCallByPrompt() {
        ChatOptions options = ChatOptions.builder()
                // 指定模型
                .model("deepseek-v3")
                // 温度 模型回答随机性
                .temperature(0.7)
                .build();

        Message sysMessage = new SystemMessage("你是一个AI助手，回答问题前请先说尊敬的主人");
        Message userMessage = new UserMessage("你是谁？告诉我什么样的生活才是幸福的？");

        // 提示词结构化
        Prompt prompt = new Prompt.Builder()
                .chatOptions(options)
                .messages(sysMessage, userMessage)
                .build();

        ChatResponse response = dashScopeChatModel.call(prompt);

        System.out.println(response.getResult().getOutput().getText());
    }


    @Test
    public void testCallByStream() throws InterruptedException {
        ChatOptions options = ChatOptions.builder()
                // 指定模型
                .model("deepseek-v3")
                // 温度 模型回答随机性
                .temperature(0.7)
                .build();

        Message sysMessage = new SystemMessage("你是一个高级JAVA架构，拥有20年开发经验");
        Message userMessage = new UserMessage("你对AI保有什么看法，我目前是一个五年JAVA开发工程师，可以分享一下该怎么学习AI或者转行做AI大语言应用开发可行吗？");

        // 提示词结构化
        Prompt prompt = new Prompt.Builder()
                .chatOptions(options)
                .messages(sysMessage, userMessage)
                .build();

        Flux<ChatResponse> flux = dashScopeChatModel.stream(prompt);
        flux
                .concatMap(response -> {
                    String text = response.getResult().getOutput().getText();
                    // 将每个字符转换为单独的流
                    return Flux.fromStream(text.chars()
                            .mapToObj(c -> String.valueOf((char) c)));
                })
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(() -> System.out.println("\n--- 流式输出完成 ---"))
                .subscribe();


        // 防止程序过快退出
        Thread.sleep(1000000L);
    }


}
