package com.hqy.cloud.llm;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import jakarta.annotation.PostConstruct;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.Resource;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.Map;

/**
 * 提示词模板测试类
 * <pre>
 *     PromptTemplate，有一个默认渲染器（TemplateRenderer），用于将模板渲染成字符串。
 * </pre>
 * @author hongqy
 * @date 2026/3/2
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class PromptTemplateTest {

    @Autowired
    private ChatModel chatModel;

    private ChatClient chatClient;

    @PostConstruct
    public void init() {
        this.chatClient = ChatClient.builder(chatModel)
                .defaultSystem("你是有用的助手")
                .defaultAdvisors(new SimpleLoggerAdvisor())
                // 默认操作
                .defaultOptions(DashScopeChatOptions.builder()
                        .temperature(0.7)
                        .build())
                .build();
    }

    @Test
    public void testTemplate() throws InterruptedException {
        String template = "帮我推荐基本有关于{topic}的书籍";
        PromptTemplate promptTemplate = PromptTemplate.builder().template(template)
                .variables(Map.of("topic", "java")).build();

        Flux<String> flux = chatClient.prompt(promptTemplate.create())
                .stream().content();

        flux
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(() -> System.out.println("\n--- 流式输出完成 ---"))
                .subscribe();


        // 防止程序过快退出
        Thread.sleep(1000000L);
    }


    @Value("classpath:/templates/test_topic_prompt_template.st")
    private Resource testTopicPromptTemplate;

    @Test
    public void testTemplateByResources() throws InterruptedException {
        PromptTemplate promptTemplate = PromptTemplate.builder().resource(testTopicPromptTemplate)
                .variables(Map.of("topic", "java")).build();

        Flux<String> flux = chatClient.prompt(promptTemplate.create())
                .stream().content();

        flux
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(() -> System.out.println("\n--- 流式输出完成 ---"))
                .subscribe();


        // 防止程序过快退出
        Thread.sleep(1000000L);
    }


}
