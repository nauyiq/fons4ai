package com.fons.cloud.ai;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import jakarta.annotation.PostConstruct;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import reactor.core.publisher.Flux;

import java.math.BigDecimal;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * 结构化输出
 * @author hongqy
 * @date 2026/3/2
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class StructuredOutputTest {

    @Autowired
    private DashScopeChatModel chatModel;

    private ChatClient chatClient;

    @PostConstruct
    public void init() {
        this.chatClient = ChatClient.builder(chatModel)
                .defaultAdvisors(new SimpleLoggerAdvisor())
                // 默认操作
                .defaultOptions(DashScopeChatOptions.builder()
                        .temperature(0.7)
                        .build())
                .build();
    }

    @Test
    public void testJsonFormat() throws InterruptedException {
        BeanOutputConverter<Object> converter = new BeanOutputConverter<>(Object.class);
        // 借助BeanOutputConverter获取结构化输出的格式
        String format = converter.getFormat();

        PromptTemplate promptTemplate = PromptTemplate.builder()
                .template("请给我推荐基本AI相关的书, 以{format}格式输出")
                .build();

        Flux<String> flux = chatClient.prompt(promptTemplate.create(Map.of("format", format))).stream().content();

        flux
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(() -> System.out.println("\n--- 流式输出完成 ---"))
                .subscribe();

        // 防止程序过快退出
        Thread.sleep(1000000L);

    }

    @Test
    public void testBeanOutputConverter() {
        BeanOutputConverter<Book> converter = new BeanOutputConverter<>(Book.class);

        PromptTemplate promptTemplate = PromptTemplate.builder()
                .template("请给我推荐基本AI相关的书, 以{format}格式输出")
                .build();

        String content = chatClient.prompt(promptTemplate.create(Map.of("format", converter.getFormat()))).call().content();

        Book book = converter.convert(content);

        System.out.println(book);

    }

    @Test
    public void testConverter() {
        Book entity = chatClient.prompt("请给我推荐基本心理学相关的书").call().entity(Book.class);
        System.out.println(entity);

        List<Book> books = chatClient.prompt("请给我推荐基本心理学相关的书").call().entity(new ParameterizedTypeReference<>() {
        });
        System.out.println(books);
    }




    private record Book(String title, String author, BigDecimal price, @JsonPropertyDescription("以中文描述") String summary) {

    }



}
