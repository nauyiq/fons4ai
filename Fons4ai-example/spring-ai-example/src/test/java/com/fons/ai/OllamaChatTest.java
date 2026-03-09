package com.fons.ai;

import jakarta.annotation.PostConstruct;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.concurrent.CountDownLatch;

/**
 * @author hongqy
 * @date 2026/3/4
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class OllamaChatTest {

    @Autowired
    private OllamaChatModel ollamaChatModel;

    private ChatClient chatClient;

    @PostConstruct
    public void init() {
        this.chatClient = ChatClient.builder(ollamaChatModel)
                .defaultOptions(ChatOptions.builder()
                        .temperature(0.7)
                        .build())
                .build();
    }


    @Test
    public void test() throws InterruptedException {
        Flux<String> content = this.chatClient.prompt("你是谁?").stream().content();
        CountDownLatch countDownLatch = new CountDownLatch(1);
        content
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(countDownLatch::countDown)
                .subscribe();
        countDownLatch.await();
    }



}
