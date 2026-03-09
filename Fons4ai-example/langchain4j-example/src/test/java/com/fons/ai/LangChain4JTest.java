package com.fons.cloud.ai;

import cn.hutool.core.lang.UUID;
import dev.langchain4j.agent.tool.*;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.service.tool.DefaultToolExecutor;
import dev.langchain4j.service.tool.ToolExecutor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * LLM应用开发框架 LangChain4J
 * <pre>
 *   LangChain 是一个非常知名的开源框架，他的主要作用是简化基于大型语言模型构建应用程序的过程
 *  LangChain是py的LLM应用开发框架，Java的LangChain4J是其Java实现。
 * </pre>
 * @author hongqy
 * @date 2026/2/28
 */

@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class LangChain4JTest {

    @Autowired
    private OpenAiChatModel openAiChatModel;
    @Autowired
    private OpenAiStreamingChatModel openAiStreamingChatModel;

    @Test
    public void testOpenAiChatModel() {
        System.out.println(openAiChatModel.chat("你是谁?"));
    }

    @Test
    public void testOpenAiStreamingChatModel() throws InterruptedException {
        Flux<String> flux = Flux.create(fluxSink -> {
            openAiStreamingChatModel.chat("你是谁?", new StreamingChatResponseHandler() {
                @Override
                public void onPartialResponse(String partialResponse) {
                    fluxSink.next(partialResponse);
                }

                @Override
                public void onCompleteResponse(ChatResponse completeResponse) {
                    fluxSink.complete();
                }

                @Override
                public void onError(Throwable error) {
                    fluxSink.error(error);
                }
            });
        });

        CountDownLatch countDownLatch = new CountDownLatch(1);
        flux
                .delayElements(Duration.ofMillis(50)) // 添加延迟增强流式效果
                .doOnNext(System.out::print)
                .doOnComplete(countDownLatch::countDown)
                .subscribe();
        countDownLatch.await();

    }

    @Test
    public void testLowLevelApi() {
        // 低层次API整合测试

        // 1. 创建一个消息列表来存储对话消息， 用于对话记忆
        List<ChatMessage> messages = new ArrayList<>();

        //2.构造用户提示词
        UserMessage userMessage = UserMessage.from("2025年11月11日，深圳的气温怎样？");
        messages.add(userMessage);

        List<ToolSpecification> toolSpecifications = ToolSpecifications.toolSpecificationsFrom(TemperatureTools.class);

        //3. 创建ChatRequest，并指定工具列表
        ChatRequest request = ChatRequest.builder()
                .messages(userMessage)
                // 指定并定义工具列表
                .toolSpecifications(toolSpecifications )
                .toolChoice(ToolChoice.AUTO)
                .build();

        //4. 调用模型
        ChatResponse chatResponse = openAiChatModel.chat(request);
        AiMessage aiMessage = chatResponse.aiMessage();
        messages.add(aiMessage);

        //5.执行工具
        List<ToolExecutionRequest> toolExecutionRequests = chatResponse.aiMessage().toolExecutionRequests();
        toolExecutionRequests.forEach(toolExecutionRequest -> {
            ToolExecutor toolExecutor = new DefaultToolExecutor(new TemperatureTools(), toolExecutionRequest);
            System.out.println("execute tool " + toolExecutionRequest.name());
            String result = toolExecutor.execute(toolExecutionRequest, UUID.fastUUID().toString());
            ToolExecutionResultMessage toolExecutionResultMessages = ToolExecutionResultMessage.from(toolExecutionRequest, result);
            //6.把工具执行结果添加到chatMessages中
            messages.add(toolExecutionResultMessages);
        });

        //7.重新构造ChatRequest，并使用之前的对话chatMessages，以及指定toolSpecifications
        ChatRequest finalRequest = ChatRequest.builder()
                .messages(messages)
                .toolSpecifications(toolSpecifications)
                .build();

        System.out.println(openAiChatModel.chat(finalRequest).aiMessage().text());


    }

    private class TemperatureTools {
        @Tool(value = "Get temperature by city and date", name = "getTemperatureByCityAndDate")
        public String getTemperatureByCityAndDate(@P("city for get Temperature") String city, @P("date for get Temperature") String date) {
            System.out.println("getTemperatureByCityAndDate invoke...");
            return "23摄氏度";
        }
    }




}
