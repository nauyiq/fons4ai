package com.fons.ai;

import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import jakarta.annotation.PostConstruct;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringExtension;

/**
 * 提示词结构化工程测试类
 * @author hongqy
 * @date 2026/2/28
 */
@ExtendWith(SpringExtension.class)
@SpringBootTest(classes = ExampleMain.class)
public class PromptEngineerTest {

    @Autowired
    private DashScopeChatModel chatModel;

    private ChatClient chatClient;

    @PostConstruct
    public void init() {
        this.chatClient = ChatClient.builder(chatModel)
                .defaultSystem("你是有用的助手")
                .defaultAdvisors(new SimpleLoggerAdvisor())
                // 默认操作
                .defaultOptions(DashScopeChatOptions.builder()
                        .model("deepseek-v3")
                        .topP(0.7)
                        .temperature(0.7)
                        .build())
                .build();
    }


    @Test
    public void role() {
        // 明确角色设定  为模型指定明确的角色，有助于其理解任务的上下文和预期输出
        String role = "你是一个毒舌博主，说话很噎人，请根据用户问题，怼他";

        System.out.println(this.chatClient.prompt().system(role)
                        .user("怎么学习JAVA？").call().content());
    }

    @Test
    public void fewShot() {
        // 在提示中加入少量示例，让模型通过这些示例学习思考模式和输出风格
        System.out.println(this.chatClient.prompt("""
                请根据用户输入的数字，给出结果，不需要思考过程，直接给出数字结果即可，推理过程参考：
                                1 = 5
                                2 = 10
                                3 = 15
                                ，如果用户给的不是个数字，请回复:无法回答，请输入数字
                """).user("9").call().content());

        System.out.println(chatClient.prompt().system("""
                请你根据用户输入的问题做改写，主要有以下改写策略：
                1、改写其中的错别字。
                2、做内容精简，帮用户的一堆废话精简成简单的一句话
                可以参考以下实例：
                
                Input：ni好
                Output ：{"错别字改写":"你好","内容精简":""}
                
                Input：我今天心情不错，我想知道今天是什么天气才让我心情这么好的？
                Output ：{"错别字改写":"","内容精简":"今天是什么天气？"}
                """).user("今天是不是天气很糟糕，为什么我新情很差").call().content());
    }

    @Test
    public void formatOutput() {
        // 结构化输出
        System.out.println(chatClient.prompt("请生成包括书名、作者和类别的三本虚构的、非真实存在的中文书籍清单，并以 JSON 格式提供，其中包含以下键:book_id、title、author、genre。")
                .system("你是一个富有创意的作家").
                user("帮我推荐几本AI相关的书籍").call().content());
    }

    @Test
    public void step() {
        // 指定步骤 让大模型一步一步执行
        System.out.println(chatClient.prompt().system(
                """
                        执行以下操作：
                                            1-用一句话概括下面文本。
                                            2-将摘要翻译成英语。
                                            3-在英语摘要中列出每个人名。
                                            4-输出一个 JSON 对象，其中包含以下键：english_summary，num_names。
                                            请用换行符分隔您的答案。
                        """
        ).user("韩立假扮厉飞雨，在虚天殿中从一群元婴修士中获取到了虚天鼎，并击杀了极阴老祖").call().content());

    }

    @Test
    public void cot() {
        // 思维链模式，  是一种通过引导模型“一步一步思考”来解决复杂问题的方法。它不是让模型直接给出最终答案，而是要求模型首先输出一个详细的、逻辑连贯的推理过程，然后再基于这个过程得出结论。

        System.out.println(chatClient.prompt("""
                    一个水果摊有5箱苹果，每箱重15公斤。今天卖掉了35公斤，还剩下多少公斤苹果？
                
                                    请一步一步思考，并给出最终答案。
                """).system("你是个ai").user("").call().content());


    }

}
