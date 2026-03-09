package com.fons.cloud.ai;

import cn.hutool.http.HttpRequest;
import cn.hutool.http.HttpResponse;
import cn.hutool.http.HttpUtil;
import org.junit.jupiter.api.Test;

/**
 * @author hongqy
 * @date 2026/2/28
 */
public class HttpClientCallerTest {

    // 百炼大模型KEY
    private static final String API_KEY = "sk-ec6470314d654bc3ae5cbadbad93e083";
    // 百炼大模型URL
    private static final String API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";

    /**
     * 使用HTTP发送请求，不使用流式处理
     */
    @Test
    public void testNoSteamSendClient() {
        // 测试使用请求体
        String requestBody = """
                {
                    "model": "qwen-plus",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": "你好，介绍下JAVA？"
                        }
                    ],
                    "stream": false
                }
                """;

        HttpRequest request = HttpUtil.createPost(API_URL);

        request
                // content-type
                .header("Content-Type", "application/json")
                // 设置API KEY
                .header("Authorization", "Bearer " + API_KEY)
                .header("X-DashScope-SSE", "enable")
                .body(requestBody);


        try (HttpResponse response = request.execute())  {
            String body = response.body();
            System.out.println(body);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }


    /**
     * 使用HTTP发送请求，使用流式处理
     */
    @Test
    public void testSteamSendClient() {
        // 测试使用请求体
        String requestBody = """
                {
                    "model": "qwen-plus",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": "你是谁?"
                        }
                    ],
                    "stream": true
                }
                """;

        HttpRequest request = HttpUtil.createPost(API_URL);

        request
                // content-type
                .header("Content-Type", "application/json")
                // 设置API KEY
                .header("Authorization", "Bearer " + API_KEY)
                .header("X-DashScope-SSE", "enable")
                .body(requestBody);


        try (HttpResponse response = request.execute())  {
            String body = response.body();
            System.out.println(body);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }





}
