package com.fons.cloud.ai.agent.infrastructure.service;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.constants.AgentResultCode;
import com.fons.cloud.ai.agent.constants.ImageGenProvider;
import com.fons.cloud.ai.agent.infrastructure.config.ImageGenerationProperties;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * 图片生成服务
 * <pre>
 *     调用LLM生成图片, 必须是支持生成图片的模型而非文本模型
 * </pre>
 * @author hongqy
 */
@Slf4j
@RequiredArgsConstructor
public class ImageGenerationService {
    private final ImageGenerationProperties properties;

    /**
     * 根据提示词生成图片
     * @param prompt 提示词
     * @return 图片的URL
     */
    public String generateImage(String prompt) {
        ImageGenProvider provider = properties.getProvider();
        if (provider == ImageGenProvider.QWEN) {
            // TODO 暂时写死千问 临时逻辑后续有需要再优化
            return generateWithQwen(prompt);
        }
        throw BusinessRuntimeException.of(AgentResultCode.NOT_SUPPORT_IMAGE_GEN_PROVIDER);
    }

    /**
     * 使用千问生成图片
     * @param prompt 提示词
     * @return 图片的URL
     */
    private String generateWithQwen(String prompt) {
        try {
            // 构建请求参数
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("model", properties.getModel());

            // input 使用 messages 格式
            Map<String, Object> textContent = new HashMap<>();
            textContent.put("text", prompt);

            Map<String, Object> userMessage = new HashMap<>();
            userMessage.put("role", "user");
            userMessage.put("content", new Object[]{textContent});

            Map<String, Object> input = new HashMap<>();
            input.put("messages", new Object[]{userMessage});
            requestBody.put("input", input);

            Map<String, Object> parameters = new HashMap<>();
            parameters.put("negative_prompt", "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。");
            parameters.put("prompt_extend", true);
            parameters.put("watermark", false);
            parameters.put("size", "1664*928");
            requestBody.put("parameters", parameters);

            // 创建HTTP请求
            HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                    .uri(URI.create(properties.getBaseUrl()))
                    .timeout(Duration.ofMinutes(5));

            // 添加请求头
            requestBuilder.header("Content-Type", "application/json");
            requestBuilder.header("Authorization", "Bearer " + properties.getApiKey());

            // 添加请求体
            String bodyStr = JSON.toJSONString(requestBody);
            requestBuilder.POST(HttpRequest.BodyPublishers.ofString(bodyStr));

            HttpRequest request = requestBuilder.build();

            // 发送请求
            HttpResponse<String> response = HTTP_CLIENT.send(request,
                    HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                JSONObject jsonResponse = JSON.parseObject(response.body());
                log.info("Qwen图像生成响应: {}", jsonResponse);

                // 从响应中直接获取图片URL
                JSONObject output = jsonResponse.getJSONObject("output");
                if (output != null && output.containsKey("choices")) {
                    com.alibaba.fastjson2.JSONArray choices = output.getJSONArray("choices");
                    if (choices != null && choices.size() > 0) {
                        JSONObject choice = choices.getJSONObject(0);
                        JSONObject message = choice.getJSONObject("message");
                        if (message != null && message.containsKey("content")) {
                            com.alibaba.fastjson2.JSONArray contents = message.getJSONArray("content");
                            if (contents != null && contents.size() > 0) {
                                JSONObject content = contents.getJSONObject(0);
                                if (content.containsKey("image")) {
                                    String imageUrl = content.getString("image");
                                    log.info("Qwen图像生成成功，URL: {}", imageUrl);
                                    return imageUrl;
                                }
                            }
                        }
                    }
                }
            } else {
                log.error("Qwen HTTP请求失败，状态码: {}, 响应: {}", response.statusCode(), response.body());
            }
        } catch (Exception e) {
            log.error("Qwen图像生成失败", e);
        }
        return null;
    }

    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
}
