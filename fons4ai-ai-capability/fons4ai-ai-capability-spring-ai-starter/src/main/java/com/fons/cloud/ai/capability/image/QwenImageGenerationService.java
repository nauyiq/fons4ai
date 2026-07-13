package com.fons.cloud.ai.capability.image;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.capability.config.ImageGenerationProperties;
import com.fons.cloud.ai.capability.constants.AiCapabilityResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.extern.slf4j.Slf4j;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * 千问图像生成实现。
 *
 * @author hongqy
 */
@Slf4j
public class QwenImageGenerationService implements ImageGenerationService {

    private final ImageGenerationProperties properties;
    private final HttpClient httpClient;

    public QwenImageGenerationService(ImageGenerationProperties properties) {
        this(properties, HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build());
    }

    QwenImageGenerationService(ImageGenerationProperties properties, HttpClient httpClient) {
        this.properties = properties;
        this.httpClient = httpClient;
    }

    @Override
    public String generateImage(String prompt) {
        if (properties.getProvider() == ImageGenProvider.QWEN) {
            return generateWithQwen(prompt);
        }
        throw BusinessRuntimeException.of(AiCapabilityResultCode.NOT_SUPPORT_IMAGE_GEN_PROVIDER);
    }

    private String generateWithQwen(String prompt) {
        try {
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("model", properties.getModel());

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

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(properties.getBaseUrl()))
                    .timeout(Duration.ofMinutes(5))
                    .header("Content-Type", "application/json")
                    .header("Authorization", "Bearer " + properties.getApiKey())
                    .POST(HttpRequest.BodyPublishers.ofString(JSON.toJSONString(requestBody)))
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                log.error("千问图像生成请求失败，状态码: {}", response.statusCode());
                return null;
            }
            String imageUrl = extractImageUrl(JSON.parseObject(response.body()));
            if (imageUrl != null) {
                log.info("千问图像生成成功");
            }
            return imageUrl;
        } catch (Exception e) {
            log.error("千问图像生成失败", e);
            return null;
        }
    }

    private String extractImageUrl(JSONObject jsonResponse) {
        JSONObject output = jsonResponse.getJSONObject("output");
        if (output == null) {
            return null;
        }
        JSONArray choices = output.getJSONArray("choices");
        if (choices == null || choices.isEmpty()) {
            return null;
        }
        JSONObject message = choices.getJSONObject(0).getJSONObject("message");
        if (message == null) {
            return null;
        }
        JSONArray contents = message.getJSONArray("content");
        if (contents == null || contents.isEmpty()) {
            return null;
        }
        return contents.getJSONObject(0).getString("image");
    }
}
