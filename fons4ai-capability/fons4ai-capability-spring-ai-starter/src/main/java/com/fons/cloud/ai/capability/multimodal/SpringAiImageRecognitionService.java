package com.fons.cloud.ai.capability.multimodal;

import cn.hutool.core.io.IoUtil;
import com.fons.cloud.ai.capability.config.MultimodalProperties;
import com.fons.cloud.ai.capability.constants.AiCapabilityResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeTypeUtils;

import java.io.InputStream;
import java.util.List;

/**
 * 基于 Spring AI OpenAI 兼容接口的图片识别实现。
 *
 * @author hongqy
 */
@Slf4j
@RequiredArgsConstructor
public class SpringAiImageRecognitionService implements ImageRecognitionService {

    private static final String DEFAULT_RECOGNIZE_REQUEST_MESSAGE =
            "请描述这张图片的内容，包括场景、对象、布局、颜色、文字信息，直接输出纯文本描述，不要多余说明。";

    private final MultimodalProperties properties;
    private OpenAiChatModel openAiChatModel;

    @PostConstruct
    public void init() {
        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .temperature(properties.getTemperature())
                .model(properties.getModel())
                .build();
        openAiChatModel = OpenAiChatModel.builder()
                .openAiApi(OpenAiApi.builder()
                        .baseUrl(properties.getBaseUrl())
                        .apiKey(properties.getApiKey())
                        .build())
                .defaultOptions(options)
                .build();
    }

    @Override
    public String recognizeImage(InputStream imageStream) {
        return recognizeImage(IoUtil.readBytes(imageStream));
    }

    @Override
    public String recognizeImage(byte[] imageBytes) {
        try {
            if (imageBytes == null || imageBytes.length == 0) {
                throw BusinessRuntimeException.of(AiCapabilityResultCode.RECOGNIZE_IMAGE_FILE_IS_EMPTY);
            }
            String userPrompt = StringUtils.defaultIfBlank(
                    properties.getRecognizeUserMessage(),
                    DEFAULT_RECOGNIZE_REQUEST_MESSAGE);
            UserMessage userMessage = UserMessage.builder()
                    .text(userPrompt)
                    .media(List.of(new Media(
                            MimeTypeUtils.IMAGE_PNG,
                            new ByteArrayResource(imageBytes))))
                    .build();
            ChatResponse response = openAiChatModel.call(new Prompt(userMessage));
            String text = response.getResult().getOutput().getText();
            return StringUtils.isBlank(text) ? "[无法识别图片内容]" : text.trim();
        } catch (Exception e) {
            log.error("多模态图片识别异常", e);
            throw BusinessRuntimeException.of(
                    AiCapabilityResultCode.FAILED_EXECUTE_MULTIMODAL_IMAGE_RECOGNITION);
        }
    }
}
