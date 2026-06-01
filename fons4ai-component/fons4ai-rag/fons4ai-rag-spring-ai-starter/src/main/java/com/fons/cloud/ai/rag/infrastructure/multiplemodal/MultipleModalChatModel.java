package com.fons.cloud.ai.rag.infrastructure.multiplemodal;

import cn.hutool.core.io.IoUtil;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.config.MultipleModalConfigProperties;
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
import org.springframework.stereotype.Component;
import org.springframework.util.MimeTypeUtils;

import java.io.InputStream;
import java.util.List;

/**
 * 多模态模型能力
 * @author hongqy
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MultipleModalChatModel {
    private static final String DEFAULT_RECOGNIZE_REQUEST_MESSAGE = "请描述这张图片的内容，包括场景、对象、布局、颜色、文字信息，直接输出纯文本描述，不要多余说明。";

    private final MultipleModalConfigProperties properties;
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

    public String recognizeImage(InputStream imageStream) {
        return recognizeImage(IoUtil.readBytes(imageStream));
    }

    /**
     * 图片识别， 使用多模态model
     * @param imageBytes
     * @return
     */
    public String recognizeImage(byte[] imageBytes) {
        try {
            if (imageBytes == null || imageBytes.length == 0) {
                throw BusinessRuntimeException.of(RagResultCode.RECOGNIZE_IMAGE_FILE_IS_EMPTY);
            }
            // 用户提示词
            String recognizeUserMessage = StringUtils.isBlank(properties.getRecognizeUserMessage()) ? DEFAULT_RECOGNIZE_REQUEST_MESSAGE : properties.getRecognizeUserMessage();
            // 使用多模态模型识别图片
            ByteArrayResource imageResource = new ByteArrayResource(imageBytes);
            UserMessage userMessage = UserMessage.builder()
                    .text(recognizeUserMessage)
                    .media(List.of(new Media(MimeTypeUtils.IMAGE_PNG, imageResource)))
                    .build();
            ChatResponse response = openAiChatModel.call(new Prompt(userMessage));
            String resp = response.getResult().getOutput().getText();
            if (resp == null || resp.trim().isEmpty()) {
                return "[无法识别图片内容]";
            }
            return resp.trim();
        } catch (Exception e) {
            log.error("多模态图片识别异常", e);
            throw BusinessRuntimeException.of(RagResultCode.FAILED_EXECUTE_MULTIMODAL_IMAGE_RECOGNITION);
        }
    }

    /**
     * 获取多模态模型
     * @return
     */
    public OpenAiChatModel getModel() {
        return openAiChatModel;
    }


}
