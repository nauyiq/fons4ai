package com.fons.cloud.ai.agent.infrastructure.utils;

import com.fons.cloud.ai.agent.model.request.AgentInputContent;
import com.fons.cloud.ai.agent.model.request.AgentInputContentType;
import com.fons.cloud.ai.agent.model.request.AgentRequest;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.DataBlock;
import io.agentscope.core.message.Source;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.URLSource;
import io.agentscope.core.message.UserMessage;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * AgentScope消息转换器。
 *
 * <p>只负责common输入与AgentScope原生消息之间的转换，不读取AgentState，
 * 不处理运行状态和消息发送。</p>
 *
 * @author hongqy
 */
@Slf4j
public class AgentScopeMessageConverter {

    private static final AgentScopeMessageConverter INSTANCE = new AgentScopeMessageConverter();

    private AgentScopeMessageConverter() {
    }

    /**
     * 获取消息转换器单例。
     *
     * @return 消息转换器
     */
    public static AgentScopeMessageConverter getInstance() {
        return INSTANCE;
    }

    /**
     * 将common多模态输入转换为AgentScope用户消息。
     *
     * @param request Agent请求
     * @return AgentScope用户消息
     */
    public UserMessage createUserMessage(AgentRequest request) {
        try {
            List<ContentBlock> blocks = new ArrayList<>(request.getContents().size());
            for (AgentInputContent content : request.getContents()) {
                blocks.add(createContentBlock(content));
            }
            return new UserMessage(blocks);
        } catch (SystemIntervalException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            log.warn("Failed to convert common input to AgentScope UserMessage, runMessageId:{}",
                    request.getMessageId(), exception);
            throw SystemIntervalException.of("Failed to convert Agent multimodal input");
        }
    }

    /**
     * 将单个common输入内容转换为AgentScope内容块。
     *
     * @param content common输入内容
     * @return AgentScope内容块
     */
    private ContentBlock createContentBlock(AgentInputContent content) {
        if (content.getType() == AgentInputContentType.TEXT) {
            return TextBlock.builder()
                    .text(content.getText())
                    .build();
        }

        DataBlock.Builder builder = DataBlock.builder()
                .source(createDataSource(content));
        if (StringUtils.isNotBlank(content.getName())) {
            builder.name(content.getName());
        }
        return builder.build();
    }

    /**
     * 创建AgentScope多模态数据源。
     *
     * @param content common多模态输入内容
     * @return AgentScope数据源
     */
    private Source createDataSource(AgentInputContent content) {
        if (content.getUri() != null) {
            return URLSource.builder()
                    .url(content.getUri().toString())
                    .mimeType(content.getMimeType())
                    .build();
        }
        return Base64Source.builder()
                .mediaType(content.getMimeType())
                .data(Base64.getEncoder().encodeToString(content.getData()))
                .build();
    }

}
