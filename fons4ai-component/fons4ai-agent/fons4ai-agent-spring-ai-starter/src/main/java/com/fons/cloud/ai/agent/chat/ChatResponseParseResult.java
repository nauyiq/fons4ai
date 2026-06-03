package com.fons.cloud.ai.agent.chat;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.agent.common.response.ChunkResult;
import com.fons.cloud.ai.agent.utils.ThinkMessageParser;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.ResultCode;
import lombok.*;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.deepseek.DeepSeekAssistantMessage;

import javax.validation.constraints.NotNull;
import java.util.List;

/**
 * @author hongqy
 */
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class ChatResponseParseResult {

    /**
     * 分块结果
     */
    private List<ChunkResult> chunks;

    /**
     * 当前段落是否
     */
    @Builder.Default
    private boolean inThink = false;


    /**
     * 解析AI模型输出结果, 对于思考模型可能会有推理过程 因此要把推理过程输出
     * <pre>
     *     对于不同厂商的think模型存在差异
     *     1. deepseek，openai，等模型输出推理内容是放在独立的reasoning_content里面，
     *     而部分厂商是嵌入在正文里面的，比如MiniMax输出的内容是<think...>...</think/>等标签表示推理内容
     *     2. spring ai已经支持获取思考模型的思考过程， 但是像MiniMax这种内容还是需要自己解析
     * </pre>
     * @param response
     * @return
     */
    public static ChatResponseParseResult parseResult(@NotNull ChatResponse response, boolean inThink) {
        Assert.notNull(response, () -> BusinessRuntimeException.of(ResultCode.INVALID_DATA));
        Assert.notNull(response.getResult(), () -> BusinessRuntimeException.of(ResultCode.INVALID_DATA));
        Generation result = response.getResult();
        // 优先使用spring ai支持的方式获取思考内容
        String reasoning = result.getOutput() instanceof DeepSeekAssistantMessage ? ((DeepSeekAssistantMessage) result.getOutput()).getReasoningContent()
                : (String) result.getMetadata().get("reasoningContent");
        if (StringUtils.isNotBlank(reasoning)) {
            // 不为空则直接返回 则认为是标准的思考模型
            return ChatResponseParseResult.builder()
                    .chunks(List.of(ChunkResult.builder()
                            .text(result.getOutput().getText())
                            .reasoning(reasoning)
                            .build()))
                    .build();
        } else {
            // 为空的时候使用Think解析器解析
            ThinkMessageParser.ParseResult parse = ThinkMessageParser.parse(result.getOutput().getText(), inThink);
            return ChatResponseParseResult.builder()
                    .chunks(parse.segments().stream().map(segment -> ChunkResult.builder()
                            .text(segment.thinking() ? null : segment.content())
                            .reasoning(segment.thinking() ? segment.content() : null)
                            .build()).toList())
                    .inThink(parse.inThink())
                    .build();
        }

    }


}
