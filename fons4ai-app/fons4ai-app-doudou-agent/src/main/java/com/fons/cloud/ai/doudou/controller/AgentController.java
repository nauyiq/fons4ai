package com.fons.cloud.ai.doudou.controller;

import com.fons.cloud.ai.doudou.application.AgentApplicationService;
import com.fons.cloud.ai.doudou.common.dto.ChatRequest;
import com.fons.cloud.auth.utils.AuthUtils;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.StringUtils;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

/**
 * 智能体控制器
 * @author hongqy
 */
@RestController
@RequiredArgsConstructor
@RequestMapping("/doudou/agent")
@Tag(name = "智能体管理", description = "提供网页搜索、文件问答和PPT生成的流式接口")
public class AgentController {
    private final AgentApplicationService agentApplicationService;

    @Operation(summary = "智能问答", description = "接收用户查询并返回流式响应，使用联网搜索获取信息")
    @GetMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> chatStream(String query, String conversationId) {
        if (StringUtils.isAnyBlank(query, conversationId)) {
            return Flux.error(new IllegalArgumentException("查询参数不能为空"));
        }
        Long userId = AuthUtils.getCurrentUserId();
        return agentApplicationService.chatStream(ChatRequest.builder()
                        .question(query)
                        .conversationId(conversationId)
                        .userId(String.valueOf(userId))
                .build());
    }

}
