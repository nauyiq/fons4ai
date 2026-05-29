package com.fons.cloud.ai.doudou.controller;

import com.fons.cloud.ai.doudou.application.AiSessionApplicationService;
import com.fons.cloud.ai.doudou.common.constants.DouDouAgentResultCode;
import com.fons.cloud.ai.doudou.common.dto.PageQuerySessionsRequest;
import com.fons.cloud.ai.doudou.common.vo.MessageVO;
import com.fons.cloud.ai.doudou.common.vo.SessionDetailVO;
import com.fons.cloud.ai.doudou.common.vo.SessionInfoVO;
import com.fons.cloud.ai.doudou.domain.entity.AiSession;
import com.fons.cloud.ai.doudou.domain.service.AiSessionDomainService;
import com.fons.cloud.auth.utils.AuthUtils;
import com.fons.cloud.common.result.PageResult;
import com.fons.cloud.common.result.R;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 会话管理控制器
 * @author hongqy
 */
@RestController
@RequiredArgsConstructor
@RequestMapping("/doudou/session")
@Tag(name = "会话管理", description = "会话查询、列表、删除等接口")
public class SessionController {
    private final AiSessionDomainService sessionDomainService;
    private final AiSessionApplicationService aiSessionApplicationService;

    @GetMapping("list")
    @Operation(summary = "查询会话列表", description = "分页查询会话列表")
    public R<PageResult<SessionInfoVO>> pageQuerySessionList(
            @Parameter(description = "页码，默认1") @RequestParam(defaultValue = "1") Integer pageNum,
            @Parameter(description = "页大小，默认10") @RequestParam(defaultValue = "10") Integer pageSize) {
        
        // 构建请求参数
        PageQuerySessionsRequest request = PageQuerySessionsRequest.builder()
                .pageNum(pageNum)
                .pageSize(pageSize)
                .userId(String.valueOf(AuthUtils.getCurrentUserId()))
                .build();
        
        // 调用服务层
        PageResult<AiSession> pageResult = sessionDomainService.selectSessionListWithFirstRecord(request);
        
        // 转换为VO
        List<SessionInfoVO> sessionInfoVOList = pageResult.getResultList().stream()
                .map(session -> SessionInfoVO.fromAiSession(session, null))
                .toList();

        return R.ok(new PageResult<>(
                pageResult.getCurrentPage(),
                pageResult.getPageSize(),
                pageResult.getTotal(),
                sessionInfoVOList
        ));
    }

    @GetMapping("/{conversationId}")
    @Operation(summary = "查询会话详情", description = "根据conversationId查询会话详情")
    public R<SessionDetailVO> getSessionDetail(@PathVariable String conversationId) {
        List<AiSession> aiSessions = sessionDomainService
                .querySessionsBySessionId(conversationId, String.valueOf(AuthUtils.getCurrentUserId()));

        if (CollectionUtils.isEmpty(aiSessions)) {
            return R.failed(DouDouAgentResultCode.NOT_FOUND_SESSION);
        }


        // 从最新记录获取AGENT类型和文件ID
        String agentType = aiSessions.getFirst().getAgentType();
        String fileId = aiSessions.getFirst().getFileId();

        // 构建VO
        SessionDetailVO vo =
                SessionDetailVO.builder()
                .conversationId(conversationId)
                .agentType(agentType)
                .fileId(fileId)
                .messages(aiSessions.stream().map(MessageVO::fromAiSession).collect(Collectors.toList()))
                .build();

        return R.ok(vo);
    }

    @DeleteMapping("/{conversationId}")
    @Operation(summary = "删除会话", description = "删除指定会话及其关联数据（ai_file_info和ai_ppt_inst）")
    public R<Boolean> deleteSession(@PathVariable String conversationId) {
        // 获取会话
        AiSession aiSession = sessionDomainService.getLastOneBySessionId(conversationId, String.valueOf(AuthUtils.getCurrentUserId()));
        if (aiSession == null) {
            return R.failed(DouDouAgentResultCode.NOT_FOUND_SESSION);
        }
        return aiSessionApplicationService.deleteSession(aiSession);
    }


}
