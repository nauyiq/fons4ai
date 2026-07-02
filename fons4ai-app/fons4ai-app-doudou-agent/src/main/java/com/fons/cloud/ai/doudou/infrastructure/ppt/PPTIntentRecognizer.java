package com.fons.cloud.ai.doudou.infrastructure.ppt;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.common.constants.PptIntent;
import com.fons.cloud.ai.doudou.common.dto.ChatRequest;
import com.fons.cloud.ai.doudou.common.dto.PptIntentResult;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.service.AiPptInstDomainService;
import com.fons.cloud.ai.doudou.infrastructure.prompt.PPTAgentPrompt;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

/**
 * PPT 意图识别器
 * <pre>
 *     根据用户的请求识别是要创建PPT还是修改PPT 还是重新上传PPT
 * </pre>
 * @author hongqy
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class PPTIntentRecognizer {
    private final ChatModel chatModel;
    private final AiPptInstDomainService aiPptInstDomainService;

    private ChatClient chatClient;

    @PostConstruct
    public void init() {
        chatClient = ChatClient.builder(chatModel)
                // 意图识别提示词， 由大模型进行语义识别 判断用户是想创建PPT还是修改PPT
                .defaultSystem(PPTAgentPrompt.recognizePrompt())
                .build();
    }

    /**
     * 意图识别
     * @param request
     * @return
     */
    public PptIntentResult recognize(ChatRequest request) {
        Assert.notNull(request, "chatRequest must not be null");
        AiPptInst lastPptInst = aiPptInstDomainService.getLastPptInst(request.getConversationId());
        if (lastPptInst == null) {
            // 不存在PPT实例 用户需要生成PPT
            return new PptIntentResult(PptIntent.CREATE_PPT, "会话中无PPT实例，默认新建");
        }

        PptInstStatus status = lastPptInst.getStatusEnum();
        String errorMsg = lastPptInst.getErrorMsg();
        if (isNeedsResume(status, errorMsg, request.getQuestion())) {
            log.info("检测到断点重连需求: status={}, errorMsg={}", status, errorMsg);
            return new PptIntentResult(PptIntent.RESUME_PPT, "检测到上次执行未完成，从状态 " + status + " 继续执行");
        }

        // 如果是SUCCESS状态，调用LLM进行意图识别（CREATE_PPT 或 MODIFY_PPT）
        if (status == PptInstStatus.SUCCESS) {
            // 调用LLM进行意图识别
            return recognizeWithLLM(request.getQuestion());
        }

        // 对于其他中间状态（非失败），也默认为CREATE_PPT（新建）
        log.info("状态为 {}，默认新建", status);
        return new PptIntentResult(PptIntent.CREATE_PPT, "状态为 " + status + "，默认新建");
    }

    /**
     * 是否需要断点重连
     * @param status
     * @param errorMsg
     * @param question
     * @return
     */
    private boolean isNeedsResume(PptInstStatus status, String errorMsg, String question) {
        // 如果有错误信息，说明上次执行失败，需要重连
        if (StringUtils.hasText(errorMsg)) {
            return true;
        }

        // 检查用户是否明确表示要继续
        String lowerQuery = question.toLowerCase();
        String[] resumeKeywords = {"继续", "重试", "resume", "retry", "继续执行", "继续生成"};
        for (String keyword : resumeKeywords) {
            if (lowerQuery.contains(keyword)) {
                return true;
            }
        }

        // 对于中间状态（非SUCCESS、非INIT），如果用户没有明确要求新建，则继续
        if (status != PptInstStatus.SUCCESS && status != PptInstStatus.INIT) {
            // 检查用户是否明确要求新建
            String[] newKeywords = {"新建", "重新", "重新生成", "new", "create new"};
            for (String keyword : newKeywords) {
                if (lowerQuery.contains(keyword)) {
                    // 用户明确要新建
                    return false;
                }
            }
            // 默认继续
            return true;
        }
        return false;
    }

    /**
     * 调用LLM进行意图识别
     * @param question
     * @return
     */
    private PptIntentResult recognizeWithLLM(String question) {
        BeanOutputConverter<PptIntentResult> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {});
        try {
            String content = chatClient
                    .prompt()
                    .user("<question>" + question + "</question>")
                    .call().content();
            log.info("LLM-PPT意图识别响应: {}", content);
            return converter.convert(content);
        } catch (Exception e) {
            log.info("调用LLM进行意图识别时出错：{}", e.getMessage());
            return new PptIntentResult(PptIntent.CREATE_PPT, "意图识别失败，默认新建");
        }


    }

}
