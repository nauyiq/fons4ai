package com.fons.cloud.ai.agent.standard.adaptor;

import com.alibaba.cloud.ai.graph.RunnableConfig;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.checkpoint.BaseCheckpointSaver;
import com.alibaba.cloud.ai.graph.checkpoint.Checkpoint;

import java.util.Objects;

/** Alibaba 原生恢复的包内公共支持，不保存审批或 Agent 运行状态。 */
public final class AlibabaResumeSupport {
    private AlibabaResumeSupport() {
    }

    /** 校验会话与 thread 关联，并从指定 Saver 读取 checkpoint。 */
    public static ResumeCheckpoint load(AgentResumeRequest request,
                                        String expectedThreadId,
                                        BaseCheckpointSaver checkpointSaver,
                                        String missingMessage) {
        Objects.requireNonNull(request, "request cannot be null");
        Objects.requireNonNull(checkpointSaver, "checkpointSaver cannot be null");
        if (!Objects.equals(expectedThreadId, request.threadId())) {
            throw new IllegalArgumentException("conversationId does not match threadId");
        }
        RunnableConfig lookup = RunnableConfig.builder()
                .threadId(request.threadId())
                .checkPointId(request.checkpointId())
                .build();
        try {
            Checkpoint checkpoint = checkpointSaver.get(lookup)
                    .orElseThrow(() -> new IllegalArgumentException(missingMessage));
            return new ResumeCheckpoint(lookup, checkpoint);
        } catch (RuntimeException error) {
            throw error;
        } catch (Exception error) {
            throw new IllegalStateException("failed to load Alibaba checkpoint", error);
        }
    }

    /** 把已校验请求转换为 Alibaba HumanInTheLoopHook 的恢复配置。 */
    public static RunnableConfig feedbackConfig(ResumeCheckpoint resume,
                                                AgentResumeRequest request) {
        InterruptionMetadata source = HumanFeedbacks.fromCheckpoint(resume.checkpoint());
        InterruptionMetadata feedback = HumanFeedbacks.apply(source, request.action(),
                request.comment(), request.editedArguments());
        return RunnableConfig.builder(resume.lookup()).addHumanFeedback(feedback).build();
    }

    /** 已验证的恢复 checkpoint 与查找配置。 */
    public record ResumeCheckpoint(RunnableConfig lookup, Checkpoint checkpoint) {
        public ResumeCheckpoint {
            Objects.requireNonNull(lookup, "lookup cannot be null");
            Objects.requireNonNull(checkpoint, "checkpoint cannot be null");
        }
    }
}
