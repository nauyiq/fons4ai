package com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.support;

import cn.hutool.http.HttpUtil;
import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.infrastructure.service.ImageGenerationService;
import com.fons.cloud.ai.doudou.common.constants.PptInstStatus;
import com.fons.cloud.ai.doudou.common.dto.PPTScheme;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.infrastructure.ppt.strategy.PPTStateAgentStrategyContext;
import com.fons.cloud.ai.doudou.infrastructure.prompt.PPTAgentPrompt;
import com.fons.cloud.file.api.OssStoreService;
import com.fons.cloud.file.common.request.OssUploadRequest;
import com.fons.cloud.file.common.response.OssObjectResponse;
import jakarta.annotation.Resource;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.core.scheduler.Schedulers;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Schema生成策略
 *
 * @author hongqy
 */
@Slf4j
@Component
public class SchemaBuilderStrategy extends AbstractPPTStateAgentStrategy {

    @Resource
    private ImageGenerationService imageGenerationService;
    @Resource
    private OssStoreService ossStoreService;

    @Override
    protected void doExecute(PPTStateAgentStrategyContext ctx) {
        Sinks.Many<String> sink = ctx.getSink();
        AiPptInst inst = ctx.getInst();
        sink.tryEmitNext(createThinkingResponse("正在设计PPT详细内容...\n"));
        // 提示词
        String prompt = PPTAgentPrompt.getPptSchemePrompt(inst.getPptSchema(), inst.getOutline());
        // JSON转换器
        BeanOutputConverter<PPTScheme> converter = new BeanOutputConverter<>(new ParameterizedTypeReference<>() {
        });

        Disposable disposable = Mono.fromCallable(() -> {
                    // 请求LLM输出PPT内容
                    String called = ctx.getChatModel().call(prompt);
                    PPTScheme scheme = converter.convert(called);

                    // 处理图片生成
                    processImageGeneration(scheme, sink, ctx);

                    // 更新包含图片URL的schema
                    executeNext(ctx, inst::setPptSchema, JSON.toJSONString(scheme));

                    return null;
                })
                .doOnError(err -> {
                    log.error("Schema生成异常", err);
                    executeFailed(ctx, "Schema生成异常");
                })
                .subscribeOn(Schedulers.boundedElastic())
                .subscribe();

        // 保存 disposable 到任务管理器，用于停止任务
        setDisposable(ctx.getConversationId(), disposable);
    }

    /**
     * 处理图片生成
     * @param scheme
     * @param sink
     * @param ctx
     */
    private void processImageGeneration(PPTScheme scheme, Sinks.Many<String> sink, PPTStateAgentStrategyContext ctx) {
        if (scheme == null || CollectionUtils.isEmpty(scheme.getSlides())) {
            log.warn("PPT方案为空或图片列表为空");
            return;
        }
        // 首先收集所有需要生成图片的字段
        List<ImageGenerationTask> tasks = createImageGenerationTasks(scheme);
        if (CollectionUtils.isEmpty(tasks)) {
            return;
        }
        int total = tasks.size();
        sink.tryEmitNext(createThinkingResponse("✅PPT内容设计完成，开始生成图片素材\n"));
        sink.tryEmitNext(createThinkingResponse("共需生成 " + total + " 张图片，开始生成...\n"));

        if (CollectionUtils.isEmpty(tasks)) {
            log.info("没有需要生成图片的任务, conversationId:{}", ctx.getConversationId());
            return;
        }

        for (int i = 0; i < tasks.size(); i++) {
            ImageGenerationTask task = tasks.get(i);
            int current = i + 1;

            sink.tryEmitNext(createThinkingResponse("正在生成图片 (" + current + "/" + total + ")... \n"));

            try {
                // 调用图片生成服务
                String originalImageUrl = imageGenerationService.generateImage(task.prompt);

                // 下载图片并且上传到OSS
                String url = downloadImageFromUrlAndUploadOss(originalImageUrl, "ppt/" + ctx.getConversationId() + "/images/" + System.currentTimeMillis() + "_" + (i + 1) + ".png");
                task.fieldData.setUrl(url);
                sink.tryEmitNext(createThinkingResponse("✅ 图片生成完成 (" + current + "/" + total + ")\n"));
            } catch (Exception e) {
                log.error("图片生成或上传失败: {}", task.prompt, e);
                sink.tryEmitNext(createThinkingResponse("⚠ 图片生成失败 (" + current + "/" + total + "): \n" + task.key));
                // 使用空字符串
                task.fieldData.setUrl("");
            }
        }

        sink.tryEmitNext(createThinkingResponse("✅ 所有图片生成完成\n"));
        sink.tryEmitNext(createThinkingResponse("✅素材准备就绪，开始渲染PPT\n"));

    }

    private String downloadImageFromUrlAndUploadOss(String originalImageUrl, String objKey) {
        // 下载二进制
        byte[] bytes = HttpUtil.downloadBytes(originalImageUrl);
        if (bytes == null || bytes.length == 0) {
            throw new RuntimeException("图片下载失败");
        }

        // 将文件上传到MinIO
        OssUploadRequest request = OssUploadRequest.builder()
                .objectKey(objKey)
                .inputStream(new ByteArrayInputStream(bytes))
                .build();

        OssObjectResponse response = ossStoreService.upload(request);
        log.info("文件上传MinIO成功: {}", response.getObjectKey());
        return response.getAccessUrl();
    }



    /**
     * 创建图片生成任务
     * @param scheme
     * @return
     */
    private List<ImageGenerationTask> createImageGenerationTasks(PPTScheme scheme) {
        List<ImageGenerationTask> tasks = new ArrayList<>();

        for (PPTScheme.Slide slide : scheme.getSlides()) {
            if (slide.getData() == null) {
                continue;
            }

            for (Map.Entry<String, PPTScheme.FieldData> entry : slide.getData().entrySet()) {
                String key = entry.getKey();
                PPTScheme.FieldData fieldData = entry.getValue();
                if (fieldData == null) {
                    continue;
                }
                // 类型 只处理image和background类型
                String type = fieldData.getType();
                if (!"image".equalsIgnoreCase(type) && !"background".equalsIgnoreCase(type)) {
                    continue;
                }

                // 如果url已经有值，跳过
                if (StringUtils.isNotBlank(fieldData.getUrl())) {
                    continue;
                }

                // url为空，需要用content作为提示词生成图片
                String imageGenPrompt = fieldData.getContent();
                if (StringUtils.isBlank(imageGenPrompt)) {
                    continue;
                }

                tasks.add(new ImageGenerationTask(key, fieldData, imageGenPrompt));
            }
        }

        return tasks;
    }

    @AllArgsConstructor
    private static class ImageGenerationTask {
       private String key;
       private PPTScheme.FieldData fieldData;
       private String prompt;
    }

    @Override
    protected PptInstStatus nextStatus() {
        return PptInstStatus.RENDER;
    }

    @Override
    public PptInstStatus getStatus() {
        return PptInstStatus.SCHEMA;
    }
}
