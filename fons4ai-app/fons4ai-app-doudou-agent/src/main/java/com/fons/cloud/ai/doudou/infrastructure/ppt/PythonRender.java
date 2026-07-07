package com.fons.cloud.ai.doudou.infrastructure.ppt;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.FileNameUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.util.StrUtil;
import com.fons.cloud.ai.doudou.common.constants.DouDouAgentResultCode;
import com.fons.cloud.ai.doudou.domain.entity.AiPptInst;
import com.fons.cloud.ai.doudou.domain.entity.AiPptTemplate;
import com.fons.cloud.ai.doudou.domain.service.AiPptTemplateDomainService;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.file.api.OssStoreService;
import com.fons.cloud.file.common.request.OssObjectRequest;
import com.fons.cloud.file.common.request.OssUploadRequest;
import com.fons.cloud.file.common.response.OssObjectResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Python-PPT渲染器, 要求运行环境中支持Python3
 * @author hongqy
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class PythonRender {
    private static final String PYTHON_SCRIPT_RESOURCE = "python/render_ppt.py";

    private final OssStoreService ossStoreService;
    private final AiPptTemplateDomainService aiPptTemplateDomainService;

    /**
     * 可访问的Python脚本路径
     */
    private String accessScriptPath;

    /**
     * 模板路径缓存
     */
    private Map<Long, String> templatePathMap = new ConcurrentHashMap<>();

    /**
     * 根据PPT实例渲染PPT， 由调用方自己保证schema已经处理完成
      * @param inst
     * @return
     */
    public String render(AiPptInst inst) throws Exception {
        log.info("开始渲染PPT， instId={}", inst.getId());
        // 获取模板
        AiPptTemplate pptTemplate = aiPptTemplateDomainService.findByTemplateCode(inst.getTemplateCode());
        Assert.notNull(pptTemplate, () -> BusinessRuntimeException.of(DouDouAgentResultCode.PPT_TEMPLATE_NOT_EXIST));

        String pptSchema = inst.getPptSchema();

        // 获取Python脚本路径
        String pyScriptPath = getPyScriptPath();
        // 获取模板文件路径
        String templateFilePath = getTemplateFilePath(pptTemplate);
        // ppt渲染后的输出目录
        String outputDir = getOutputDir();

        String outputFileName = "ppt_" + inst.getId() + "_" +
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")) + ".pptx";

        String outputFilePath = outputDir + File.separator + outputFileName;

        File templateFile = new File(templateFilePath);
        if (!templateFile.exists()) {
            throw new RuntimeException("模板文件不存在: " + templateFilePath);
        }

        // ---------- 构建命令 ----------
        List<String> command = List.of(
                "python",
                pyScriptPath,
                "--template", templateFilePath,
                "--output", outputFilePath
        );

        log.info("执行Python命令: {}", String.join(" ", command));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);

        Map<String, String> env = pb.environment();

        env.put("PYTHONIOENCODING", "utf-8");

        // ---------- 处理 JSON 传递 ----------
        // Windows 环境变量长度有限（32KB），大 JSON 会失败
        // 超过 20KB 自动写入临时文件
        if (pptSchema.length() > 20000) {

            Path tempFile = Files.createTempFile("ppt_schema_", ".json");
            Files.writeString(tempFile, pptSchema, StandardOpenOption.TRUNCATE_EXISTING);

            env.put("PPT_SCHEMA_FILE", tempFile.toAbsolutePath().toString());
            log.info("JSON过大，使用临时文件传递: {}", tempFile);

        } else {
            env.put("PPT_SCHEMA", pptSchema);
        }

        // ---------- 启动 ----------
        Process process = pb.start();

        // ---------- 读取输出 ----------
        StringBuilder output = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {

            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                log.info("Python输出: {}", line);
            }
        }

        // ---------- 等待（最多5分钟） ----------
        long timeoutMs = 5 * 60 * 1000L;
        long startTime = System.currentTimeMillis();
        boolean finished = false;

        while (System.currentTimeMillis() - startTime < timeoutMs) {
            try {
                int exitCode = process.exitValue();
                // 如果能获取到退出码，说明进程已结束
                finished = true;
                if (exitCode != 0) {
                    log.error("Python执行失败: {}", output);
                    throw new RuntimeException("Python脚本执行失败:\n" + output);
                }
                break;
            } catch (IllegalThreadStateException e) {
                // 进程还在运行，继续等待
                Thread.sleep(1000);
            }
        }

        if (!finished) {
            process.destroyForcibly();
            throw new RuntimeException("Python执行超时");
        }

        int exitCode = process.exitValue();

        if (exitCode != 0) {
            log.error("Python执行失败: {}", output);
            throw new RuntimeException("Python脚本执行失败:\n" + output);
        }

        // ---------- 检查输出 ----------
        File outputFile = new File(outputFilePath);
        if (!outputFile.exists()) {
            throw new RuntimeException("PPT未生成: " + outputFilePath);
        }

        // ---------- 上传到MinIO ----------
        log.info("PPT生成成功，开始上传到MinIO");
        byte[] fileBytes = Files.readAllBytes(outputFile.toPath());

        // 构建OSS对象名称: ppt/{conversationId}/{filename}
        String objectName = "ppt/" + inst.getConversationId() + "/" + outputFileName;
        OssObjectResponse response = ossStoreService.upload(OssUploadRequest.builder()
                .objectKey(objectName)
                .inputStream(new ByteArrayInputStream(fileBytes))
                .build());

        log.info("PPT已上传到oss: {}", response.getAccessUrl());

        // ---------- 删除本地文件 ----------
        try {
            Files.deleteIfExists(outputFile.toPath());
            log.info("本地PPT文件已删除: {}", outputFilePath);
        } catch (Exception e) {
            log.warn("删除本地文件失败: {}", outputFilePath, e);
        }

        return response.getAccessUrl();
    }




    private String getPyScriptPath() {
        if (StringUtils.isNotBlank(accessScriptPath) && FileUtil.exist(accessScriptPath)) {
            return accessScriptPath;
        }

        try {
            // 从classpath下将py脚本copy到全局可执行的目录
            ClassPathResource resource = new ClassPathResource(PYTHON_SCRIPT_RESOURCE);
            Path tempFile = Files.createTempFile("render_ppt_", ".py");
            try (InputStream is = resource.getInputStream()) {
                Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
            }
            tempFile.toFile().deleteOnExit();
            accessScriptPath = tempFile.toAbsolutePath().toString();
            log.info("Python脚本已提取到临时文件: {}", accessScriptPath);
            return accessScriptPath;
        } catch (Exception e) {
            log.error("Python脚本提取失败", e);
            throw new RuntimeException(e);
        }

    }

    private String getTemplateFilePath(AiPptTemplate pptTemplate) {
        String templatePath = templatePathMap.get(pptTemplate.getId());
        if (StringUtils.isNotBlank(templatePath) && FileUtil.exist(templatePath)) {
            return templatePath;
        }

        String accessObjectKey = pptTemplate.getFilePath();
        Assert.notEmpty(accessObjectKey, () -> BusinessRuntimeException.of(DouDouAgentResultCode.PPT_TEMPLATE_NOT_EXIST));

        // 不存在时需要将模板文件从OSS下载到本地
        try {
            OssObjectResponse response = ossStoreService.download(OssObjectRequest.builder().objectKey(accessObjectKey).build());
            String objectKey = response.getObjectKey();
            String suffix = FileNameUtil.getSuffix(objectKey);
            Path tempFile = Files.createTempFile( pptTemplate.getId().toString() + StrUtil.DASHED + "doudou-ppt-template", "." + suffix);
            String path = tempFile.toAbsolutePath().toString();
            FileUtil.writeFromStream(response.getInputStream(), path);
            templatePathMap.put(pptTemplate.getId(), path);
            return path;
        } catch (Exception e) {
            log.error("下载PPT模板文件失败， accessObjectKey={}", accessObjectKey);
            throw BusinessRuntimeException.of(DouDouAgentResultCode.PPT_TEMPLATE_NOT_EXIST.getCode(), e);
        }
    }

    private String getOutputDir() {
        String projectRoot = System.getProperty("user.dir");
        String outputDir = projectRoot + File.separator + "output" + File.separator + "ppt";
        try {
            Path path = Paths.get(outputDir);
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (Exception e) {
            log.error("创建输出目录失败: {}", outputDir, e);
        }
        return outputDir;
    }

}

