package com.fons.cloud.ai.rag.infrastructure.document.reader;

import cn.hutool.core.io.FileTypeUtil;
import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.document.Document;

import java.io.File;
import java.util.*;

/**
 * @author hongqy
 * @date 2026/3/11
 */
public abstract class AbstractDocumentReaderStrategy implements DocumentReaderStrategy {

    @Override
    public final List<Document> read(DocumentReaderContext context) {
        // 1. 检查上下文
        checkContext(context);
        // 2. 读取文档
        List<Document> documents = doRead(context);
        // 3. 文档清洗
        return cleanDocuments(context, documents);
    }

    @Override
    public boolean isSupport(DocumentReaderContext context) {
        Assert.notNull(context, () -> BusinessRuntimeException.of(RagResultCode.INVALID_DOCUMENT_FILES.getCode(), "DocumentReaderContext should not be null"));
        List<File> files = context.getFiles();
        Assert.notEmpty(files, () -> BusinessRuntimeException.of(RagResultCode.INVALID_DOCUMENT_FILES.getCode(), "Document files should not be empty"));
        DocumentType documentType = documentType();
        return files.stream().allMatch(e -> documentType.match(FileTypeUtil.getType(e)));
    }

    protected void checkContext(DocumentReaderContext context) {
        List<File> files = context.getFiles();
        DocumentType documentType = context.getDocumentType();
        if (documentType != null) {
            Assert.isTrue(files.stream().allMatch(e -> documentType.match(FileTypeUtil.getType(e))));
        }
    }

    private List<Document> cleanDocuments(DocumentReaderContext context, List<Document> documents) {
        if (CollectionUtils.isEmpty(documents)) {
            return Collections.emptyList();
        }

        if (!context.isCleanDocument()) {
            return documents;
        }

        return doCleanDocuments(documents);
    }

    protected List<Document> doCleanDocuments(List<Document> documents) {
        return documents.stream().map(document -> {
            if (document == null || StringUtils.isBlank(document.getText())) {
                return null;
            }
            String text = document.getText();

            // 1. 去掉多余空白字符（空格、制表符、换行等）
            text = text.replaceAll("\\s+", " ").trim();

            // 2. 去掉无意义的乱码或特殊符号
            text = text.replaceAll("[^\\p{L}\\p{N}\\p{P}\\p{Z}\\n]", "");

            // 3. 可选：统一大小写
            // text = text.toLowerCase();

            // 4. 按换行拆分段落，去除重复段落
            String[] paragraphs = text.split("\\n+");
            Set<String> seen = new LinkedHashSet<>();
            for (String para : paragraphs) {
                String trimmed = para.trim();
                if (!trimmed.isEmpty()) {
                    seen.add(trimmed);
                }
            }

            text = String.join("\n", seen);

            return new Document(text, document.getMetadata());
        }).filter(Objects::nonNull).toList();
    }


    protected abstract List<Document> doRead(DocumentReaderContext context);

}
