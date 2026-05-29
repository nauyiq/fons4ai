package com.fons.cloud.ai.rag.document.reader;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.document.Document;

import java.util.*;

/**
 * @author hongqy
 */
public abstract class AbstractDocumentReaderStrategy implements DocumentReaderStrategy {

    @Override
    public final List<Document> read(DocumentReaderRequest request) {
        // 1. 检查上下文
        checkContext(request);
        // 2. 读取文档
        List<Document> documents = doRead(request);
        // 3. 文档清洗
        return cleanDocuments(request, documents);
    }

    private List<Document> cleanDocuments(DocumentReaderRequest request, List<Document> documents) {
        if (CollectionUtils.isEmpty(documents)) {
            return Collections.emptyList();
        }

        if (!request.isCleanDocument()) {
            return documents;
        }

        return doCleanDocuments(documents);
    }

    protected void checkContext(DocumentReaderRequest request) {
        DocumentType documentType = request.getDocumentType();
        if (documentType != null) {
            Assert.isTrue(documentType.match(request.getFileType()));
        }
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

    protected abstract List<Document> doRead(DocumentReaderRequest request);
}
