package com.fons.cloud.ai.rag.document.reader.support;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.ai.rag.document.reader.AbstractDocumentReaderStrategy;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.markdown.MarkdownDocumentReader;
import org.springframework.ai.reader.markdown.config.MarkdownDocumentReaderConfig;
import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;

import java.util.List;

/**
 * @author hongqy
 */
public class MarkdownReaderStrategy extends AbstractDocumentReaderStrategy {

    @Override
    protected List<Document> doRead(DocumentReaderRequest request) {
        Boolean horizontalRuleCreateDocument = request.getBoolean(DocumentReaderRequest.HORIZONTAL_RULE_CREATE_DOCUMENT, false);
        Boolean includeCodeBlock = request.getBoolean(DocumentReaderRequest.INCLUDE_CODE_BLOCK, false);
        Boolean includeBlockquote = request.getBoolean(DocumentReaderRequest.INCLUDE_BLOCKQUOTE, false);
        // 读取配置
        MarkdownDocumentReaderConfig config = MarkdownDocumentReaderConfig.builder()
                // 水平线分割生成新文档
                .withHorizontalRuleCreateDocument(horizontalRuleCreateDocument)
                // 不包含代码块
                .withIncludeCodeBlock(includeCodeBlock)
                // 不包含引用
                .withIncludeBlockquote(includeBlockquote)
                // 添加文件名元数据
                .withAdditionalMetadata("filename", request.getFileName())
                .build();
        Resource resource = new InputStreamResource(request.getInputStream());
        return new MarkdownDocumentReader(resource,config).get();
    }

    @Override
    public DocumentType documentType() {
        return null;
    }
}
