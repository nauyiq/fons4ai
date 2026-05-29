package com.fons.cloud.ai.rag.common.request;

import cn.hutool.core.map.MapUtil;
import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.annotation.JSONField;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.common.request.ParameterRequest;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.StringUtils;

import java.io.File;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * @author hongqy
 * @date 2026/3/11
 */
@Getter
@Setter
public class DocumentReaderRequest extends ParameterRequest {

    // ------ markdown 额外配置参数 start ------
    // 水平线分割生成新文档，默认false
    public static final String HORIZONTAL_RULE_CREATE_DOCUMENT = "horizontalRuleCreateDocument";
    // 是否包含代码块，默认false
    public static final String INCLUDE_CODE_BLOCK = "includeCodeBlock";
    // 是否包含引用，默认false
    public static final String INCLUDE_BLOCKQUOTE = "includeBlockquote";

    // ------ markdown 额外配置参数 end ------


    // ------ PDF额外配置参数 start ------
    // 忽略顶部N个单位的页眉, 默认50
    public static final String PAGE_TOP_MARGIN = "pageTopMargin";
    // 忽略底部N个单位的页脚，默认50
    public static final String PAGE_BOTTOM_MARGIN = "pageBottomMargin";
    // 每N页作为一个文档, 默认1
    public static final String PAGES_PER_DOCUMENT = "pagesPerDocument";
    // 每页再额外删掉前N行，默认0
    public static final String NUMBER_OF_TOP_TEXT_LINES_TO_DELETE = "numberOfTopTextLinesToDelete";
    // ------ PDF额外配置参数 end ------

    /**
     * 文档类型, 指定时则使用对应文档类加载策略， 为空则会根据文件类型进行文档类型选择
     */
    private DocumentType documentType;

    /**
     * 是否清理文档
     */
    private boolean cleanDocument;

    /**
     * 文件名
     */
    private String fileName;

    /**
     * 文件类型
     */
    private String fileType;

    /**
     * 文件
     */
    @JSONField(serialize = false)
    private InputStream inputStream;

    @Override
    public String toString() {
        return JSON.toJSONString(this);
    }

    private DocumentReaderRequest(DocumentType documentType, String fileType,  InputStream inputStream) {
        this.documentType = documentType;
        this.fileType = fileType;
        this.inputStream = inputStream;
    }

    public static DocumentReaderContextBuilder builder() {
        return new DocumentReaderContextBuilder();
    }

    public static class DocumentReaderContextBuilder {
        private DocumentType documentType;
        private String fileType;
        private String fileName;
        private Map<String, Object> params;
        private boolean cleanDocument;
        private InputStream inputStream;

        public DocumentReaderContextBuilder cleanDocument(boolean cleanDocument) {
            this.cleanDocument = cleanDocument;
            return this;
        }

        public DocumentReaderContextBuilder documentType(DocumentType documentType) {
            this.documentType = documentType;
            return this;
        }

        public DocumentReaderContextBuilder fileName(String fileName) {
            this.fileName = fileName;
            return this;
        }

        public DocumentReaderContextBuilder fileType(String fileType) {
            this.fileType = fileType;
            return this;
        }

        public DocumentReaderContextBuilder inputStream(InputStream inputStream) {
            this.inputStream = inputStream;
            return this;
        }

        public DocumentReaderContextBuilder param(String key, Object value) {
            if (params == null) {
                params = new HashMap<>();
            }
            params.put(key, value);
            return this;
        }

        public DocumentReaderContextBuilder params(Map<String, Object> params) {
            this.params = params;
            return this;
        }

        public DocumentReaderRequest build() {
            if (inputStream == null) {
                throw new UnsupportedOperationException("文件输入流为空");
            }
            if (StringUtils.isBlank(fileType)) {
                throw new UnsupportedOperationException("文件类型为空");
            }
            DocumentReaderRequest context = new DocumentReaderRequest(documentType, fileType, inputStream);
            if (MapUtil.isNotEmpty(this.params)) {
                context.addParameters(this.params);
            }
            context.setCleanDocument(this.cleanDocument);
            context.setFileName(this.fileName);
            return context;
        }

    }


}
