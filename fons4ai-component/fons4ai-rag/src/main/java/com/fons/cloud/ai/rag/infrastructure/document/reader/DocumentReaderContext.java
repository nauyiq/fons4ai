package com.fons.cloud.ai.rag.infrastructure.document.reader;

import cn.hutool.core.map.MapUtil;
import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.annotation.JSONField;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.common.request.ParameterRequest;
import lombok.Getter;
import lombok.Setter;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author hongqy
 * @date 2026/3/11
 */
@Getter
@Setter
public class DocumentReaderContext extends ParameterRequest {

    private DocumentType documentType;

    private boolean cleanDocument;

    @JSONField(serialize = false)
    private List<File> files;

    @Override
    public String toString() {
        return JSON.toJSONString(this);
    }

    private DocumentReaderContext(DocumentType documentType, List<File> files) {
        this.documentType = documentType;
        this.files = files;
    }

    public static DocumentReaderContextBuilder builder() {
        return new DocumentReaderContextBuilder();
    }

    public static class DocumentReaderContextBuilder {
        private DocumentType documentType;
        private Map<String, Object> params;
        private boolean cleanDocument;

        public DocumentReaderContextBuilder cleanDocument(boolean cleanDocument) {
            this.cleanDocument = cleanDocument;
            return this;
        }

        public DocumentReaderContextBuilder documentType(DocumentType documentType) {
            this.documentType = documentType;
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

        public DocumentReaderContext build(File... files) {
            if (documentType == null) {
                documentType = DocumentType.TEXT;
            }
            DocumentReaderContext context = new DocumentReaderContext(documentType, List.of(files));
            if (MapUtil.isNotEmpty(this.params)) {
                context.addParameters(this.params);
            }
            context.setCleanDocument(this.cleanDocument);
            return context;
        }

    }


}
