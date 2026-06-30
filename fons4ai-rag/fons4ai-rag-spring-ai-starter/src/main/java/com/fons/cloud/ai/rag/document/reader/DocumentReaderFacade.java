package com.fons.cloud.ai.rag.document.reader;

import cn.hutool.core.lang.Assert;
import cn.hutool.core.map.MapUtil;
import cn.hutool.extra.spring.SpringUtil;
import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.ai.rag.common.request.DocumentReaderRequest;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.SmartInitializingSingleton;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * @author hongqy
 */
@Slf4j
@Component
public class DocumentReaderFacade implements SmartInitializingSingleton {
    private final Map<DocumentType, DocumentReaderStrategy> strategies;

    public DocumentReaderFacade() {
        this.strategies = MapUtil.newHashMap();
    }

    public DocumentReaderFacade(final List<DocumentReaderStrategy> strategiesList) {
        this.strategies = MapUtil.newHashMap(strategiesList.size());
        strategiesList.forEach(strategy -> strategies.put(strategy.documentType(), strategy));
    }

    public List<Document> read(DocumentReaderRequest request) throws BusinessRuntimeException {
        try {
            // 1. 选择策略
            DocumentReaderStrategy usingStrategy = null;
            DocumentType documentType = request.getDocumentType();
            if (documentType != null) {
                usingStrategy = strategies.get(documentType);
            } else {
                for (DocumentReaderStrategy strategy : strategies.values()) {
                    if (strategy.isSupport(request)) {
                        usingStrategy = strategy;
                        break;
                    }
                }
            }
            // 2. 执行策略
            Assert.notNull(usingStrategy, "Not found strategy for document type: " + usingStrategy.documentType());
            log.info("Using strategy [{}] to read document, request:{}", usingStrategy.documentType(), request);
            return usingStrategy.read(request);
        } catch (BusinessRuntimeException e) {
            log.warn("Failed execute to read document, code:{}, message:{}", e.getCode(), e.getMessage(), e);
            throw BusinessRuntimeException.of(e);
        } catch (Exception e) {
            log.warn("Failed execute to read document, message:{}", e.getMessage(), e);
            throw BusinessRuntimeException.of(RagResultCode.FAILED_EXECUTED_READ_DOCUMENT.getCode(), e);
        }
    }


    @Override
    public void afterSingletonsInstantiated() {
        Map<String, DocumentReaderStrategy> beans = SpringUtil.getBeansOfType(DocumentReaderStrategy.class);
        if (MapUtil.isNotEmpty(beans)) {
            for (DocumentReaderStrategy strategy : beans.values()) {
                if (!strategies.containsKey(strategy.documentType())) {
                    strategies.put(strategy.documentType(), strategy);
                }
            }
        }


    }
}
