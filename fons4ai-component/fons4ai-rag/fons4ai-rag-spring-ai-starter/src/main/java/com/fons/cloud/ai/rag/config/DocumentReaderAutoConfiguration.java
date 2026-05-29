package com.fons.cloud.ai.rag.config;

import com.fons.cloud.ai.rag.document.reader.DocumentReaderStrategy;
import com.fons.cloud.ai.rag.document.reader.support.*;
import com.fons.cloud.ai.rag.document.reader.DocumentReaderFacade;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * @author hongqy
 */
@Configuration
public class DocumentReaderAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public DocumentReaderStrategy jsonReaderStrategy() {
        return new JsonReaderStrategy();
    }

    @Bean
    @ConditionalOnMissingBean
    public DocumentReaderStrategy markdownReaderStrategy() {
        return new MarkdownReaderStrategy();
    }

    @Bean
    @ConditionalOnMissingBean
    public DocumentReaderStrategy pdfReaderStrategy() {
        return new PdfReaderStrategy();
    }

    @Bean
    @ConditionalOnMissingBean
    public DocumentReaderStrategy textReaderStrategy() {
        return new TextReaderStrategy();
    }

    @Bean
    @ConditionalOnMissingBean
    public DocumentReaderStrategy documentReaderStrategy() {
        return new DocReaderStrategy();
    }

    @Bean
    DocumentReaderFacade documentReaderFacade(final List<DocumentReaderStrategy> strategiesList) {
        return new DocumentReaderFacade(strategiesList);
    }


}
